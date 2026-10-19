# Touchline

Touchline is a soccer match analysis project that explores match patterns through tracking data, evidence-backed findings, and tactical hypotheses.

## Layout

```
src/soccer_tactics/
  schema.py       # the one tracking format every source converts into
  loaders/        # dataset -> schema (Metrica sample data first)
  analytics/
    features.py       # velocity, speed, acceleration (teleport glitches masked)
    xt.py             # expected threat grid fitted from pass and shot events
    pitch_control.py  # time-to-intercept control, and control x xT zone threat
    impact.py         # per-player impact (xT added) and build-up detriment (turnover cost)
  agent/          # LLM tool-calling layer over analytics
  cv/             # phase 2: video -> tracking, must emit the same schema
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
git clone --depth 1 https://github.com/metrica-sports/sample-data data/raw/metrica
pytest
```

## Usage

```python
from soccer_tactics.loaders import load_metrica_game
from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.xt import fit_xt, add_xt
from soccer_tactics.analytics.pitch_control import pitch_control, zone_threat

tracking, events = load_metrica_game("data/raw/metrica", game=1)
grid = fit_xt(events)                      # 16x12 expected-threat surface
passes = add_xt(events, grid)              # xt_gain per pass; failed passes lose the xT they started with
frame = add_kinematics(tracking).query("frame == 20000 and team != 'ball'")
home_threat, away_threat = zone_threat(pitch_control(frame), grid)

# player scores: fit xT on *other* matches than the ones you score (a grid fitted on the same
# passes has seen them), so with two games, score each with a grid fitted on the other
from soccer_tactics.analytics.impact import pass_values, player_impact

other_events = load_metrica_game("data/raw/metrica", game=2)[1]
table = player_impact(pass_values(events, fit_xt(other_events), tracking), tracking, min_minutes=30)
outfield = table[~table.is_gk]                      # goalkeepers are flagged, not ranked
outfield.nlargest(5, "impact_p90")                  # most value created per 90
outfield.nlargest(5, "buildup_turnovers_p90")       # most build-up turnovers per 90
```

### Agent tools

The tool registry is provider-neutral: `specs()` produces Anthropic-shaped tool
definitions and `call()` dispatches validated analytics operations. Load the
tracking frame, events, fitted xT grid, and optional player table once:

```python
from soccer_tactics.agent.tools import call, set_state

set_state(tracking=tracking, events=events, xt_grid=fit_xt(other_events), impact=table)
call("rank_players", metric="buildup_turnovers_p90", n=5)
call("turnover_zones", team="home", n=5)
call("threat_map", match_id=1, time_s=42.0, n=10)
```

The agent tools expose pass-only impact, time-to-intercept pitch control, and
xT-weighted turnover costs. They do not turn the two-game validation sample
into a predictive benchmark: single-match detriment is noisy, and the threat
map is a spatial explanation rather than a shot-probability model.

## Extending the engine

New tracking sources can be converted once to the canonical schema and loaded
without changing downstream analytics:

```python
from soccer_tactics.loaders import load_canonical

tracking, events = load_canonical("tracking.parquet", "events.parquet")
```

The current extension points are intentionally conservative:

- `player_impact_with_actions` preserves pass metrics and adds shot totals plus
  xT-weighted defensive action value for `tackle`, `interception`, and
  `recovery` events. These values are explanatory until they are calibrated on
  more matches.
- `constant_velocity_forecast` is a transparent trajectory baseline. It
  establishes the prediction schema and evaluation seam before a learned model
  is introduced.
- `cv.detections_to_tracking` and `TrackingAdapter` define the video contract:
  a detector must emit pitch-mapped metre coordinates and stable team/player
  labels. Model-specific inference and camera calibration remain a later
  integration.

## Validation

Reproduce with `python scripts/<name>.py data/raw/metrica`. Two games is a small sample; treat these as sanity checks, not benchmarks.

| Check | Result | Script |
|---|---|---|
| xT surfaces fitted on game 1 vs game 2 | Spearman 0.96 smoothed, 0.65 raw (shuffled cells: 0.11 at the 95th percentile) | `validate_xt_split_half.py` |
| Pitch control at a pass's end point, completed vs intercepted | AUC 0.964 [0.954, 0.972]; nearest-distance baseline 0.944 | `validate_pitch_control.py` |
| Zone-threat share 3 s before a shot, vs random moments | 91st percentile, but ball x alone gives 90th (paired diff +0.012, CI -0.019 to +0.042) | `validate_zone_threat.py` |
| `impact_p90`, first vs second half (37 outfield players) | Spearman 0.49 [0.17, 0.76], permutation p = 0.002 | `validate_player_impact.py` |
| `detriment_p90`, first vs second half (37 outfield players) | Spearman 0.20 [-0.14, 0.50], p = 0.25: not distinguishable from noise |  same |
| `buildup_turnovers_p90`, first vs second half | Spearman 0.39 [0.09, 0.63], p = 0.022 |  same |

Read these with their limits in mind:
- Metrica events have no carries, so progress by dribbling is invisible to xT.
- 47 shots and 7 goals (penalties excluded) cannot support per-zone conversion, so shot value is the angle the goal subtends, calibrated to the observed goal count. Fit on more matches before trusting it.
- Pitch control is the simplified time-to-intercept form (no ball flight time). The pass AUC is partly by construction: a failed pass's end point is where an opponent got the ball. Velocity extrapolation is what beats the naive distance baseline.
- The zone-threat composite tracks territory but, at this sample size, does not beat ball position as a shot predictor. Its value is the spatial breakdown (which zones, which players), not a single score.
- Player scores count passes only, so tackles, interceptions and shots earn no credit. Impact is moderately tied to pass volume (outfield Spearman 0.47 with passes per 90).
- Within one match, build-up detriment is mostly noise: the top 10 of 210 build-up turnovers carry 28% of the total cost, and a typical player has about 5 of them. Rank on the turnover count, and use the xT cost to explain which losses hurt. Pool several matches before naming anyone (a rough Spearman-Brown extrapolation from a wide interval puts reliability 0.7 at around five matches).
- Goalkeepers are flagged with `is_gk` (the deepest player on each team). Their long kicks, lost far from any pressure, otherwise top the detriment ranking.

## Roadmap

1. Schema + loaders for tracking and events (done: Metrica sample games 1-2)
2. Features: velocity, acceleration (done), possession phases
3. Models: pitch control, expected threat (done), trajectory forecasting
4. Player impact: build-up detriment and on-field impact scores (done, passes only)
5. Agent layer: natural-language queries backed by the analytics
6. CV pipeline: roboflow/sports -> same tracking schema

The remaining phases now have working seams and baselines: canonical file
loading, action-aware impact, constant-velocity forecasting, and a CV adapter.
Dataset-specific loaders, learned forecasting, and video model calibration
still require external data and should be validated against ground truth before
being presented as production capabilities.

### IDSSE cross-match evaluation

IDSSE is the verified second source: seven full Bundesliga matches with paired
DFL XML event and TRACAB position files, published under CC BY 4.0. See
`docs/research/idsse-dataset.md` for attribution, source links, and download
details. The optional parser is installed with:

```bash
pip install -e ".[idsse]"
```

After downloading the XML archive into `data/raw/idsse`, list matches and run
leave-one-match-out evaluation:

```python
from soccer_tactics.loaders import list_idsse_matches, load_idsse_game
from soccer_tactics.analytics.cross_match import leave_one_match_out

ids = list_idsse_matches("data/raw/idsse")
matches = [load_idsse_game("data/raw/idsse", match_id) for match_id in ids]
passes, players = leave_one_match_out(matches)
```

Each held-out match is scored with an xT grid fitted only on the other matches,
so reported pass value does not train on the match it evaluates.

Before the archive is available, run the same evaluation flow against two
deterministic placeholder matches:

```bash
python scripts/validate_cross_match.py
```

When a valid IDSSE download is available, pass its directory and at least two
match ids to the same script. The evaluation code and output schema stay the
same:

```bash
python scripts/validate_cross_match.py data/raw/idsse DFL-MAT-... DFL-MAT-...
```
