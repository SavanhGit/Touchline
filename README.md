# Soccer Match Analysis Tool

Tracking-data analytics engine for soccer: movement prediction, zone and player risk, defensive weakness detection, with an LLM agent on top.

## Layout

```
src/soccer_tactics/
  schema.py       # the one tracking format every source converts into
  loaders/        # dataset -> schema (Metrica sample data first)
  analytics/
    features.py       # velocity, speed, acceleration (teleport glitches masked)
    xt.py             # expected threat grid fitted from pass and shot events
    pitch_control.py  # time-to-intercept control, and control x xT zone threat
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
```

## Validation

Reproduce with `python scripts/<name>.py data/raw/metrica`. Two games is a small sample; treat these as sanity checks, not benchmarks.

| Check | Result | Script |
|---|---|---|
| xT surfaces fitted on game 1 vs game 2 | Spearman 0.96 smoothed, 0.65 raw (shuffled cells: 0.11 at the 95th percentile) | `validate_xt_split_half.py` |
| Pitch control at a pass's end point, completed vs intercepted | AUC 0.964 [0.954, 0.972]; nearest-distance baseline 0.944 | `validate_pitch_control.py` |
| Zone-threat share 3 s before a shot, vs random moments | 91st percentile, but ball x alone gives 90th (paired diff +0.012, CI -0.019 to +0.042) | `validate_zone_threat.py` |

Read these with their limits in mind:
- Metrica events have no carries, so progress by dribbling is invisible to xT.
- 47 shots and 7 goals (penalties excluded) cannot support per-zone conversion, so shot value is the angle the goal subtends, calibrated to the observed goal count. Fit on more matches before trusting it.
- Pitch control is the simplified time-to-intercept form (no ball flight time). The pass AUC is partly by construction: a failed pass's end point is where an opponent got the ball. Velocity extrapolation is what beats the naive distance baseline.
- The zone-threat composite tracks territory but, at this sample size, does not beat ball position as a shot predictor. Its value is the spatial breakdown (which zones, which players), not a single score.

## Roadmap

1. Schema + loaders for tracking and events (done: Metrica sample games 1-2)
2. Features: velocity, acceleration (done), possession phases
3. Models: pitch control, expected threat (done), trajectory forecasting
4. Player impact: build-up detriment and on-field impact scores
5. Agent layer: natural-language queries backed by the analytics
6. CV pipeline: roboflow/sports -> same tracking schema
