# Touchline

**What patterns contributed to this result, and what adjustments might have helped?** Touchline explores that question in a local Streamlit dashboard using Metrica Sports sample tracking and events. It presents measured patterns, source-linked moments and tactical hypotheses with tradeoffs. It cannot establish why a team lost or predict what a change would achieve.

## Explore the demo

The [featured analysis](docs/demo/featured-analysis.md) examines game 1's anonymous away team, which lost 3–0. Both perspectives of Games 1–3 are supported; xT is fitted on the other two available matches. The saved portfolio screenshots and walkthrough show the earlier two-game release.

| View | What to look for | Screenshot |
|---|---|---|
| Overview | Complete result, goals, both-team comparisons and up to three findings | [Overview](docs/demo/overview.png) |
| Risk & progression | Buildup losses, turnover cost, completed entry routes and channel rates | [Risk & progression](docs/demo/risk-progression.png) |
| Players | Pass impact, buildup turnover count/cost, tracked minutes and nearest-opponent context | [Players](docs/demo/players.png) |
| Evidence | Source-event arrow, timestamp, player positions, trails, slider and optional control | [Evidence](docs/demo/evidence.png) |

[Watch the 80-second edited browser walkthrough](docs/demo/walkthrough.mp4). See [capture notes and attribution](docs/demo/README.md) and [release validation](docs/VALIDATION.md).

## Roadmap

- **V1 — Complete (October 20, 2026):** the local Metrica-based analysis dashboard, evidence views and portfolio demo.
- **V2 — Complete (October 21, 2026):** seven audited IDSSE matches in an origin-only catalogue; synchronization, attack direction, pass endpoints and IDSSE xT/progression remain deferred.
- **V3 — Complete (October 21, 2026):** optional local question answering with Metrica report evidence and IDSSE origin-only citations.
- **V4 — In progress:** observed possessions, clipped transitions, recognized defensive actions, and shot counts/outcomes. Held-out source validation is recorded separately from tactical-effect validation.
- **Later:** derive tracking from recorded video, then work toward live analysis.

See the [full roadmap](docs/ROADMAP.md) for scope and validation details.

## Windows setup

Install Python 3.12 and Git. In PowerShell, from this repository:

```powershell
python --version  # use Python 3.12
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard,demo]" -c requirements-demo.txt
.\.venv\Scripts\python.exe -m pip check
git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/raw/metrica
```

The data clone supplies `data/raw/metrica/data/Sample_Game_1` and `Sample_Game_2`; keep all external source files out of Git. If that data directory already exists, reuse the complete local clone. `requirements-demo.txt` pins tested direct dependencies; it is **not a transitive lockfile**. The `demo` extra supplies media tooling and is included to reproduce this environment. No account or API key is required.

### Optional Metrica Game 3

Install the local EPTS adapter (Kloppy `>=3.17,<4`):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[metrica-epts]"
```

Reuse a complete local Metrica Sports sample-data source. Game 3 needs these three files together:

```text
data/raw/metrica/data/Sample_Game_3/
  Sample_Game_3_metadata.xml
  Sample_Game_3_tracking.txt
  Sample_Game_3_events.json
```

No loader downloads data. The catalogue exposes Game 3 only when its local source set and optional dependency are ready. Raw data stays under ignored `data/raw/`. The Sequences & shots tab presents observed possessions, 10-second transition windows, recognized defensive outcomes per 90 observed team-minutes, and channel shots/goals. Counts do not estimate shot quality or tactical effects.

### Optional local match assistant

Touchline's assistant uses a model served on your computer by [Ollama](https://ollama.com/download/windows). Install a model yourself; Touchline never downloads one automatically. The default is `qwen3:8b`:

```powershell
ollama pull qwen3:8b
```

Start Ollama, launch Touchline as usual, then use **Ask the local match assistant** below the selected match report. Chat requests go only to `http://localhost:11434`; no API key or paid model call is used. The analysis dashboard remains available when Ollama is stopped.

If Qwen3 8B responds too slowly on your computer, install a smaller model and select it for the current PowerShell session:

```powershell
ollama pull qwen3:4b
$env:TOUCHLINE_OLLAMA_MODEL = "qwen3:4b"
```

Metrica answers can cite selected-team report moments and player metrics. IDSSE answers can cite recorded event origins only; they cannot infer destinations, progression, tracking synchronization, attack direction, or xT. Chat history stays in the current Streamlit session and resets when the source, match, or team changes.

### Optional IDSSE match catalogue

IDSSE support uses the optional Floodlight parser. Download the seven match XML triplets from the [public IDSSE dataset record](https://api.figshare.com/v2/articles/28196177) and place the `matchinformation`, `events_raw`, and `positions_raw_observed` XML files together under the external `data/raw/idsse` directory. Keep all raw XML outside Git; `data/raw/` is ignored. From the repository root, install the parser and generate the source-quality manifest with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[idsse]"
.\.venv\Scripts\python.exe scripts/validate_idsse.py --data-root data/raw/idsse --output docs/validation/idsse-v1.json
```

The IDSSE selector requires a valid manifest covering all seven discovered matches and current fingerprints for every XML file. Only candidates that pass the source event-origin quality gate are selectable; rejected candidates remain excluded with their reasons recorded in the manifest. Changing, adding, or removing XML files requires regenerating the manifest. The origin-only findings retain source coordinates, timestamps, and event-code provenance. They do not establish tracking synchronization or attack direction; pass endpoints remain unknown, so IDSSE xT and progression are unavailable. No held-out IDSSE evaluation is claimed. See [validation notes](docs/VALIDATION.md) for the recorded audit and its limits.

IDSSE data is attributed to the **Deutsche Fußball Liga (DFL)** under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper citation: Bassek et al., “An integrated dataset of spatiotemporal and event data in elite soccer,” *Scientific Data* 12, 195 (2025), [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). The dataset and this project’s derived use remain subject to the linked source terms.

Launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally [localhost:8501](http://localhost:8501)). Select game 1, away and Overview for the featured example.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/validate_release.py data/raw/metrica --output docs/validation/v4
.\.venv\Scripts\python.exe scripts/validate_cross_match.py --metrica data/raw/metrica --output docs/validation/v4
```

The release validator requires the real source files and exports all six match/team reports, source/suggestion audits, the featured report, and six held-out team folds. Both commands run the same source gate; either one reproduces both JSON artifacts. Missing data must not be interpreted as a passing real-data gate. See [validation commands, observed outputs and limitations](docs/VALIDATION.md) and the [V4 machine-readable report](docs/validation/v4/release-validation.json).

## Analytical API

The report engine is independent of Streamlit:

```python
from soccer_tactics.loaders import load_metrica_bundle, load_metrica_game
from soccer_tactics.analytics.report import analyze_match

match = load_metrica_bundle("data/raw/metrica", game=1)
training = load_metrica_bundle("data/raw/metrica", game=2)
report = analyze_match(match, "away", [training])
tracking, events = load_metrica_game("data/raw/metrica", game=1)  # compatible tuple API
```

Lower-level APIs remain available: `analytics.features.add_kinematics`, `analytics.xt.fit_xt` / `add_xt`, `analytics.pitch_control.pitch_control` / `zone_threat`, and `analytics.impact.pass_values` / `player_impact`. Fit xT on other matches when scoring player impact. The optional match assistant uses context-bound tools and local Ollama inference; action-aware metrics, constant-velocity forecasts, a CV adapter, and video inference remain development work in the [roadmap](docs/ROADMAP.md).

## Interpretation and attribution

- Three anonymous sample matches are a sanity-check dataset, not a coaching benchmark. Home/away labels do not link team identity across matches; defensive comparisons use the other matches as a pooled dataset baseline. The minimum five relevant events and three supporting events are display safeguards, not statistical significance tests.
- Possessions contain observed events only. Transition windows require an observed opponent turnover or retained successful defensive action, end by 10 seconds or possession/period end, and miss unrecorded activity. Recognized defense includes only TACKLE-WON/LOST, INTERCEPTION, and THEFT source subtypes; excluded actions are counted in quality metadata.
- Pass-only metrics miss carries and other actions. Failed pass endpoints are loss/interception locations, not intended targets. Nonentries can be useful circulation; channel counts do not establish defensive responsibility.
- Player eligibility defaults to 15 observed tracked minutes. Inferred goalkeepers appear separately by default. Single-match turnover cost is exploratory and can be dominated by a few events.
- Complete goals, including penalties, determine the displayed result. Modeling shots exclude penalties. Unknown or missing observations remain unavailable.
- Evidence is a **tracking reconstruction**, not match video. Simplified pitch control omits ball flight and acceleration and is not pass-success probability. Suggestions are hypotheses with tradeoffs, not causal claims or predicted score improvements.

Source data: [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data), provided by Metrica Sports for research and demonstration. Team and player identifiers remain anonymous. Screenshots and the walkthrough visualize those sample data; raw datasets are not redistributed in this repository. Check the source repository's current terms before further reuse or redistribution.
