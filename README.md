# Touchline

**What patterns contributed to this result, and what adjustments might have helped?** Touchline explores that question in a local Streamlit dashboard using Metrica Sports sample tracking and events. It presents measured patterns, source-linked moments and tactical hypotheses with tradeoffs. It cannot establish why a team lost or predict what a change would achieve.

## Explore the demo

The [featured analysis](docs/demo/featured-analysis.md) examines game 1's anonymous away team, which lost 3–0. Both perspectives of games 1–2 are supported; xT is fitted on the other game.

| View | What to look for | Screenshot |
|---|---|---|
| Overview | Complete result, goals, both-team comparisons and up to three findings | [Overview](docs/demo/overview.png) |
| Risk & progression | Buildup losses, turnover cost, completed entry routes and channel rates | [Risk & progression](docs/demo/risk-progression.png) |
| Players | Pass impact, buildup turnover count/cost, tracked minutes and nearest-opponent context | [Players](docs/demo/players.png) |
| Evidence | Source-event arrow, timestamp, player positions, trails, slider and optional control | [Evidence](docs/demo/evidence.png) |

[Watch the 54-second sampled browser walkthrough](docs/demo/walkthrough.mp4). See [capture notes and attribution](docs/demo/README.md) and [release validation](docs/VALIDATION.md).

## Windows setup

Install Python 3.12 and Git. In PowerShell, from this repository:

```powershell
python --version  # use Python 3.12
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard,demo]" -c requirements-demo.txt
.\.venv\Scripts\python.exe -m pip check
git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/raw/metrica
```

The data clone supplies `data/raw/metrica/data/Sample_Game_1` and `Sample_Game_2`; keep these external source CSVs out of Git. If that data directory already exists, reuse the complete local clone. `requirements-demo.txt` pins tested direct dependencies; it is **not a transitive lockfile**. The `demo` extra supplies media tooling and is included to reproduce this environment. No account or API key is required.

Launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally [localhost:8501](http://localhost:8501)). Select game 1, away and Overview for the featured example.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/validate_release.py data/raw/metrica
```

The release validator requires the real source files and exports all four match/team reports plus the featured source audit. Missing data must not be interpreted as a passing real-data gate. See [validation commands, observed outputs and limitations](docs/VALIDATION.md) and the [machine-readable report](docs/demo/release-validation.json).

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

Lower-level APIs remain available: `analytics.features.add_kinematics`, `analytics.xt.fit_xt` / `add_xt`, `analytics.pitch_control.pitch_control` / `zone_threat`, and `analytics.impact.pass_values` / `player_impact`. Fit xT on other matches when scoring player impact. Canonical file loading, experimental agent-tool adapters, action-aware metrics, constant-velocity forecasts and a CV adapter are development seams; operational chat and video inference are future work in the [roadmap](docs/ROADMAP.md).

## Interpretation and attribution

- Two anonymous sample matches are a sanity-check dataset, not a coaching benchmark. The minimum five relevant events and three supporting events are display safeguards, not statistical significance tests.
- Pass-only metrics miss carries and other actions. Failed pass endpoints are loss/interception locations, not intended targets. Nonentries can be useful circulation; channel counts do not establish defensive responsibility.
- Player eligibility defaults to 15 observed tracked minutes. Inferred goalkeepers appear separately by default. Single-match turnover cost is exploratory and can be dominated by a few events.
- Complete goals, including penalties, determine the displayed result. Modeling shots exclude penalties. Unknown or missing observations remain unavailable.
- Evidence is a **tracking reconstruction**, not match video. Simplified pitch control omits ball flight and acceleration and is not pass-success probability. Suggestions are hypotheses with tradeoffs, not causal claims or predicted score improvements.

Source data: [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data), provided by Metrica Sports for research and demonstration. Team and player identifiers remain anonymous. Screenshots and the walkthrough visualize those sample data; raw datasets are not redistributed in this repository. Check the source repository's current terms before further reuse or redistribution.
