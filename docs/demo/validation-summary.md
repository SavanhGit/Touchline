# Release validation run

Captured 2026-10-19. This run summary holds environment and run metadata separately from the deterministic `release-validation.json` export.

## Inputs and environment

- Source data repository Git SHA: `e706dd506b360d69d9d123d5b8026e7294b13996`
- Source code Git SHA: `262b1bc37f46a6623a87560959977a74b462444c`
- Python 3.12.14; NumPy 2.3.5; pandas 3.0.1; Streamlit 1.66.0; Plotly 7.1.0; pytest 9.1.1; SciPy 1.18.1
- Metrica games 1 and 2, local source CSVs under `data/raw/metrica`

## Commands and observed results

All commands ran from the repository root with the existing `.venv`; each exited 0.

| Command | Observed result |
|---|---|
| `\.venv\Scripts\python.exe -m pytest tests/test_release_validation.py -q` | 26 passed in 37.92s |
| `\.venv\Scripts\python.exe scripts/validate_release.py` | Validated 4 real team reports; exported `docs/demo` |
| `\.venv\Scripts\python.exe scripts/validate_xt_split_half.py data/raw/metrica` | xT surface Spearman 0.646 raw, 0.956 after one smoothing pass; shuffled-cell baseline mean -0.002, 95th percentile 0.110 |
| `\.venv\Scripts\python.exe scripts/validate_pitch_control.py data/raw/metrica` | 2,162 passes; velocity pitch-control AUC 0.964, bootstrap 95% interval [0.954, 0.972] |
| `\.venv\Scripts\python.exe scripts/validate_zone_threat.py data/raw/metrica` | At 3 seconds before shots, threat-share difference vs ball-x baseline +0.012, 95% interval [-0.019, +0.042] |
| `\.venv\Scripts\python.exe scripts/validate_player_impact.py data/raw/metrica` | Outfield split-half Spearman: impact p90 0.49 [0.17, 0.76], detriment p90 0.20 [-0.14, 0.50], buildup-turnover count p90 0.39 [0.09, 0.63] |

The featured away-team game-1 report has the complete score 3–0, game 2 as its only xT training match, 22/76 left-channel buildup losses, 9/137 successful final-third entries from centre starts, and 13/18 opponent shots through the opponent's centre channel. The saved overview dashboard shows the same two finding counts; its summary cards show 61/238 total buildup losses, 362/473 completed passes, 31 final-third entries, and 6 non-penalty modeling shots.

The release command was run twice with unchanged input. SHA-256 was identical across runs for both exports:

- `release-validation.json`: `3de837ec1831975dc24b2d9ae1a89924f59509901e51bc62c29367f973b55bf1`
- `featured-analysis.md`: `6e92614a204fb11f13e1a3d93fc1f8648871b0e7cc66d31d32abcc1c47249b9d`

Follow-up review added regressions requiring source-orientation evidence for every evidence period, checking win/draw prose, comparing the complete penalty-excluded modeling-shot identity set to raw `SHOT` rows, and serializing nested NaN/infinity values as JSON `null`. The focused suite passed 26 tests after these changes, and the release export still produced the same hashes.

Run timing is approximate and intentionally excluded from the deterministic JSON report.
