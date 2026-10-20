# Validation notes

## Checkpoint baseline (2026-10-19)

Task 0 ran in the existing primary checkout on `codex/tactical-analyst-test`, starting at `c70357af3c5c7d97b064cedd198133f34cb8c7e9`. The branch already contained uncommitted V1 implementation; no isolated worktree was created for this task. Existing unrelated worktrees are not part of this validation.

The existing `.venv` is healthy and contains the optional dashboard dependencies. Both Metrica games have nonempty event, home-tracking and away-tracking CSVs under `data/raw/metrica/data/`. No dependency installation or dataset download was needed. The checkpoint preserves the existing implementation without changing product code.

From the repository root, `.\.venv\Scripts\python.exe -m pytest -q` completed with **78 passed in 69.46s (0:01:09)**, exit code 0, no skips, failures or reported warnings. This includes the real-data tests and Streamlit AppTest with both matches present.

The release script and numerical validation scripts listed later were not rerun during Task 0; their existing reports and numerical claims are preserved as prior evidence, not newly verified results. Full release verification still requires the optional dashboard dependencies and both real matches.

Validated locally on Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, Streamlit 1.66.0, Plotly 7.1.0, pytest 9.1.1 and SciPy 1.18.1.

## Real data

Metrica game 1: **Home 3–0 Away**, 3,278,383 tracking rows, 1,045 modeling events. Game 2: **Home 3–2 Away**, 3,188,704 tracking rows, 1,164 modeling events. Game 2 has one scored penalty; all five goals appear in match context while four appear in the filtered modeling shots.

Both matches: 25 Hz, periods 1 and 2, no duplicate player/object keys, no conflicting clocks, no gaps in the global frame sequence, and 100% modeling-event-to-player tracking coverage. Median source-event alignment is 0.331 m and 0.343 m; p90 is 0.506 m and 0.510 m. All modeling shots start in the attacking half.

There are 32,859 and 21,888 rows outside nominal pitch bounds. These include out-of-play ball/player positions; they are reported rather than silently discarded. Source loaders drop missing player positions, so a zero `missing_positions` count refers to retained canonical rows, not a guarantee that every player was observed at every frame. Per-player minutes count observed samples and may understate time during missing stretches.

## Reproduced baseline checks

| Check | Result |
|---|---|
| xT agreement between games | Spearman 0.646 raw, 0.956 with one smoothing pass |
| Pitch control at pass endpoint | AUC 0.964, bootstrap interval [0.954, 0.972] |
| Nearest-distance baseline | AUC 0.944 |
| Threat share vs ball position, 3 seconds before a shot | Paired difference +0.012, interval [-0.019, +0.042] |
| Outfield pass impact split-half reliability | Spearman 0.49, interval [0.17, 0.76] |
| Outfield detriment split-half reliability | Spearman 0.20, interval [-0.14, 0.50] |
| Outfield buildup turnover count split-half reliability | Spearman 0.39, interval [0.09, 0.63] |

These are sanity checks on two matches, not predictive benchmarks. Pass endpoints encode realized outcomes. The zone-threat composite has not demonstrated an advantage over ball position before shots. Tactical hypotheses have no causal or intervention validation.

The xT validation script previously labeled an already-smoothed fit as raw. It now passes `smooth=0` explicitly before comparing smoothing levels.

## Reproduce

From the repository root in PowerShell, after installing dependencies and downloading the data:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_release.py
.\.venv\Scripts\python.exe scripts/validate_xt_split_half.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_pitch_control.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_zone_threat.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_player_impact.py data/raw/metrica
```

`validate_release.py` checks all four match/team combinations and matches every displayed evidence reference back to the original event CSV. It exports the featured losing-team analysis and a JSON audit with quality, counts, training IDs, findings and moments under `docs/demo/`.

Synthetic tests cover counts, channel boundaries, away rotation, empty events, insufficient evidence, unknown scores, training leakage, multiple match IDs, repeated frames between halves, varying sampling rates, orientation of imported events, and unavailable tracking. Streamlit AppTest exercises match/team selection, finding and moment navigation, the time slider and pitch-control toggle.

The core suite can run without dashboard dependencies. Dashboard tests skip if Streamlit/Plotly or real matches are unavailable; release validation requires both real matches and does not substitute placeholder data.

## Controller browser QA (2026-10-19)

After the subagent browser transport limitation described below, the main session inspected the local app directly. Overview, Risk & progression, Players, and Evidence opened for both teams in both real matches (16 view checks), with the expected view headings and no visible exceptions. Match switching showed the selected match's evidence IDs and the correct home/away result context.

Desktop layouts were inspected at 1100 and 1440 pixels wide. Summary values and controls were readable, pitch proportions were retained, and no overlapping controls were observed. The evidence slider reached both -3 and +3 seconds and the pitch-control toggle rendered the displayed frame with its limitations. Four game-1-away screenshots are retained under `docs/demo/`. This checks presentation and navigation, not the causal validity of tactical hypotheses.

The existing `.streamlit/config.toml` was inspected and retained: it already supplies the dark theme, loopback-only server and disabled usage telemetry. No new configuration setting was needed.

## Dashboard resilience checkpoint (Task 5, 2026-10-19)

The focused dashboard run, `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -q`, passed **7 tests in 38.98s**, including the existing real-data navigation smoke test. After adding absent-root and zero-byte-file cases and an explicit NaN-rate fixture, the synthetic-only run, `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -q -k 'not navigation'`, passed **8 tests, 1 deselected in 9.18s**. Product code did not change between these runs. No full package suite was repeated.

Synthetic AppTest now covers absent source roots, empty match directories, a partial six-file download, zero-byte source files, unknown versus known 0–0 scores, unavailable rates (including NaN), no findings, no eligible outfield players or goalkeepers, missing evidence tracking, slider endpoints and the control toggle, and switching from three findings to none across match/team contexts. AppTest executes all four view bodies; it does not establish visual layout quality. The real smoke test exercises finding/moment selection, a slider change, control toggle and match switching.

Manual browser QA was attempted but **no viewport inspection was completed by the Task 5 implementer**. The local Streamlit server started at `http://127.0.0.1:8501/`; its health endpoint returned HTTP 200. The supported `cua_repl` browser inventory exposed an in-app browser, but creating a visible tab returned `IAB visibility is not supported in a subagent thread`. A background-tab attempt stalled and was aborted after approximately 299 seconds. The controller will perform the actual browser checks and record their results during the remaining Task 5/Task 7 verification. Readability, overlap and pitch proportions at roughly 1100 px and 1440 px remain unverified here. No screenshots, publication or site registration were performed.

## Original-source release audit (Task 6, 2026-10-19)

The release gate now verifies each displayed evidence item against its unique canonical event and the corresponding original Metrica row, including event semantics, player, team, period, source time and normalized coordinates after independently inferred half orientation. Failed passes must point to eligible `BALL LOST` rows; complete goals and penalty flags are compared with the raw event source. Audit failures raise `ValueError`, so optimized Python cannot remove the gate. Featured quality prose is generated from measured values, and non-finite unavailable JSON numbers become `null`.

The focused source-audit suite passed **21 tests**. The release command validated all four match/team reports and confirmed only the other match was used for training. Two unchanged-input exports produced identical SHA-256 hashes for `release-validation.json` and `featured-analysis.md`. The featured game-1 away report matches the saved dashboard findings: 22/76 left-channel buildup losses and 9/137 centre-start final-third entries; its overall cards show 61/238 buildup losses, 362/473 completed passes, 31 entries and 6 modeling shots. Fresh xT, pitch-control, zone-threat and player-impact checks all exited 0. Their measured results and runtime/source-data metadata are in [`docs/demo/validation-summary.md`](demo/validation-summary.md).
