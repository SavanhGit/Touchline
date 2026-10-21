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

## IDSSE source-origin catalogue (Task 4 audit)

The recorded real-data audit at `data/raw/idsse` accepted **7/7 matches** for source event-origin views and fingerprinted **21/21 XML files** (three file families per match). Every source score matched the complete scored-goal records, and the audit had no manifest errors or candidate rejection reasons. The machine-readable output is [`docs/validation/idsse-v1.json`](validation/idsse-v1.json). On 2026-10-21, a fresh controller audit again accepted 7/7 with 21/21 fingerprints, zero errors, and an exact parsed match to this checked-in manifest. Final Metrica release validation confirmed all four real team reports. The full suite passed **337 tests**. The dashboard's IDSSE selector and origin-only view were manually inspected.

The catalogue fails closed unless all seven manifest candidate IDs match all seven discovered XML matches and every XML file has its current fingerprint. Only accepted candidates can appear. The readiness scope is source event origins. IDSSE evidence carries source coordinates, timestamps, and event-code provenance; **tracking synchronization, attack direction, pass endpoints, IDSSE xT, progression, and held-out IDSSE evaluation remain deferred**. IDSSE tracking is audited for source structure, not shown as synchronized playback. Metrica endpoint reports and training remain unchanged.

Audit quality details: each period had a 25 Hz sampling estimate; source exclusions were counted by reason, including unsupported event codes, unknown pass evaluations, and three excluded penalty shots. `source_missing_positions` counts missing/nonfinite XY slots across all source object columns at observed frames, including roster slots never observed. It does not count missed active-player detections and must not be presented as player visibility or tracking coverage.

Peak memory is reported as **1600 MiB**, rounded up to the next 64 MiB, for the peak traced Python allocation upper bound. Native NumPy allocations and total process RSS are outside this measurement, so this is not a process-memory ceiling.

The source is the Deutsche Fußball Liga (DFL) IDSSE dataset, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Citation: Bassek et al., “An integrated dataset of spatiotemporal and event data in elite soccer,” *Scientific Data* 12, 195 (2025), [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). Keep the raw XML in the ignored `data/raw/idsse` directory, outside Git.

Reproduce the IDSSE audit and focused documentation/release checks from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/validate_idsse.py --data-root data/raw/idsse --output docs/validation/idsse-v1.json
.\.venv\Scripts\python.exe -m pytest tests/test_roadmap.py tests/test_release_validation.py -q
```

Task 4's focused validator tests passed **31 tests** after the final review fixes; Task 5's focused catalogue/dashboard tests passed **26 tests**. The final origin-only V2 release checks completed on 2026-10-21; broader IDSSE synchronization, attack direction, endpoints, xT/progression, and held-out evaluation remain deferred.

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

The focused source-audit suite passed **26 tests after follow-up audit fixes** (the initial baseline passed 21). The release command validated all four match/team reports and confirmed only the other match was used for training. Two unchanged-input exports produced identical SHA-256 hashes for `release-validation.json` and `featured-analysis.md`. The featured game-1 away report matches the saved dashboard findings: 22/76 left-channel buildup losses and 9/137 centre-start final-third entries; its overall cards show 61/238 buildup losses, 362/473 completed passes, 31 entries and 6 modeling shots. Fresh xT, pitch-control, zone-threat and player-impact checks all exited 0. Their measured results and runtime/source-data metadata are in [`docs/demo/validation-summary.md`](demo/validation-summary.md).

## Windows setup and portfolio media (2026-10-19)

A separate ignored `.venv-check` was created using Python 3.12.14 and installed with `python -m pip install -e ".[dev,dashboard,demo]" -c requirements-demo.txt`. Installation exited 0; `python -m pip check` returned `No broken requirements found.` Pip reported a nonfatal failure to check its latest version after installation. The Windows `py` launcher was absent, so the existing Python 3.12 executable created the environment. The README uses `python`, which must resolve to Python 3.12. Constraints pin direct dependencies rather than every transitive dependency.

The clean environment launched `python -m streamlit run app.py --server.headless true --server.port 8517`; its local endpoint returned HTTP 200. Real browser inspection and interaction capture provide rendered-dashboard checks in addition to that HTTP check. Source data was reused locally and remains ignored.

The portfolio walkthrough is an 80-second H.264 MP4 at 1440 × 1000 and 3 fps, encoded from 240 sequential real browser captures. A caption band identifies Metrica Sports and distinguishes measured findings from hypotheses. Playback timing is normalized, so this is edited low-frame-rate browser footage, not match video or seamless high-frame-rate screen recording. Full sequential FFmpeg decoding returned all 240 frames without failure. Visual checks at 0, 40 and 79.67 seconds confirmed readable captions and expected opening, Players midpoint and game-2 ending. Additional sequence inspection confirmed the four views, second supporting moment, slider endpoints, control toggle and match switch. The dedicated Evidence PNG shows the full pitch/source arrow; scrolling clips lower content in some walkthrough frames.

Local Markdown links in the main README, roadmap and demo README resolve. Raw data, environments and capture scratch files are ignored. See [demo notes and media](demo/README.md) for attribution and capture details.

Native in-app browser playback served over local HTTP ran from start to completion: video duration 80 seconds, readyState 4; playback advanced while unpaused at 7.07, 27.88 and 35.63 seconds and finished with `ended=true`, `currentTime=80`. Visual playback checks confirmed game-1 away Overview at opening, risk/channel chart at 27.88 seconds, player table at 35.63 seconds and game-2 away 3–2 Overview at the end, each with readable captions. Native seeking was unavailable; full sequential decoding and the additional evidence-frame inspections cover the remaining footage.

## V4 held-out validation (2026-10-21)

V4 remains **in progress**. This section is the current three-match audit; older sections above describe earlier releases and are not current V4 counts. Raw source data was reused locally through ignored junctions, never downloaded, modified, or committed. Kloppy 3.19.1 was installed in the shared Python 3.12 environment.

Reproduce from the repository root after installing `.[dev,dashboard,metrica-epts]` and providing all three local source sets:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_release.py data/raw/metrica --output docs/validation/v4
.\.venv\Scripts\python.exe scripts/validate_cross_match.py --metrica data/raw/metrica --output docs/validation/v4
```

Either script command runs the same complete source gate and exports both [release audits](validation/v4/release-validation.json) and [held-out comparisons](validation/v4/cross-match-validation.json). The latter command preserves the older standalone numerical experiment when `--metrica` is absent. The README documents the optional adapter, Game 3 file layout, attribution, and one-command dashboard launch.

The final full suite, with all three Metrica source sets and optional Kloppy installed, exited 0: **449 passed in 427.96s (0:07:07)**, with no skipped tests or reported warnings. The focused source/audit unit selection passed **38 tests** with the real export deselected; the full suite includes that real export. An earlier full run before the final citation-field regression passed 448 tests. Local Markdown links and `git diff --check` passed.

### Source and descriptive gates

The real-data command exited 0: `Validated 6 real team reports, 6 held-out team folds, 18 suggestions; exported docs\validation\v4`. Scores are Home 3–0 Away, Home 3–2 Away, and Home 0–2 Away. Game 3 score also agrees with EPTS metadata. Complete goal records, penalty flags, and penalty-excluded modeling shot identities match the original feeds. Source totals include only explicitly recognized defensive subtypes; ambiguous exclusions remain listed by reason in each report quality record.

| Game | Passes | Modeling shots | Tackles | Interceptions | Recoveries |
|---|---:|---:|---:|---:|---:|
| metrica_1 | 1021 | 24 | 66 | 162 | 23 |
| metrica_2 | 1141 | 23 | 88 | 145 | 31 |
| metrica_3 | 1369 | 20 | 72 | 127 | 26 |

Both periods have contiguous 25 Hz frame clocks, no duplicate object keys or conflicting clocks, and every retained event lies within its period tracking frame/time bounds. Every canonical event was audited against the original CSV or Game 3 JSON: exact type/outcome, player/team, period/frame/source seconds, and normalized coordinate conversion. Period orientation is inferred independently from the original teams’ shot origins and compared with canonical coordinates; source events agree with tracking-derived orientation. Sequence membership, team/match/period scope, shot counts, clocks, turnover references, and the clipped 10-second limit are checked. The xT provenance of every report contains exactly the other two match IDs.

Event-to-player tracking coverage is 100% in all three matches. Alignment p90 is 0.507 m, 0.509 m, and 0.000 m; Game 3 events coincide with its sampled player positions, so zero is source agreement, not independent sensor-accuracy evidence. Out-of-bounds samples and dropped missing player positions remain source limitations. Rates use observed tracking samples, not assumed 90-minute matches.

### Leave-one-match-out descriptive comparisons

A shot rate means **sequences containing at least one modeling shot / observed sequences**, not shots divided by sequences. The possession baseline uses the same held-out team over its whole observed match. It is descriptive and overlapping with the transition sample, not an independent control. Five of six transition rates are below this baseline; Game 2 away is slightly above. This is not evidence of benefit, harm, or significance.

| Game / team | Transition shot sequences / transitions | Possession shot sequences / possessions |
|---|---:|---:|
| metrica_1 / home | 5/100 (5.00%) | 18/202 (8.91%) |
| metrica_1 / away | 0/103 (0.00%) | 6/197 (3.05%) |
| metrica_2 / home | 4/89 (4.49%) | 13/197 (6.60%) |
| metrica_2 / away | 5/91 (5.49%) | 10/188 (5.32%) |
| metrica_3 / home | 4/116 (3.45%) | 9/211 (4.27%) |
| metrica_3 / away | 4/119 (3.36%) | 11/217 (5.07%) |

Defensive comparisons use a **cross-match dataset baseline**, never a same-team baseline: anonymous home/away identities cannot be linked across games. For each action type, sum both teams’ actions across the other two matches, then divide by their summed observed team-minutes and multiply by 90. Do not average per-team rates. Held-out source rows are excluded from the baseline. The JSON retains baseline match IDs, raw action numerator, exposure denominator, and both rates.

| Game / team | Action | Actions / observed team-minutes | Per 90 | Other-match dataset actions / team-minutes | Dataset per 90 |
|---|---|---:|---:|---:|---:|
| metrica_1 / home | interception | 72/96.671 | 67.032 | 272/379.889 | 64.440 |
| metrica_1 / home | recovery | 12/96.671 | 11.172 | 57/379.889 | 13.504 |
| metrica_1 / home | tackle | 33/96.671 | 30.723 | 160/379.889 | 37.906 |
| metrica_1 / away | interception | 90/96.671 | 83.790 | 272/379.889 | 64.440 |
| metrica_1 / away | recovery | 11/96.671 | 10.241 | 57/379.889 | 13.504 |
| metrica_1 / away | tackle | 33/96.671 | 30.723 | 160/379.889 | 37.906 |
| metrica_2 / home | interception | 74/94.104 | 70.773 | 289/385.023 | 67.554 |
| metrica_2 / home | recovery | 21/94.104 | 20.084 | 49/385.023 | 11.454 |
| metrica_2 / home | tackle | 37/94.104 | 35.386 | 138/385.023 | 32.258 |
| metrica_2 / away | interception | 71/94.104 | 67.904 | 289/385.023 | 67.554 |
| metrica_2 / away | recovery | 10/94.104 | 9.564 | 49/385.023 | 11.454 |
| metrica_2 / away | tackle | 51/94.104 | 48.776 | 138/385.023 | 32.258 |
| metrica_3 / home | interception | 50/95.841 | 46.953 | 307/381.549 | 72.415 |
| metrica_3 / home | recovery | 7/95.841 | 6.573 | 54/381.549 | 12.738 |
| metrica_3 / home | tackle | 35/95.841 | 32.867 | 154/381.549 | 36.326 |
| metrica_3 / away | interception | 77/95.841 | 72.308 | 307/381.549 | 72.415 |
| metrica_3 / away | recovery | 19/95.841 | 17.842 | 54/381.549 | 12.738 |
| metrica_3 / away | tackle | 37/95.841 | 34.745 | 154/381.549 | 36.326 |

### Channel shots and goals

Each cell below is **channel shots / team modeling shots; channel goals / team modeling goals**. Zero-goal denominators remain explicit raw `0/0` counts; no goal-share percentage is inferred. Channels follow each team’s attacking direction. Game 2 away has one modeling goal and two complete score goals because its scored penalty is excluded from modeling.

| Game / team | Right | Centre | Left |
|---|---|---|---|
| metrica_1 / home | 4/18; 1/3 | 13/18; 2/3 | 1/18; 0/3 |
| metrica_1 / away | 1/6; 0/0 | 4/6; 0/0 | 1/6; 0/0 |
| metrica_2 / home | 2/13; 0/3 | 9/13; 3/3 | 2/13; 0/3 |
| metrica_2 / away | 1/10; 0/1 | 9/10; 1/1 | 0/10; 0/1 |
| metrica_3 / home | 3/9; 0/0 | 6/9; 0/0 | 0/9; 0/0 |
| metrica_3 / away | 2/11; 0/2 | 6/11; 2/2 | 3/11; 0/2 |

### Suggestions audited separately

All **18/18 displayed suggestions** passed their descriptive support audit: independently recounted numerators/denominators satisfy the display thresholds, every cited moment belongs to the stated supporting pattern, and a nonempty adjustment and visible tradeoff are present. Full cited event fields were checked against source rows. The JSON records each finding’s category, title, counts, citations, tradeoff, and descriptive accuracy separately from tactical effect. Supporting moments are examples, not an exhaustive list of all counted events.

**Tactical effect is not evaluated.** A stated tradeoff is a plausible consideration, not an observed counterfactual outcome. The source data cannot validate whether closer cover, forward support, or a route change would alter the score. There are only three independent anonymous matches, no random assignment, no significance claim, and no learned shot-quality estimate. Counts omit unobserved carries and defensive work. The report finding cap can hide otherwise available summary patterns.

### Remaining release gates

The Task 4 six-selection Streamlit AppTest smoke passed against its recorded implementation revision; deterministic chat tests verify active-report citations and causal abstention. Live V4 Ollama answers and human browser layout/evidence checks at the final V4 revision remain unrecorded. V4 stays in progress pending those owner acceptance gates. Automated source/descriptive validation does not substitute for them.
