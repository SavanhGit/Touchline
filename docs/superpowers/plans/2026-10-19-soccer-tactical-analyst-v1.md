# Soccer Tactical Analyst V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish and verify Touchline, a local post-match dashboard that connects measured patterns to source moments and cautious tactical hypotheses.

**Architecture:** Keep source ingestion and complete match context separate from modeling events. A UI-independent report engine consumes a match bundle and held-out training matches; Streamlit presents the resulting evidence without depending on global agent state. Reuse the implementation already in the working tree, filling verified gaps rather than rebuilding it.

**Tech Stack:** Python; NumPy/pandas/PyArrow; pytest/SciPy; Streamlit/Plotly; optional ImageIO/FFmpeg/Pillow for demo media. Use Python 3.12 for the demonstrated Windows environment; preserve the package's declared Python >=3.10 compatibility unless a separately reviewed dependency requirement changes it.

**Spec:** [Soccer Tactical Analyst V1 Design](../specs/2026-10-19-soccer-tactical-analyst-v1-design.md). Read the spec and this entire plan before accepting a task.

## Global Constraints

- V1 delivery: a local interactive Streamlit dashboard, a source-checked losing-team example, screenshots, a short recorded walkthrough, and reproducible Windows instructions.
- Start with Metrica sample games 1–2 and support either team.
- Keep existing tuple-returning loaders compatible.
- Canonical tracking is in metres, centered on a nominal 105 × 68 m pitch. Home attacks +x after orientation in every period; away attacks −x.
- Object identity is `(match_id, period, frame, team, player_id)`; temporal joins must include period and match.
- Default player eligibility is at least 15 tracked minutes; missing observations must not be counted as observed time.
- Fit xT on other matches only. Reject the held-out match in the training set; record training IDs.
- Show at most one finding in each category and at most three total. A finding needs a denominator of at least five relevant events and at least three supporting events.
- These are display safeguards, not significance tests.
- Keep complete scored goals, including penalties, separate from penalty-excluded modeling shots.
- Label the view tracking reconstruction, not match video.
- V1 excludes chat, model-provider selection, public hosting, live tracking, arbitrary match downloads, learned forecasting, video inference, and expanded tactical models.
- Keep source datasets external to Git; use deterministic tests and no paid API calls.
- Current dependency declarations: numpy>=1.26, pandas>=2.1, pyarrow>=14; dev pytest>=8/scipy>=1.11; dashboard streamlit>=1.40,<2/plotly>=5.24,<8. `requirements-demo.txt` is a tested direct-dependency constraints file, not a complete lockfile.

## Review Focus

1. Bundles with stale/empty quality metadata or nonfinite coordinates must not bypass correctness checks or create plausible false findings — Task 1.
2. Source-rate rounding, explicit invalid fps and missing samples must not silently change player eligibility or tracked minutes — Task 2.
3. Exactly 5/3 events, zero denominators, tied channels and shot-free training must yield deterministic findings or an explicit absence/error — Task 3.
4. Missing tracking, repeated frame numbers, half boundaries and discontinuous trails must not fabricate a reconstruction — Task 4.
5. Partial data, unknown scores and match/team changes must not crash the app, imply 0–0, or reuse another report's evidence — Task 5.

---

## Starting state and execution safety

Inspected on **2026-10-19** in `C:\Users\Minec\Documents\soccer-tactics`, branch `codex/tactical-analyst-test`. HEAD is `c70357af3c5c7d97b064cedd198133f34cb8c7e9`. This is a normal checkout, not a linked worktree.

**Important:** Most V1 code is still uncommitted, including entire new files. A new worktree created from branch HEAD alone will omit it. Do not reset, clean, stash-and-drop, or rebuild from that older HEAD. The orchestrator must first preserve the authorized working state as a reviewed checkpoint, or continue in this checkout. Do not automatically create another branch/worktree just because a task mentions isolation.

| Component | Existing state | Required next action |
|---|---|---|
| `MatchBundle`, Metrica complete goals, quality checks | Implemented | Verify contracts and harden invalid inputs |
| Orientation, pressure keys, kinematics, source-rate minutes | Corrections present | Add boundary regressions; inspect compatibility |
| Deterministic report engine | Implemented | Verify exact eligibility, ties, evidence and provenance |
| Four-view Streamlit dashboard and Plotly pitches | Implemented | Test empty/error states and inspect visual behavior |
| Real-data release script and featured report | Present | Strengthen source checks and regenerate after final changes |
| README/setup | Still describes the older analytics engine | Update for the actual dashboard release |
| Screenshots and walkthrough | Absent in `docs/demo/` at inspection | Capture after UI verification |
| Final full-suite result | Not established in this planning pass | Collect fresh evidence; do not copy old counts |

This planning task does not execute agents, alter product code, or certify previous test runs. All task checkboxes remain open until their acceptance is verified.

## File responsibilities and ownership

| Unit | Files | Responsibility |
|---|---|---|
| Data boundary | `src/soccer_tactics/match.py`, `loaders/bundle.py`, `loaders/canonical.py`, `schema.py` | Match context, source quality, goals, orientation |
| Existing metric corrections | `analytics/features.py`, `analytics/impact.py`, `agent/tools.py` | Match/period-safe calculations; retain existing APIs |
| Explanation engine | `analytics/report.py` | Counts, findings, evidence and provenance; no Streamlit |
| Evidence presentation | `dashboard/charts.py` | Plotly rendering and bounded tracking-window helpers |
| App | root `app.py`, `.streamlit/config.toml` | Selection, caching, readable tables and four views |
| Release audit | `scripts/validate_release.py`, `docs/demo/release-validation.json` | Original-source checks and reproducible output |
| Portfolio | `README.md`, `docs/VALIDATION.md`, `docs/demo/*`, `docs/ROADMAP.md` | Setup, methods, screenshots, walkthrough and roadmap |
| Test support | `tests/factories.py` (new), existing/new tests named below | Small synthetic inputs; no imports from the app |

Paths under `src/soccer_tactics/` in this table are repository-relative. Task ownership is exclusive while that task is executing. Avoid splitting the 200-line report module or the app into extra layers unless a concrete task cannot be expressed cleanly without doing so.

## Agent coordination and task gates

Dependency order: **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**. This is intentionally sequential for the first completion pass because the contracts and evidence IDs are shared. Do not dispatch multiple writers to `tests/test_report.py`, `match.py`, `report.py`, or `app.py` at once.

Each implementer receives the spec, global constraints, its task, interfaces from dependencies, and the latest checkpoint SHA. Its completion note must list changed files, exact commands/results, any skips, and remaining limitations. A reviewer checks the task against the spec and diff before the next dependent task starts. Return rejected changes to the same implementer; do not silently add scope.

For already implemented behavior, run the existing regression first. A passing check means preserve that behavior; it does not justify deleting/reimplementing it to manufacture a red step. For an uncovered gap, add a failing test, observe the failure, make the smallest correction, rerun the targeted checks, then commit only that task's reviewed files. Commit commands below are for future execution, not an instruction to commit during planning.

### Task 0: Preserve the current implementation and establish the baseline

**Owner:** Orchestrator. **Depends on:** approved execution handoff.

**Files:** Inspect all current tracked/untracked project changes; update `docs/VALIDATION.md` with the observed baseline. Do not modify raw data or unrelated user files.

**Interfaces:** Consumes the current working checkout; produces a checkpoint SHA containing all existing V1 source needed by later agents and a recorded baseline command/result.

- [ ] **Step 1: Inventory the checkout.** Run `git branch --show-current`, `git status --short`, `git diff --stat`, and `git diff --check`. Confirm the testing branch and review untracked files individually; note that no worktree isolation currently exists.
- [ ] **Step 2: Verify prerequisites.** Use the existing `.venv\Scripts\python.exe` if healthy. If absent, use Python 3.12 to create `.venv`, then `python -m pip install -e ".[dev,dashboard,demo]" -c requirements-demo.txt`. Reuse an existing Metrica download; clone `https://github.com/metrica-sports/sample-data` to `data/raw/metrica` only when absent. Verify both games have event, home and away CSVs.
- [ ] **Step 3: Run the baseline.** From repository root: `.\.venv\Scripts\python.exe -m pytest -q`. Record the actual total and any skips/failures in `docs/VALIDATION.md`; full release verification requires installed optional dashboard dependencies and both real matches. Diagnose a failure before changing its expected answer.
- [ ] **Step 4: Preserve the implementation.** Review and stage the existing V1 changes by explicit paths from the ownership table, plus current dependency/config/test changes. Exclude `data/raw`, `.venv`, caches, unrelated files and secrets. Commit a checkpoint such as `chore: checkpoint tactical analyst v1 work in progress` and record its SHA for agents. Do not push or merge as part of this task.

**Acceptance:** Future workers can access the complete current implementation, and its baseline is reported honestly rather than described as clean merely because a branch exists.

### Task 1: Verify match context and data-boundary validation

**Files:** Modify `src/soccer_tactics/match.py`, `loaders/bundle.py`, `schema.py`, `loaders/canonical.py`, and `analytics/report.py` only to delegate bundle checks. Create `tests/factories.py` and `tests/test_match_bundle.py`; update `tests/test_report.py` imports if extracting its existing helper.

**Interfaces:**
- Preserve `load_metrica_bundle(root: str | Path, game: int = 1) -> MatchBundle`, `infer_fps(tracking: pd.DataFrame) -> float`, and `inspect_quality(tracking: pd.DataFrame, events: pd.DataFrame) -> dict`.
- Add `validate_bundle(bundle: MatchBundle) -> None` in `match.py`; `analyze_match` calls it for selected and training bundles. It checks actual data, not just cached quality claims.
- Test helper `make_bundle(match_id: str = "test", *, fps: float = 1.0) -> MatchBundle` in `tests/factories.py` preserves the existing synthetic report example: 15 minutes of tracking, ten home passes with six losses/two entries, five away shots, unknown score. Adjust frames from source seconds when fps changes.

- [ ] **Step 1: Add boundary tests.** Keep existing orientation and real-penalty tests. Add these assertions, using copies of `make_bundle()` and `dataclasses.replace`:

```python
assert make_bundle().score is None
assert replace(make_bundle(), score_known=True).score == {"home": 0, "away": 0}
# test_invalid_bundle_without_quality_metadata:
with pytest.raises(ValueError, match="duplicate"):
    validate_bundle(replace(b, tracking=pd.concat([b.tracking, b.tracking.iloc[:1]]), quality={}))
# test_nonfinite_positions_and_conflicting_clock_are_rejected:
# parameterize NaN/+inf x or y, and one frame with two time_s values; each raises ValueError.
# test_goal_records_belong_to_the_bundle:
# foreign match_id when present, invalid team or nonfinite goal timestamp raises ValueError.
```

- [ ] **Step 2: Observe uncovered failures.** Run `.\.venv\Scripts\python.exe -m pytest tests/test_match_bundle.py tests/test_schema.py tests/test_report.py -q`. Expect new validation tests to fail before corrections; preserve passing orientation/penalty behavior.
- [ ] **Step 3: Implement the boundary.** Validate required columns, finite retained tracking coordinates/times, finite event origins and pass endpoints, positive finite fps, declared match IDs, unique object keys, a single clock per period/frame, valid goal teams/times, and nondecreasing source time as frames advance inside each period. A shot endpoint may be missing: retain its origin and omit the unavailable arrow. Raise actionable `ValueError`s; do not repair or drop data invisibly. Empty modeling events are allowed; empty tracking cannot form a match report. Preserve reported out-of-bounds positions and the unknown/known-score distinction.
- [ ] **Step 4: Verify.** Rerun the Step 2 command. Expected: all pass, with the real penalty test run when both games exist. Independently assert game scores exactly `{"home": 3, "away": 0}` and `{"home": 3, "away": 2}`; game 2 has one penalty goal, five complete goals, four modeling goals.
- [ ] **Step 5: Commit reviewed files.** Stage only this task's files and commit `fix: validate complete match bundles and source context`.

**Acceptance:** Bad or ambiguous source data cannot enter a plausible report, regardless of whether quality metadata was supplied.

### Task 2: Lock down match/period isolation and source-rate minutes

**Files:** Modify `src/soccer_tactics/analytics/features.py`, `analytics/impact.py`, `agent/tools.py` if their regressions expose gaps. Create `tests/test_temporal_integrity.py`; preserve existing feature, impact and report tests.

**Interfaces:** Preserve `add_kinematics(df, fps=25, window=5, max_speed=12.0, ball_max_speed=40.0) -> pd.DataFrame`, `add_pressure(passes, tracking) -> pd.DataFrame`, and `player_impact(passes, tracking, min_minutes=15.0, *, fps=None) -> pd.DataFrame`. The legacy fps argument to kinematics remains accepted; source `time_s` is authoritative for derivatives. Explicit player-impact fps overrides inference and must be positive and finite.

- [ ] **Step 1: Add parameterized regressions.** Check rates 10, 25 and 30 Hz; two matches sharing player IDs; halves sharing frames; dropped samples; zero/negative time deltas in direct kinematics calls; and fps values 0, −1, NaN and infinity. Pin:

```python
# test_sample_rate_minutes_and_cutoff, using make_bundle(fps=fps)
assert np.allclose(table.minutes, 15.0)
# test_dropped_samples_reduce_observed_minutes: remove ten seconds for player h;
# obtain this table with min_minutes=0 so the below-cutoff player remains inspectable.
assert np.isclose(table.set_index("player_id").loc["h", "minutes"], 15 - 10 / 60)
# A separate call with the default 15-minute cutoff excludes h.
# test_invalid_explicit_fps:
with pytest.raises(ValueError, match="fps"):
    player_impact(passes, tracking, fps=invalid_fps)
```

- [ ] **Step 2: Run the new tests.** `.\.venv\Scripts\python.exe -m pytest tests/test_temporal_integrity.py tests/test_features_and_agent.py tests/test_impact.py -q`. Record which safeguards already pass.
- [ ] **Step 3: Correct only demonstrated gaps.** Keep match/period/player grouping and full pressure join keys; use rounded inferred rates to avoid floating-point exclusion at exactly 15 minutes; validate explicit fps rather than using truthiness. For invalid/nonpositive derivative intervals return unavailable motion, never infinity. For multiple-match impact tables infer rates per match unless a caller explicitly overrides them. Missing samples reduce tracked duration instead of extending it across gaps.
- [ ] **Step 4: Rerun targeted tests and compatibility checks.** Include `tests/test_cross_match.py` and `tests/test_roadmap.py`; expected all pass, including the original synthetic 15-minute fixtures. A known constant-speed runner remains approximately 2 m/s across all source rates.
- [ ] **Step 5: Commit** `fix: preserve temporal isolation and sampling-rate metrics` with only task-owned changes.

**Acceptance:** Time normalization and player eligibility agree across sources without cross-match or cross-half contamination.

### Task 3: Verify deterministic findings, provenance and evidence contracts

**Files:** Modify `src/soccer_tactics/analytics/report.py`; extend `tests/test_report.py`. Use `tests/factories.py` from Task 1.

**Interfaces:** Retain `Evidence`, `Finding`, `MatchReport` and `analyze_match(match_bundle: MatchBundle, team: str, training_matches: Iterable[MatchBundle]) -> MatchReport`. Add the missing type annotation without changing call sites. Report table columns and category strings `buildup`, `progression`, `danger` remain compatible with the dashboard. `channel(y) -> np.ndarray` follows the exact spec boundaries.

- [ ] **Step 1: Extend the hand-computed tests.** Preserve the ten-pass fixture's six losses, four completions, two entries, 0.6 buildup loss rate, and three-of-five opponent concentration. Add boundary/tie/leakage assertions:

```python
# test_exact_eligibility_thresholds: analyze fixtures containing the named counts.
assert any(f.category == "buildup" for f in five_three_report.findings)
assert not any(f.category == "buildup" for f in four_three_report.findings)
assert not any(f.category == "buildup" for f in five_two_report.findings)
# test_report_limits_and_references:
assert len(report.findings) <= 3
assert len({f.category for f in report.findings}) == len(report.findings)
assert all(1 <= len(f.evidence) <= 3 for f in report.findings)
assert report.provenance["training_match_ids"] == ["training"]
```

The three threshold fixtures contain respectively five/three, four/three and five/two buildup passes/losses in one channel; use the helper's other match for training. Also test empty events, no passes, zero denominators, all-success passes, unknown result, repeated same-frame events, tied channels, and training with no shots. Spy on `fit_xt` to assert selected-match events are absent, in addition to the existing same-ID rejection test.

- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_report.py -q`. Existing behavior may already pass; any new failure becomes the next correction, not a reason to change the spec's 5/3 values.
- [ ] **Step 3: Make findings match the spec exactly.** Preserve category rules, counts, rates and costs; use right/centre/left for final ties. Distinguish progression nonentry supporters from successful-entry numerator. Return `None`/NaN for unavailable aggregate rates, fewer findings for insufficient data, and a clear shot-calibration error for shot-free training. Deterministic evidence IDs include the ordered event ordinal; validate each evidence object against every source event field, not just the frame. Preserve input DataFrames and byte-equivalent evidence IDs for repeated calls on unchanged ordered input.
- [ ] **Step 4: Rerun and inspect wording.** All tests pass; every suggestion has supporting events and a tradeoff. Explicitly assert away rotation at channel boundaries and that failed pass endpoints are never described as intended targets. Verify no copy claims causation, a guaranteed remedy, or a shot probability from pitch control.
- [ ] **Step 5: Commit** `feat: verify deterministic tactical findings and evidence`.

**Acceptance:** Every displayed number and moment is reproducible, and the report describes the limits of its hypotheses accurately.

### Task 4: Verify pitch figures and honest tracking reconstruction

**Files:** Modify `src/soccer_tactics/dashboard/charts.py`; create `tests/test_dashboard_charts.py`.

**Interfaces:** Preserve `pitch(title: str = "", attacking_team: str | None = None) -> go.Figure`, `pass_map(rows: pd.DataFrame, team: str, title: str, losses: bool = False) -> go.Figure`, `tracking_window(bundle: MatchBundle, evidence: Evidence, radius_s: float = 3.4) -> pd.DataFrame`, `snapshot(window: pd.DataFrame, target_s: float, fps: float) -> pd.DataFrame`, and `evidence_pitch(frame, evidence, *, control=False, trail=None) -> go.Figure`.

- [ ] **Step 1: Add geometry/availability tests.** Inspect Plotly traces and annotations rather than screenshot pixels. Pin:

```python
# test_snapshot_does_not_borrow_from_another_half
assert set(tracking_window(bundle, moment).period) <= {moment.period}
# test_snapshot_far_from_available_clock
assert snapshot(window, target_s=5000, fps=25).empty
# test_empty_and_one_team_frames_have_no_control_heatmap
assert not any(trace.type == "heatmap" for trace in fig.data)
```

Also assert the snapshot tolerance at `max(1.5/fps, .06)`, the ±3.4-second window, away-map coordinate negation, a fixed source arrow across slider changes, and a trail gap larger than the tolerance producing separate segments or `None` separators rather than a connecting line.

- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_charts.py -q`. Expected new gap tests fail if the current renderer connects missing segments.
- [ ] **Step 3: Correct the helpers/rendering.** Preserve canonical home→/away← reconstruction coordinates and separate attacking-frame maps. Keep positions finite, do not extrapolate missing evidence, and break trails at missing intervals; keep only the two preceding seconds. Calculate control only from the displayed frame with both teams. Empty frames must still allow a source-event arrow without pretending players were observed.
- [ ] **Step 4: Rerun the chart tests and the existing missing-tracking report test.** Expected all pass; no global agent state or Streamlit import enters these helpers.
- [ ] **Step 5: Commit** `fix: make evidence reconstruction period-safe and gap-aware`.

**Acceptance:** The figures never display an invented path, player observation or control surface.

### Task 5: Complete dashboard selection, empty states and readable presentation

**Files:** Modify `app.py` and `.streamlit/config.toml`; extend `tests/test_dashboard.py`; record manual observations in `docs/VALIDATION.md`. Do not redesign the four-view structure.

**Interfaces:** Preserve `main() -> None`, session keys `selected_match`, `selected_team`, `player_metric`, `show_control`, and context-specific finding/moment/offset keys. Preserve `load_bundle(game, revision) -> MatchBundle` and `load_report(game, team, revision) -> MatchReport`. For synthetic UI tests, use an AppTest script that imports the app and substitutes its `DATA` path and loader functions before calling `main()`. Clear decorated caches between tests. No product data-root interface is added merely for testing.

- [ ] **Step 1: Add AppTest cases.** Keep the real navigation smoke test. Add synthetic tests for nonexistent/partial source files, unknown score, no findings, no eligible players, unavailable evidence, and switching from a report with three findings to one with none. Assertions include:

```python
assert not app.exception
assert any("unknown" in item.value.lower() for item in app.markdown)  # score context
assert any("No pattern" in item.value for item in app.info)  # no findings
# test_missing_sources_stop_before_modeling: loader spy call count is zero
# test_match_team_switch_does_not_reuse_evidence: selected event IDs belong to the current report
```

- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -q`. Expected the existing real navigation passes; newly uncovered empty-state failures are isolated by synthetic tests, not by downloading another dataset.
- [ ] **Step 3: Implement the minimum UI corrections.** Check all six required CSVs before modeling, not only directory existence. Present an actionable source-data message and stop safely. Format unavailable rates as “Unavailable”; never convert NaN to an integer or imply 0%. Show unknown result separately from known 0–0. Keep player eligibility/GK separation, source-clock/period labels, hypotheses/tradeoffs, and local cache ownership. Ensure report rendering does not mutate cached bundles. Keep API/implementation details in the methods expander rather than the main flow.
- [ ] **Step 4: Run synthetic and real UI tests, then inspect in a browser.** Test both matches and both teams, all four views, evidence selection, slider endpoints and control toggle. Inspect a roughly 1100-pixel-wide and a 1440-pixel-wide desktop layout: metrics/copy readable, no overlapping controls, pitch aspect ratio retained. Record observations in `docs/VALIDATION.md`; screenshots come after final verification in Task 7.
- [ ] **Step 5: Commit** `feat: complete resilient post-match dashboard views`.

**Acceptance:** Match/team navigation is reliable and the app can explain unavailable results without crashing or silently changing their meaning.

### Task 6: Strengthen and regenerate the original-source release audit

**Files:** Modify `scripts/validate_release.py`; create `tests/test_release_validation.py` and `docs/demo/validation-summary.md`; regenerate `docs/demo/featured-analysis.md` and `docs/demo/release-validation.json`; update `docs/VALIDATION.md` with results.

**Interfaces:** Keep `run(root: Path, output: Path) -> None` and its CLI arguments. Add a testable `verify_evidence(bundle: MatchBundle, evidence: Evidence, raw_events: pd.DataFrame) -> None` in the same script; compare canonical event fields plus the raw source row after unit conversion/orientation. Use explicit exceptions for audit failures so `python -O` cannot remove the release gate.

- [ ] **Step 1: Add source-audit tests.** A correct source reference succeeds; a right frame with wrong player, team, period, timestamp, event type or coordinates must fail. Use small raw CSV fixtures matching the current loader format, and `pytest.raises(ValueError)` for mismatches. Test a changed synthetic quality metric appears in exported prose, preventing hardcoded “100%” claims. Verify JSON contains no NaN/Infinity and all four real reports identify only the other match as training.
- [ ] **Step 2: Run** `.\.venv\Scripts\python.exe -m pytest tests/test_release_validation.py -q`. Expected the strengthened mismatch tests fail before the current presence-only source checks are replaced.
- [ ] **Step 3: Strengthen the existing audit.** Check complete scores, goal exclusions, quality and every displayed evidence field against original source records. Preserve raw row semantics for failed passes; do not classify them as ordinary completed PASS rows. Build featured prose from measured results, not fixed text claiming alignment for arbitrary inputs. Leave the output structure compatible with current exported findings; normalize unavailable numeric values to JSON null. Include source-data Git SHA and runtime versions in a small `docs/demo/validation-summary.md`, keeping nondeterministic run metadata separate from deterministic report JSON.
- [ ] **Step 4: Run the complete real-data commands.** All must exit 0:

```powershell
.\.venv\Scripts\python.exe scripts/validate_release.py
.\.venv\Scripts\python.exe scripts/validate_xt_split_half.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_pitch_control.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_zone_threat.py data/raw/metrica
.\.venv\Scripts\python.exe scripts/validate_player_impact.py data/raw/metrica
```

Retain command summaries in `validation-summary.md`; existing results are comparison context, not expected constants to force. Run the release export twice with unchanged input and verify `release-validation.json` and `featured-analysis.md` have identical hashes. Inspect the featured away-team game-1 report and compare its displayed counts with the dashboard.
- [ ] **Step 5: Commit** `test: audit real match reports against original source events` with the reviewed regenerated artifacts.

**Acceptance:** Every featured observation can be checked against source data, and exported prose never overstates checks that were not actually run.

### Task 7: Finish Windows setup and the portfolio demo

**Files:** Modify `README.md`, `docs/VALIDATION.md`, `docs/ROADMAP.md`, `requirements-demo.txt` and `.gitignore` if needed; create `docs/demo/README.md`, `docs/demo/overview.png`, `docs/demo/risk-progression.png`, `docs/demo/players.png`, `docs/demo/evidence.png`, and `docs/demo/walkthrough.mp4`.

**Interfaces:** Normal launch remains `.\.venv\Scripts\python.exe -m streamlit run app.py`. Demo capture uses this local app and existing report artifacts; it does not introduce hosted services, fabricated visual data, account setup or an API key.

- [ ] **Step 1: Verify setup from a clean dependency context.** Create an ignored `.venv-check` with Python 3.12, install `-e ".[dev,dashboard,demo]" -c requirements-demo.txt`, and run `python -m pip check`. Add `.venv-check/` to `.gitignore` if this context is kept. Expected no unsatisfied dependencies. Existing data is reused locally; document how a new user obtains it.
- [ ] **Step 2: Update the README around the actual product.** Include the project question, four views, screenshot links, featured report, Windows venv/install/data commands, the single launch command, validation commands, attribution, and clear limitations. Keep analytical APIs documented, but describe agent/video as future work rather than an operational chat/video feature. Note that `requirements-demo.txt` pins direct dependencies and is not a transitive lockfile.
- [ ] **Step 3: Capture final screenshots.** Use game 1, away perspective, with readable text and no loading overlays. Capture each of the four views after verifying the selected report, and inspect each saved PNG. Evidence screenshot includes player positions, source arrow, timestamp and the reconstruction label.
- [ ] **Step 4: Record a 45–90-second walkthrough.** Navigate overview → risk/progression → players → evidence; select another moment, move the slider, toggle control, and briefly switch matches. Captions explain measured evidence versus hypotheses and identify Metrica Sports. Capture real browser interaction; if only still captures are available, name and describe the output as a captioned slideshow and report that continuous recording remains unfinished.
- [ ] **Step 5: Verify media and documentation.** Play the MP4 and inspect beginning/middle/end for readability and correct content. Check that all README/demo links resolve, the launch command reaches the local dashboard, and `git status --short --untracked-files=all` excludes raw data, environments and capture scratch files. Record the exact manual checks and source attribution in `docs/demo/README.md`.
- [ ] **Step 6: Commit** `docs: add reproducible Touchline portfolio demonstration` with only public-safe documentation/media and configuration changes. No push or publication is part of this task.

**Acceptance:** Someone unfamiliar with the project can run it on Windows and understand a real example without mistaking it for a validated coaching recommendation system.

### Task 8: Final integration and release handoff

**Owner:** Orchestrator plus a fresh reviewer. **Files:** Read the full branch; update only factual acceptance results in `docs/VALIDATION.md` and task checkboxes after checks succeed.

**Interfaces:** Consumes the completed V1 artifacts; produces a verified local-release handoff and an explicit list of remaining limitations.

- [ ] **Step 1: Verify the integrated checkout.** Run `.\.venv\Scripts\python.exe -m pytest -q`, `.\.venv\Scripts\python.exe scripts/validate_release.py`, `.\.venv\Scripts\python.exe -m pip check`, and `git diff --check` against the final code. Confirm both real matches and dashboard dependencies are present, so missing-data/optional-dependency skips cannot masquerade as release success. Record actual counts; expected zero failures and no missing-data/dashboard skips.
- [ ] **Step 2: Review the branch against the spec.** A fresh reviewer checks source correctness, held-out fitting, exact denominators, evidence fidelity, compatibility, UI empty states, artifact reproducibility, and misleading product claims. Do not start V2 as a way to avoid a V1 defect.
- [ ] **Step 3: Resolve actionable findings.** Return each issue to its owning task, add a failing regression when applicable, fix and rerun affected checks. Repeat the final gate only if the reviewed code changed.
- [ ] **Step 4: Hand off locally.** Report the branch/checkpoint, verified test results, launch command, featured report, screenshots and recording links, plus real limitations. Do not merge, push, create a PR or publish a site unless the user requests that integration step.

**Acceptance:** V1 is complete only when data, reports, UI, tests, media and setup all satisfy their gates. “Code written” alone is not release completion.

## Later work packages — separate plans, not executable V1 tasks

| Package | Prerequisite | First work | Acceptance before integration | Decisions intentionally deferred |
|---|---|---|---|---|
| V2: IDSSE catalogue | V1 accepted; real XML obtainable | Verify parser API and complete match triplets, synchronize clocks/orientation, add bundle context | Real-data quality checks; per-match held-out evaluation; source semantics/attribution documented | Download strategy and any replacement dataset |
| V3: Question-answering | Stable V1 evidence/report interfaces | Design per-session context and tools replacing global state; add a small question/evidence evaluation set | Correct numbers, current match/team scope, evidence citations, clear unsupported answers; dashboard works without credentials | Model provider, budget, local vs API access |
| V4: Richer analysis | Additional event coverage | Select one possession/transition/defensive question and baseline | Added value on held-out matches; no causal claims from descriptive fit | Which model/question has sufficient data |
| V5: Recorded video | Reference tracking and usable footage | Evaluate player/ball detection, IDs and pitch mapping through the existing adapter | Location/identity/visibility error measured; missing off-camera players explicit | Video model, hardware and source rights |
| V6: Live analysis | Validated recorded-video pipeline | Design incremental inputs, past-only motion features and latency budget | No future-frame leakage; measured latency and incomplete-data behavior | Live deployment and performance budget |

Do not expand these rows into invented provider APIs, model architectures or performance promises. Each package starts with its own data/feasibility review and user-approved design.

## Plan self-review and handoff

Coverage: complete goals and data conventions → Tasks 1–2; deterministic explanations and source references → Task 3; reconstruction → Task 4; interactive views and failure states → Task 5; both real matches and source audit → Task 6; reproducibility/media → Task 7; final acceptance → Task 8. The five Review Focus conditions each have an owning task and explicit regression.

Existing source APIs are retained. New interfaces are limited to bundle validation, a synthetic test factory, and a source-evidence audit helper. No agent execution method has been chosen yet. Review this document and choose subagent-driven or native execution before starting Task 0; the earlier implementation request is background context, not a reason to start executing this newly expanded plan while it is being reviewed.
