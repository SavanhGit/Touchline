# Touchline V2 IDSSE Catalogue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task by task. Each task is implemented and reviewed before the next begins. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the seven IDSSE match sources and add only quality-approved, held-out-evaluated matches to Touchline's local catalogue.

**Architecture:** Keep IDSSE XML parsing behind the existing loader boundary, preserve `load_idsse_game`, and add an IDSSE `MatchBundle` loader with complete goals, metadata, canonical orientation, and quality results. Run a reproducible leave-one-match-out audit using event-only training inputs, then let the dashboard consume only the accepted validation manifest while retaining the Metrica path.

**Tech Stack:** Python 3.10+, pandas, NumPy, optional Floodlight (`.[idsse]`), pytest, Streamlit, Plotly.

**Spec:** `docs/superpowers/specs/2026-10-20-touchline-v2-idsse-design.md`

## Global Constraints

- Keep Floodlight optional; Metrica-only setup and dashboard use must continue to work without it.
- Keep `load_idsse_game(root, match_id) -> (tracking, events)` compatible.
- Keep raw XML and other downloaded match data outside Git.
- Do not treat unvalidated matches as selectable or use them as xT training data.
- Do not add automatic user uploads, downloads, live/video tracking, public hosting, chat, or new tactical models.
- Match IDs remain unique across providers; reject self-training and mixed-source training in V2.
- Event timestamps may match observed tracking only within the same period and only within half a source frame.
- Only documented IDSSE event types with valid spatial and time keys enter reports; unknown event codes remain in quality totals.
- Extract scored goals independently of modeling shots so penalty goals remain in scores while penalty shots stay excluded from modeling.
- Attribute the Deutsche Fußball Liga (DFL) and cite Bassek et al., “An integrated dataset of spatiotemporal and event data in elite soccer,” *Scientific Data* 12, 195 (2025).
- Require all seven candidates to appear in the real-data audit; require at least two accepted matches for leave-one-match-out evaluation; a missing or skipped real-data gate is not success.
- Commit each reviewed, passing task separately to the current test branch; do not merge to `main`.

## Review Focus

1. **Period clock reset and repeated frame values:** events must resolve only against observed tracking in the same period. Pin with a two-period fixture whose clocks and frame IDs repeat, asserting the selected event frame and alignment error.
2. **Coordinate units and changing attack direction:** canonical tracking and events must both be centered in metres with home attacking +x in each period. Pin with asymmetric pitch coordinates and opposite home attack directions.
3. **Unknown and ambiguous source events:** unsupported codes or missing required coordinates must be counted with explicit reasons and excluded from canonical evidence. Pin with a fixture containing an unknown code and a partially missing pass.
4. **Score semantics for penalties and own goals:** complete scored-goal records must match source score while penalty shots remain excluded from modeling and own-goal attribution follows the documented source semantics. Pin with normal, penalty, own-goal, and non-goal set-piece rows.
5. **Catalogue cache and cross-provider isolation:** only manifest-accepted matches may appear, training must be same-provider and exclude the selected match, and stale/missing manifests must fail closed. Pin with a mixed Metrica/IDSSE catalogue fixture and report-call assertions.

---

### Task 1: Correct and test the IDSSE source adapter

**Files:**
- Modify: `src/soccer_tactics/loaders/idsse.py`
- Test: `tests/test_idsse_loader.py`
- Test: `tests/test_idsse_source_contract.py`

**Interfaces:**
- Preserve `list_idsse_matches(root: str | Path) -> list[str]` and `load_idsse_game(root: str | Path, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame]`.
- Add `IdsseSourceMatch` with fields `match_id: str`, `files: dict[str, Path]`, `positions: object`, `events: object`, `teamsheets: object`, `pitch: object`, and `match_information: object`; `_read_idsse_source(root: str | Path, match_id: str) -> IdsseSourceMatch` discovers and parses exactly one complete triplet.
- Source helpers must retain unsupported and unusable source-event counts with stable reason codes for the quality record; canonical events continue to use `schema.EVENT_COLUMNS`.

- [ ] **Step 1: Write failing source-contract tests** for complete/incomplete triplets, duplicate file-family IDs, lazy missing-Floodlight error text, event-code mapping, penalty/own-goal semantics, missing coordinates, and separate first/second-half objects.
- [ ] **Step 2: Run the focused tests** with `pytest tests/test_idsse_loader.py tests/test_idsse_source_contract.py -q`; confirm failures expose the current broad substring mapping, frame derivation, and missing source diagnostics.
- [ ] **Step 3: Implement the source adapter** so it validates exactly one file for each XML family, passes source objects through a testable Floodlight boundary, maps only source-documented codes, retains source timestamps, and reports exclusions instead of silently dropping rows. Keep the public tuple loader compatible.
- [ ] **Step 4: Verify parser contract tests** with the focused pytest command; assert `load_idsse_game` still returns the same two canonical tables for the supported source cases.
- [ ] **Step 5: Run a sampled real-data source audit** against the seven local IDSSE triplets after the dataset is available at `data/raw/idsse`; compare selected source frames and each mapped event against Floodlight objects. Record unresolved source semantics as disqualifying findings rather than guessing.
- [ ] **Step 6: Commit and push** the passing adapter and tests: `git add src/soccer_tactics/loaders/idsse.py tests/test_idsse_loader.py tests/test_idsse_source_contract.py && git commit -m "fix: validate IDSSE source event semantics"`.

### Task 2: Build canonical IDSSE match bundles

**Files:**
- Modify: `src/soccer_tactics/match.py`
- Create: `src/soccer_tactics/loaders/idsse_bundle.py`
- Modify: `src/soccer_tactics/loaders/__init__.py`
- Test: `tests/test_idsse_bundle.py`
- Test: `tests/test_match_bundle.py`

**Interfaces:**
- Add `metadata: dict = field(default_factory=dict)` to `MatchBundle`, preserving existing constructors.
- Add `load_idsse_bundle(root: str | Path, match_id: str) -> MatchBundle` and export it from `soccer_tactics.loaders`.
- Bundle metadata contains source match date, competition, source team IDs, source pitch dimensions, source team labels, and the transforms applied. `team_names` holds display labels while canonical team values stay `home` and `away`.
- The bundle loader uses `validate_bundle`, `infer_fps`, and `inspect_quality`; complete scored goals use canonical goal columns and include a `penalty` flag.

- [ ] **Step 1: Write failing bundle tests** for default-empty backward-compatible metadata, source labels/IDs, goals and penalty records, match score, actual source FPS, and `validate_bundle` rejection of invalid metadata-derived canonical rows.
- [ ] **Step 2: Run `pytest tests/test_idsse_bundle.py tests/test_match_bundle.py -q`** and confirm the new API/metadata expectations fail.
- [ ] **Step 3: Implement `load_idsse_bundle`** using Task 1 source objects. Convert source coordinates to centered metres, derive and apply per-period orientation to tracking and events, and derive frame/time alignment from observed clocks in the same period only. Assign an event frame only within half a source frame; retain the original event time and alignment error in bundle quality. Extract complete goals separately from modeling events and populate score-known, team labels, source metadata, FPS, and quality.
- [ ] **Step 4: Verify bundle tests** with `pytest tests/test_idsse_bundle.py tests/test_match_bundle.py -q`; include repeated frame IDs and clock resets across periods, missing positions, and source FPS not equal to 25.
- [ ] **Step 5: Commit and push**: `git add src/soccer_tactics/match.py src/soccer_tactics/loaders/idsse_bundle.py src/soccer_tactics/loaders/__init__.py tests/test_idsse_bundle.py tests/test_match_bundle.py && git commit -m "feat: load IDSSE match bundles"`.

### Task 3: Support memory-bounded, source-isolated xT training

**Files:**
- Modify: `src/soccer_tactics/analytics/report.py`
- Test: `tests/test_report.py`
- Test: `tests/test_idsse_training.py`

**Interfaces:**
- Add `TrainingMatch` in `analytics/report.py` with fields `match_id: str`, `source: str`, and `events: pd.DataFrame`.
- Update `analyze_match(match_bundle: MatchBundle, team: str, training_matches: Iterable[MatchBundle | TrainingMatch]) -> MatchReport`; existing `MatchBundle` training remains supported.
- Every training input is checked for canonical events, distinct match ID, and same source as the selected bundle. Provenance continues to list the exact training match IDs.

- [ ] **Step 1: Write failing report tests** for event-only training, no self-training, mixed-source rejection, empty/invalid training events, and unchanged Metrica `MatchBundle` behavior.
- [ ] **Step 2: Run `pytest tests/test_report.py tests/test_idsse_training.py -q`** and confirm the event-only input and isolation cases fail.
- [ ] **Step 3: Implement `TrainingMatch` handling** by validating only its events table and combining validated training events for `fit_xt`; do not retain training tracking frames or weaken selected-bundle validation.
- [ ] **Step 4: Verify report tests** with the focused pytest command and assert the report provenance lists only the supplied other-match IDs.
- [ ] **Step 5: Commit and push**: `git add src/soccer_tactics/analytics/report.py tests/test_report.py tests/test_idsse_training.py && git commit -m "feat: support event-only held-out training"`.

### Task 4: Create reproducible seven-match quality and held-out validation

**Files:**
- Create: `src/soccer_tactics/analytics/idsse_validation.py`
- Create: `scripts/validate_idsse.py`
- Create: `tests/test_idsse_validation.py`
- Create: `docs/validation/idsse-v1.json` (generated only after real-data acceptance)
- Modify: `.gitignore` only if needed to keep raw XML and temporary derived files outside Git.

**Interfaces:**
- Add `validate_idsse_dataset(root: str | Path) -> dict`, returning schema version, source/license references, every discovered candidate and its accepted/rejected status, quality summary, held-out training IDs, report counts/findings, and peak-memory measurement.
- CLI: `python scripts/validate_idsse.py --data-root data/raw/idsse --output docs/validation/idsse-v1.json`.
- Record a source fingerprint for every XML file using its relative name, byte size, and nanosecond modification time; Task 5 rejects a missing or stale fingerprint before listing IDSSE matches.
- The generated manifest is the sole readiness source consumed by Task 5; only candidates with both source-quality and held-out status `accepted` may be used for the dashboard or training.

- [ ] **Step 1: Write failing validation tests** for all-seven candidate accounting, deterministic output, at-least-two-match rule, every-held-out-excludes-self rule, same-source training, invalid xT calibration rejection, explicit rejection reasons, and fail-closed missing data.
- [ ] **Step 2: Run `pytest tests/test_idsse_validation.py -q`** and confirm validation API/CLI expectations fail.
- [ ] **Step 3: Implement the validator and CLI**. Parse one selected full-tracking bundle at a time; keep only accepted canonical event tables for the training set; run each accepted candidate held out against all other accepted IDSSE matches; serialize stable JSON with no raw observations. Measure peak memory for audit and one dashboard-style report and include both values.
- [ ] **Step 4: Verify synthetic validation** with `pytest tests/test_idsse_validation.py -q`, including fewer than two eligible matches and a candidate that has source-quality failures.
- [ ] **Step 5: Run the real-data gate** over all seven triplets using the command above. Require every candidate to have a clear status, at least two accepted matches, successful held-out reports for every accepted candidate, no unexplained canonical rows, source-checked scores/goals, and observed peak memory values. Keep the resulting JSON report in Git; never add XML or per-frame derived tracking.
- [ ] **Step 6: Commit and push** implementation plus the generated non-sensitive validation summary: `git add src/soccer_tactics/analytics/idsse_validation.py scripts/validate_idsse.py tests/test_idsse_validation.py docs/validation/idsse-v1.json .gitignore && git commit -m "feat: validate IDSSE held-out matches"` (omit `.gitignore` if unchanged).

### Task 5: Add a readiness-gated multi-provider catalogue and dashboard selection

**Files:**
- Create: `src/soccer_tactics/loaders/catalog.py`
- Modify: `app.py`
- Test: `tests/test_catalog.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Add `CatalogEntry` with `provider: str`, `match_id: str`, `source_path: str`, `display_name: str`, `team_names: dict[str, str]`, and `ready: bool` plus optional `reason: str`.
- Add `list_match_catalogue(metrica_root: str | Path, idsse_root: str | Path, validation_manifest: str | Path) -> list[CatalogEntry]`; Metrica games 1–2 remain available by their existing source checks, while IDSSE entries are included only when the manifest marks them accepted.
- Add provider-aware cached loading/report helpers in `app.py`; an IDSSE selected match uses `load_idsse_bundle` and `TrainingMatch` event tables from other accepted IDSSE IDs only. Metrica selected matches continue using their current Metrica held-out pair.

- [ ] **Step 1: Write failing catalogue and AppTest cases** for no IDSSE files, absent Floodlight, absent/stale manifest, accepted and rejected IDSSE entries, provider-unique IDs, source labels, training-source isolation, and unchanged Metrica match/team/evidence selection.
- [ ] **Step 2: Run `pytest tests/test_catalog.py tests/test_dashboard.py -q`** and confirm new catalogue behavior is not present.
- [ ] **Step 3: Implement the catalogue and selector**. Dashboard code consumes `CatalogEntry` values, never scans XML or infers readiness. Load full tracking only for the selected entry; cache manifests, metadata, selected bundles, event-only training data, and reports with bounded entries.
- [ ] **Step 4: Verify UI and catalogue tests** with the focused pytest command. Confirm the empty state is actionable, missing Floodlight recommends `pip install -e '.[idsse]'`, rejected candidates cannot be selected, and selected-match/team/evidence state does not leak across providers.
- [ ] **Step 5: Commit and push**: `git add src/soccer_tactics/loaders/catalog.py app.py tests/test_catalog.py tests/test_dashboard.py && git commit -m "feat: add validated IDSSE match selection"`.

### Task 6: Finish attribution, setup, and release verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/VALIDATION.md`
- Modify: `docs/demo/README.md` if screenshots are recaptured
- Modify: `tests/test_roadmap.py`
- Test: `tests/test_release_validation.py`

**Interfaces:**
- Document optional IDSSE setup, the external data directory, the validation command, the manifest gating rule, license, DFL attribution, and paper citation.
- Keep the IDSSE match selector hidden unless the checked-in manifest has passed the all-seven and held-out gates.
- Retain the V1 acceptance checkpoint and add V2 completion date only after the V2 real-data checks pass.

- [ ] **Step 1: Write/extend documentation tests** for the optional install command, raw-data exclusion, explicit DFL attribution and paper citation, reproducible validation command, and V2 acceptance record.
- [ ] **Step 2: Run focused docs/release tests** with `pytest tests/test_roadmap.py tests/test_release_validation.py -q` and confirm new documentation gates fail until completed.
- [ ] **Step 3: Update README, roadmap, and validation docs** with the installed-data workflow, source limitations, real-data audit outcome, memory measurements, and IDSSE attribution; add screenshot evidence for the accepted match selector if the environment has real data and Streamlit dependencies.
- [ ] **Step 4: Run final verification**: `pytest -q`; then run Metrica real-data release validation and dashboard smoke checks; run `python scripts/validate_idsse.py --data-root data/raw/idsse --output docs/validation/idsse-v1.json` and verify the generated file is clean and deterministic on a second run. Confirm Metrica setup, tests, and app launch work without Floodlight installed.
- [ ] **Step 5: Inspect repository contents** with `git status --short` and verify no raw XML, extracted tracking, or temporary validation data are staged.
- [ ] **Step 6: Commit and push**: `git add README.md docs/ROADMAP.md docs/VALIDATION.md docs/demo/README.md tests/test_roadmap.py tests/test_release_validation.py docs/validation/idsse-v1.json && git commit -m "docs: document IDSSE V2 validation"` (include only modified files).

## Execution Order and Release Gate

Run Tasks 1–6 in order. Each task gets a fresh implementer, a fresh reviewer, a focused test run, and a separate commit pushed to `codex/tactical-analyst-test` before starting the next task. The broad final review must use the strongest available review model and inspect the full branch against the spec. Do not mark V2 complete or expose unaccepted IDSSE matches if any of the seven source audits is missing, fewer than two matches pass both gates, any accepted held-out report trains on itself or another provider, the generated manifest is absent/non-deterministic, or the Metrica regression gate fails.
