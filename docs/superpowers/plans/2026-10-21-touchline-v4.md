# Touchline V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trustworthy possession, transition, defensive-action, and shot-pattern analysis to Touchline, with evidence-backed dashboard and local-chat access.

**Architecture:** Extend the existing Metrica loader for the third public match and recognized defensive events, then build sequence analytics independent from Streamlit. Enrich `MatchReport` and its citations before integrating dashboard panels and chat tools. Keep IDSSE origin-only and use match-level holdouts for validation.

**Tech Stack:** Python 3.10+, pandas, NumPy, pytest, Streamlit, Plotly, optional Kloppy EPTS adapter.

**Spec:** `docs/superpowers/specs/2026-10-21-touchline-v4-design.md`

## Global Constraints

- Spatial and sequence analysis uses validated Metrica event and tracking data. IDSSE remains limited to factual source-origin queries.
- Extend Metrica ingestion to the local public Sample Game 3 EPTS/FIFA data as well as Games 1–2. Keep source data outside Git and do not download data or models automatically.
- Include possession sequences, transitions, and defensive actions only where source semantics can be mapped explicitly. Preserve source provenance and omit ambiguous action labels.
- Define a transition as a new observed possession following an opponent turnover or successful defensive action. Analyze the first 10 seconds only; never infer that the transition caused a later shot.
- Add shot-location and outcome summaries with sample counts. Do not fit learned shot-quality estimates in V4.
- Keep tactical recommendations deterministic and cautious. Each recommendation needs traceable supporting moments and a stated tradeoff.
- Compare additions to simpler baselines with match-level holdouts. Report when the small number of independent matches limits conclusions.
- Extend the existing dashboard and local assistant. Chat claims about new measurements must cite allowed evidence from the active Metrica report; unsupported causal claims must be declined.
- Keep at most three top-level deterministic findings. New summaries remain available in the dashboard and local chat even when they do not qualify as a finding.

## Review Focus

- Game 3 metadata, EPTS tracking clocks, coordinates, and event clocks disagree or contain missing values; test alignment, orientation, and explicit source-quality failure.
- Defensive subtypes that represent fouls, blocks, or ambiguous challenges must not be mislabeled as won tackles or recoveries; test accepted and excluded source codes.
- Same-frame events and repeated frames across periods must sort deterministically and never merge different periods or matches; test collision cases.
- Transitions that cross the 10-second or period boundary must be excluded consistently; test exact boundary and just-over boundary.
- Sparse shots or defensive events must yield sample counts and unavailable summaries without manufactured findings; test empty and low-count matches.

---

### Task 1: Metrica Game 3 and Defensive Event Ingestion

**Files:** `src/soccer_tactics/loaders/metrica.py`, `src/soccer_tactics/loaders/catalog.py`, `src/soccer_tactics/loaders/bundle.py`, `src/soccer_tactics/schema.py`, `pyproject.toml`, focused loader tests.

**Interfaces:** Preserve `load_metrica_game(root, game)` return compatibility. Add Game 3 through `load_metrica_bundle(root, game=3)` and the catalogue, using a documented optional `metrica-epts` extra. The canonical event types remain `pass`, `shot`, `tackle`, `interception`, and `recovery`; do not add ambiguous source subtypes.

- [ ] Add failing tests for local Game 3 metadata/score, aligned tracking and events, event direction, sample-rate metadata, and catalogue inclusion.
- [ ] Add table-driven raw-event tests for recognized defensive subtypes and excluded ambiguous/foul/block subtypes, including quality counts.
- [ ] Implement Game 3 EPTS/event loading through Kloppy’s Metrica loaders and map both sources into the existing canonical bundle.
- [ ] Extend CSV Metrica loading to retain only explicitly recognized challenges/recoveries while preserving existing pass, penalty, goal, and orientation behavior.
- [ ] Run focused loader and schema tests, then all tests that do not require optional dependency; commit `feat: load third Metrica match and defensive events`.

### Task 2: Possession and Transition Analytics

**Files:** create `src/soccer_tactics/analytics/sequences.py`; create `tests/test_sequences.py`.

**Interfaces:** `segment_possessions(events: pd.DataFrame) -> pd.DataFrame`; `summarize_transitions(events: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame`. Inputs are canonical Metrica events with stable event IDs. Output rows include match, period, team, clocks, source event IDs, terminal reason, duration, completed final-third entries, shots, and available xT totals.

- [ ] Write hand-computed tests for ordinary possession, turnover, shot, successful defense, event ordering, same-frame ordering, period and match isolation, and empty input.
- [ ] Write transition tests at exactly 10 seconds, just over 10 seconds, following a successful defense, and at a period boundary.
- [ ] Run focused tests to establish the expected failures.
- [ ] Implement deterministic segmentation and transition windows without inferring carries or causal shot sequences.
- [ ] Run tests and commit `feat: model observed possessions and transitions`.

### Task 3: Report Metrics, Evidence, and Tactical Hypotheses

**Files:** `src/soccer_tactics/analytics/report.py`, related analytics modules as needed, report/player/chat-tool tests.

**Interfaces:** Preserve existing `analyze_match(match_bundle, team, training_matches) -> MatchReport` call shape and existing fields. Add `possessions`, `transitions`, `defensive_actions`, and `shot_summaries` report fields. Keep existing report fields positional-compatible by appending defaults. Each evidence item references a source match, period, frame, and timestamp.

- [ ] Add hand-computed tests for possession/transition team comparisons, defensive counts and exposure, shot channel counts/goals, evidence IDs, and existing three-finding ceiling.
- [ ] Add tests that selected-match xT never trains on the selected match and that training uses all other validated Metrica matches available.
- [ ] Add tests for unknown scores, no defensive events, low shot counts, and insufficient pattern evidence.
- [ ] Integrate the sequence outputs, defensive-action summaries, and shot-location/outcome summaries; do not add a learned shot-quality model.
- [ ] Add deterministic findings only when source-backed thresholds are met; every suggestion includes citations and a tradeoff, and no claim establishes causation.
- [ ] Run focused report/impact/agent-tool tests and commit `feat: add evidence-backed V4 report metrics`.

### Task 4: Dashboard and Local Chat Integration

**Files:** `app.py`, `src/soccer_tactics/dashboard/charts.py`, `src/soccer_tactics/agent/tools.py`, `src/soccer_tactics/agent/chat.py`, dashboard and chat tests.

**Interfaces:** Add V4 report facts to context-bound Metrica tools only. Preserve IDSSE’s existing origin-only tool contract. Dashboard match selection uses the current validated Metrica catalogue, and chat citations are restricted to active report evidence.

- [ ] Add dashboard tests for possession/transition navigation, defensive and shot summaries, empty data, and source switching.
- [ ] Add chat tests for V4 fact retrieval, evidence citations, match/team isolation, and causal-question abstention.
- [ ] Add views for possession/transition sequence evidence, defensive actions, and shot location/outcomes with sample counts and nearby limitations.
- [ ] Expose the new report facts through local tools and validate all returned citations against the selected context.
- [ ] Smoke-test both teams across Metrica Games 1–3; commit `feat: expose V4 analysis in dashboard and chat`.

### Task 5: Held-Out Validation and User Documentation

**Files:** `scripts/validate_real_data.py`, `docs/VALIDATION.md`, `docs/ROADMAP.md`, `README.md`, validation tests.

- [ ] Extend real-data validation to export both teams for Games 1–3 and verify score, event totals, source orientation, period integrity, sample rates, and other-match-only xT provenance.
- [ ] Compare sequence/defensive/shot summaries with simpler match-level baselines; include sample sizes and avoid significance claims unsupported by three matches.
- [ ] Document the optional Kloppy installation extra, local Game 3 data layout, dataset attribution, limitations, and one-command dashboard launch.
- [ ] Update V4 roadmap status only after all validation gates pass; otherwise leave V4 in progress and record the missing gate.
- [ ] Run the full suite with the optional `metrica-epts` extra and real local source files; commit `docs: document and validate V4 analysis`.

### Task 2 output columns

`segment_possessions` returns exactly the possession columns listed in the spec. `summarize_transitions` returns exactly the transition columns listed in the spec. Preserve stable source order for events sharing a frame. A transition starts on a successful tackle/interception/recovery, or at the first opponent event after an unsuccessful pass, and stops at possession end, 10 seconds, or period end (whichever comes first); if the opponent has no event before the window/period ends, emit no row.

### Task 3 finding selection

A candidate pattern requires at least five denominator events and at least three events supporting the highlighted pattern. Return at most one per category, sorted by support count descending with stable tie order `transition`, `defense`, `buildup`, `progression`, `danger`, then keep the top three. This is a display safeguard, not a significance test. Retain all report summaries independent of finding selection.
