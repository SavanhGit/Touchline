# Touchline V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trustworthy possession, transition, defensive-action, and shot-pattern analysis to Touchline, with evidence-backed dashboard and local-chat access.

**Architecture:** Extend the existing Metrica loader for the third public match and recognized defensive events, then build sequence analytics independent from Streamlit. Enrich `MatchReport` and its citations before integrating dashboard panels and chat tools. Keep IDSSE origin-only and use match-level holdouts for validation.

**Tech Stack:** Python 3.10+, pandas, NumPy, pytest, Streamlit, Plotly, optional `kloppy>=3.17,<4` EPTS adapter.

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

**Interfaces:** Preserve `load_metrica_game(root, game)` return compatibility. Add Game 3 through `load_metrica_bundle(root, game=3)` and the catalogue. Add optional extra `[metrica-epts] = ["kloppy>=3.17,<4"]`; local loading must not fetch source files. The canonical event types remain `pass`, `shot`, `tackle`, `interception`, and `recovery`; do not add ambiguous source subtypes. Map only `CHALLENGE` subtype `TACKLE-WON` to `tackle/success=True`, `TACKLE-LOST` to `tackle/success=False`, `RECOVERY/INTERCEPTION` to `interception/success=True`, and `RECOVERY/THEFT` to `recovery/success=True`. Exclude fault/advantage, aerial/ground, dribble, blank, saved, and blocked subtypes and count each exclusion reason.

- [x] Add failing tests `test_game3_epts_to_canonical_bundle`, `test_game3_event_clocks_align_to_tracking`, and `test_game3_catalogue_includes_only_complete_sources`; assert source score is 0–2, canonical columns validate, and event orientation agrees with tracking.
- [x] Add `test_defensive_event_subtype_mapping` and `test_ambiguous_defensive_events_are_excluded_and_counted` for the documented accepted/rejected subtype mapping.
- [x] Implement Game 3 EPTS/event loading through Kloppy’s Metrica loaders and map both sources into the existing canonical bundle.
- [x] Extend CSV Metrica loading to retain only explicitly recognized challenges/recoveries while preserving existing pass, penalty, goal, and orientation behavior.
- [x] Run focused loader and schema tests, then all tests that do not require optional dependency; commit `feat: load third Metrica match and defensive events`.

### Task 2: Possession and Transition Analytics

**Files:** create `src/soccer_tactics/analytics/sequences.py`; create `tests/test_sequences.py`.

**Interfaces:** `segment_possessions(events: pd.DataFrame, pass_metrics: pd.DataFrame) -> pd.DataFrame`; `summarize_transitions(events: pd.DataFrame, possessions: pd.DataFrame, pass_metrics: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame`. Inputs are same-match canonical events with stable event IDs and the `analyze_match` pass table with `event_id`, `entry`, and `success`. Possession output columns: `match_id`, `period`, `team`, `possession_id`, `start_time_s`, `end_time_s`, `start_event_id`, `end_event_id`, `event_ids`, `terminal_reason`, `duration_s`, `passes`, `completed_passes`, `final_third_entries`, `shots`, `goals`. Transition output columns: `match_id`, `period`, `team`, `transition_id`, `start_time_s`, `end_time_s`, `source_turnover_event_id`, `event_ids`, `final_third_entries`, `shots`, `goals`, `window_s`. Neither sequence table claims xT or shot quality.

- [x] Add `test_possession_counts_and_terminal_reason_hand_calculation`, `test_failed_pass_and_successful_defense_split_possessions`, `test_same_frame_order_is_stable`, `test_possessions_never_cross_period_or_match`, and `test_empty_events_return_empty_tables`.
- [x] Add `test_transition_includes_exact_window_boundary`, `test_transition_excludes_event_after_window`, `test_transition_starts_at_successful_defensive_action`, `test_no_transition_without_observed_opponent_event`, and `test_transition_never_crosses_period`.
- [x] Run focused tests to establish the expected failures.
- [x] Implement deterministic segmentation and transition windows without inferring carries or causal shot sequences.


- [x] Use the fixed transition rule: after an unsuccessful pass, start at the first opponent event observed within the 10-second window; after a successful defensive action, require a later event from that same team in the same observed possession within 10 seconds to confirm retained possession. Deduplicate both triggers by new possession, preferring the defensive action as the source when both identify the same start. Stop at possession end, 10 seconds, or period end, whichever comes first. Preserve source order for events sharing a frame.
- [x] Run `pytest tests/test_sequences.py -q` and commit `feat: model observed possessions and transitions`.

### Task 3: Report Metrics, Evidence, and Tactical Hypotheses

**Files:** `src/soccer_tactics/analytics/report.py`, related analytics modules as needed, report/player/chat-tool tests.

**Interfaces:** Preserve existing `analyze_match(match_bundle, team, training_matches) -> MatchReport` call shape and existing fields. Add `possessions`, `transitions`, `defensive_actions`, and `shot_summaries` report fields. Keep existing report fields positional-compatible by appending defaults. Each evidence item references a source match, period, frame, and timestamp.

- [x] Add `test_report_v4_summaries_match_hand_calculated_counts`, `test_defensive_actions_per_90_use_observed_tracking_exposure`, `test_shots_are_summarized_by_attacking_channel`, and `test_findings_stay_capped_at_three_with_stable_ties`.
- [x] Add `test_held_out_match_never_trains_its_xt_grid` and `test_all_other_catalogued_metrica_matches_train_report` for Games 1–3.
- [x] Add `test_unknown_score_and_empty_defense_are_explicit`, `test_low_shot_count_has_no_quality_claim`, and `test_insufficient_pattern_evidence_returns_fewer_findings`.
- [x] Integrate the sequence outputs, defensive-action summaries, and shot-location/outcome summaries; do not add a learned shot-quality model.
- [x] Add deterministic findings only when source-backed thresholds are met; every suggestion includes citations and a tradeoff, and no claim establishes causation.


- [x] Use finding denominator >= 5 and support >= 3. Keep one per category, sort by supporting-event count descending with stable tie order `transition`, `defense`, `buildup`, `progression`, `danger`, and return at most three; this is a display safeguard, not significance. Keep all summary tables even when patterns do not qualify.
- [x] Run `pytest tests/test_report.py tests/test_impact.py tests/test_agent_tools.py -q` and commit `feat: add evidence-backed V4 report metrics`.

### Task 4: Dashboard and Local Chat Integration

**Files:** `app.py`, `src/soccer_tactics/dashboard/charts.py`, `src/soccer_tactics/agent/tools.py`, `src/soccer_tactics/agent/chat.py`, dashboard and chat tests.

**Interfaces:** Add `sequence_summary`, `defensive_summary`, and `shot_summary` to context-bound Metrica tools only, each returning the report's allowed `citation_ids`. Preserve IDSSE’s existing origin-only tool contract. Dashboard match selection uses the current validated Metrica catalogue, and chat citations are restricted to active report evidence.

- [x] Add `test_dashboard_v4_sections_render_for_metrica`, `test_dashboard_v4_empty_summaries_are_explained`, and `test_idsse_stays_origin_only_after_v4`.
- [x] Add `test_chat_returns_cited_v4_report_facts`, `test_v4_chat_citations_follow_selected_match_and_team`, and `test_chat_abstains_on_causal_transition_question`.
- [x] Add views for possession/transition sequence evidence, defensive actions, and shot location/outcomes with sample counts and nearby limitations.
- [x] Expose the new report facts through local tools and validate all returned citations against the selected context.
- [x] Smoke-test both teams across Metrica Games 1–3; commit `feat: expose V4 analysis in dashboard and chat`.

### Task 5: Held-Out Validation and User Documentation

**Files:** `scripts/validate_release.py`, `scripts/validate_cross_match.py`, `docs/VALIDATION.md`, `docs/ROADMAP.md`, `README.md`, validation tests.

- [x] Extend real-data validation to export both teams for Games 1–3 and verify score, event totals, source orientation, period integrity, sample rates, and other-match-only xT provenance.
- [x] Run leave-one-match-out folds for Games 1–3; compare each held-out transition shot rate with the same team’s overall possession shot rate and held-out defensive actions per 90 with the pooled dataset-wide rate for that action type across the other two matches, labeled as a cross-match dataset baseline rather than a same-team comparison. Report shots and goals by channel against the team-wide shot/goal totals, with raw counts and denominators; no significance or learned shot-quality claims.
- [x] Audit every tactical suggestion in each held-out report: confirm its cited source events support the observation and its stated tradeoff is visible; record descriptive accuracy separately and do not claim the suggestion would change the result. Document the optional Kloppy installation extra, local Game 3 data layout, source attribution, limitations, and the one-command dashboard launch.
- [x] Update V4 roadmap status only after all validation gates pass; otherwise leave V4 in progress and record the missing gate.
- [x] Run the full suite with the optional `metrica-epts` extra and real local source files; commit `docs: document and validate V4 analysis`.
