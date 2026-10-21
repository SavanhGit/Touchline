# Touchline V4 Design: Richer Tactical Analysis

## Goal

Help a user inspect match patterns that may have contributed to a result and explore evidence-backed tactical adjustments, with clear uncertainty and source-linked moments.

## Scope and evidence rules

- Spatial and sequence analysis uses validated Metrica event and tracking data. IDSSE remains limited to factual source-origin queries.
- Extend Metrica ingestion to the local public Sample Game 3 EPTS/FIFA data as well as Games 1–2. Keep source data outside Git and do not download data or models automatically.
- Include possession sequences, transitions, and defensive actions only where source semantics can be mapped explicitly. Preserve source provenance and omit ambiguous action labels. Accept only Metrica `CHALLENGE/TACKLE-WON`, `CHALLENGE/TACKLE-LOST`, `RECOVERY/INTERCEPTION`, and `RECOVERY/THEFT` with their direct outcomes; exclude ambiguous challenge and recovery subtypes.
- Define a transition as a new observed possession following an opponent turnover or successful defensive action. Analyze the first 10 seconds only; never infer that the transition caused a later shot.
- Add shot-location and outcome summaries with sample counts. Do not fit learned shot-quality estimates in V4.
- Keep tactical recommendations deterministic and cautious. Each recommendation needs traceable supporting moments and a stated tradeoff.
- Compare additions to simpler baselines with match-level holdouts. Report when the small number of independent matches limits conclusions.
- Extend the existing dashboard and local assistant. Chat claims about new measurements must cite allowed evidence from the active Metrica report; unsupported causal claims must be declined.

## Data and report contracts

- `load_metrica_bundle(root, game)` accepts Games 1–3. Game 3 support uses the optional `metrica-epts` dependency extra and local event JSON, metadata XML, and tracking text files.
- Canonical Metrica events retain pass and shot behavior and add only recognized tackle, interception, and recovery events. The canonical `success` field means the source-indicated outcome for that event type; unsupported or ambiguous subtypes are excluded and counted in data-quality metadata.
- `segment_possessions(events, pass_metrics)` returns period-scoped sequences with team, start/end clocks, event IDs, and terminal reason. Boundaries are team changes, unsuccessful passes, shots, successful defensive actions, and period ends.
- `summarize_transitions(events, possessions, pass_metrics, window_s=10.0)` returns event-addressable transition rows and summaries; it must not cross periods or matches.
- `MatchReport` exposes possession, transition, defensive-action, and shot summaries in addition to existing fields. New report evidence uses the existing match/period/frame/time event identity.
- Keep at most three top-level deterministic findings. New summaries remain available in the dashboard and local chat even when they do not qualify as a finding.

## Validation and safety

- Match-level held-out evaluation excludes the selected match from any fitted xT or comparison baseline.
- Test unknown defensive subtypes, absent defensive events, empty possessions/transitions, period boundaries, same-frame events, unknown scores, and low shot counts.
- Do not claim statistical significance or causality from descriptive counts or the three-match Metrica sample.

## Exact report outputs

- `possessions` is one row per observed same-team, same-period sequence. Columns: `match_id`, `period`, `team`, `possession_id`, `start_time_s`, `end_time_s`, `start_event_id`, `end_event_id`, `event_ids`, `terminal_reason`, `duration_s`, `passes`, `completed_passes`, `final_third_entries`, `shots`, `goals`.
- `transitions` is one row per qualifying new possession after a turnover or successful defensive action, clipped to the earlier of 10 seconds or possession end. Columns: `match_id`, `period`, `team`, `transition_id`, `start_time_s`, `end_time_s`, `source_turnover_event_id`, `event_ids`, `final_third_entries`, `shots`, `goals`, `window_s`.
- `defensive_actions` reports per team and action type: total actions, successful outcomes, unsuccessful outcomes, observed tracking minutes, and actions per 90 observed tracking minutes. `shot_summaries` reports team/channel shots and goals in the team's attacking direction.
- A transition begins on a successful tackle/interception/recovery row only when a later same-team event in that observed possession confirms retained possession within 10 seconds; alternatively it begins at the first opponent event observed within 10 seconds after an unsuccessful pass. Emit no row without that observed confirmation. Clip events to the selected possession and its 10-second window. No transition bridges periods or matches. Neither sequence table claims xT totals or shot quality.
- Candidate findings use a denominator of at least five and at least three events supporting the highlighted pattern. Keep at most one finding per category, rank by supporting-event count descending with stable category order (`transition`, `defense`, `buildup`, `progression`, `danger`), then return at most three. This is a display rule, not a significance test.
- The optional dependency extra is `[metrica-epts]`; it contains Kloppy. Game 3 loading consumes local files only and raises an actionable optional-dependency error if the extra is missing.


The supported optional dependency range is `kloppy>=3.17,<4`, based on the Metrica EPTS/event loader interface and Python 3.10 compatibility.
