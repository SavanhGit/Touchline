# Touchline V2: IDSSE Match Catalogue Design

## Status and origin

This design follows the V2 direction in [`docs/ROADMAP.md`](../ROADMAP.md): validate the existing IDSSE XML adapter against real match files, then add IDSSE matches to the analysis catalogue only after data-quality and held-out-evaluation gates pass. V1 remains the Metrica games 1–2 release and is complete on the test branch.

The project remains a local portfolio tool for exploring post-match patterns and cautious tactical hypotheses. V2 makes more checked matches selectable; it does not claim that a larger catalogue establishes why a team lost or proves a proposed adjustment would work.

## Goals

- Load complete IDSSE `matchinformation`, `events_raw`, and `positions_raw_observed` XML triplets into validated `MatchBundle` objects.
- Preserve the existing `load_idsse_game(root, match_id) -> (tracking, events)` API while adding a bundle interface for score, goals, team labels, source metadata, frame rate, and quality results.
- Establish the event and tracking time relationship, pitch dimensions and coordinate units, per-period attack direction, event semantics, team/player identity, and sampling rate from the actual source records.
- Evaluate every accepted IDSSE match with xT fit from the other IDSSE matches only. Record one reproducible result per held-out match.
- Add accepted IDSSE matches to the dashboard catalogue after all data gates pass, while keeping Metrica reports and source-specific training sets intact.
- Keep raw XML and other downloaded match data outside Git. Attribute the Deutsche Fußball Liga (DFL) and cite the dataset paper wherever IDSSE-derived results are shown.

## Non-goals

- Do not add Metrica Sample Game 3's FIFA/EPTS and JSON format in this milestone.
- Do not treat unvalidated matches as selectable or use them as xT training data.
- Do not add arbitrary user uploads, automatic data downloads, live/video tracking, public hosting, chat, or new tactical models.
- Do not silently infer source event meanings, match scores, team identities, coordinate transforms, or event-to-tracking matches.

## Data source and optional dependency

IDSSE is a separate source from the Metrica sample. Its official repository describes seven full Bundesliga matches with match information, event data, and TRACAB positions, and requires DFL credit plus the cited paper under CC BY 4.0. The public XML families are `matchinformation`, `events_raw`, and `positions_raw_observed`. The project already has `load_idsse_game` and an optional Floodlight dependency; these are starting points, not verified compatibility claims.

Keep Floodlight optional. Metrica-only setup and dashboard use must continue to work without it. IDSSE commands and UI states must explain how to install `.[idsse]` when Floodlight is absent.

Primary sources:

- IDSSE repository, source files, and license: <https://github.com/spoho-datascience/idsse-data>
- Dataset paper: <https://doi.org/10.1038/s41597-025-04505-y>
- Public dataset record: <https://api.figshare.com/v2/articles/28196177>
- Floodlight XML reader used by the existing adapter: <https://github.com/floodlight-sports/floodlight>

## Architecture and data flow

**IDSSE XML triplet → source parser → validated canonical tracking/events and complete goals → `MatchBundle` → per-match quality report → held-out analysis → catalogue and dashboard.**

Keep `load_idsse_game` compatible. Add `load_idsse_bundle(root, match_id)` beside the existing Metrica bundle loader. Extend `MatchBundle` with a default-empty `metadata` mapping for source match date, competition, source team IDs, source pitch dimensions, and source team labels; preserve canonical `home`/`away` keys and keep display labels in `team_names`. The IDSSE bundle must use the same canonical `validate_bundle`, `inspect_quality`, `analyze_match`, and evidence contracts as Metrica. Source-specific parsing and semantics remain in `loaders/idsse.py` and a focused IDSSE bundle module; dashboard code must consume catalogue entries, not parse XML.

Catalogue entries identify provider, stable match ID, source path, display labels, readiness, and any validation reason. `list_idsse_matches` continues to return only complete file triplets. A separate validation status determines which parsed matches are safe to show in the dashboard and which can train a report. Missing/incomplete files and missing optional dependencies produce clear actionable errors. Keep catalogue metadata small and lazy-load full tracking only for the selected report; held-out training must use validated canonical event tables rather than retaining the six other matches' full tracking in memory. Measure and record peak memory for catalogue validation and dashboard analysis before enabling the IDSSE selector.

Derive period, frame rate, pitch size, team/player identity, and attack direction from source metadata/readers. Convert coordinates to centered metres and rotate each period so home attacks toward +x. Event timestamps must be matched only within their own period to observed tracking timestamps. Assign an event frame only when the nearest observed tracking time is within half a source frame; preserve source time and report the alignment error distribution. Do not fabricate missing frames or cross-period matches.

Map only documented IDSSE event types to canonical modeling passes and shots. Record the source-code mapping and treatment of outcomes, crosses, own goals, penalties, and missing coordinates. Extract complete scored goals independently of penalty-excluded modeling shots. Unknown event codes remain visible in quality counts and are not silently relabeled as passes, shots, or goals.

## Quality and analysis requirements

For every source match, produce a machine-readable quality record with:

- complete triplet discovery and XML parse results;
- source and canonical pitch dimensions, coordinate ranges, and applied unit/translation/orientation transforms;
- source sampling rates by period, inferred rates, frame gaps, duplicate object keys, conflicting clocks, and missing/nonfinite positions;
- event counts by source code and canonical type, unmatched/excluded event counts, event-to-tracking alignment error, and event tracking coverage;
- home/away team sheets, player identity coverage, full-time score, and goal/penalty identities compared with source events.

Reject structural ambiguity, nonfinite retained positions, duplicate object keys, conflicting clocks, unknown period direction, and invalid event-to-tracking matches. Retain unmatched or unsupported source rows in quality totals with an explicit reason; never count them as report evidence. Only canonical pass/shot records with documented semantics and valid spatial/time keys enter V2 reports.

Analyze each candidate IDSSE match with a calibration fitted on all other accepted IDSSE matches. Reject self-training and mixed-source training in V2. Produce one deterministic output per held-out match with source IDs, training IDs, quality summary, team scores, report counts, and findings. If the remaining training matches cannot produce a valid xT surface, that held-out match fails evaluation and is not added to the catalogue.

## Dashboard behavior

The match selector lists Metrica games 1–2 and only IDSSE matches that pass source validation and held-out evaluation. Match IDs remain unique across providers. Selecting an IDSSE match loads training data from other IDSSE matches only; selecting a Metrica match retains its existing Metrica training behavior. Team selection uses the source's home/away labels with available names; source player IDs remain unchanged.

The same four V1 views and evidence safeguards apply to every selectable match. Show IDSSE attribution in the app and exported validation artifacts. Show a readable empty state when no IDSSE data is installed and a clear install instruction when the optional Floodlight dependency is missing. Metrica-only behavior must continue unchanged.

## Verification and acceptance

- Synthetic tests cover XML triplet discovery, missing/incomplete files, malformed metadata, absent optional dependency, period clock resets, repeated frame IDs across periods, per-period orientation, coordinate units, event mapping, penalties/goals, sampling rates, unmatched timestamps, and match-bundle provenance.
- Real-data acceptance audits all seven IDSSE match triplets from an external local data directory. No raw XML is committed. A missing or skipped IDSSE real-data gate cannot count as success.
- At least two matches must pass the source and quality gates for leave-one-match-out evaluation. Every accepted match has complete source-checked scores/goals, a reproducible quality record, no unexplained canonical events, and a successful held-out report trained only on other accepted IDSSE matches. Any rejected candidate has its reason and quality findings in the validation output and is excluded from the catalogue.
- Compare canonical outputs against Floodlight's source objects for sampled frames and every selected evidence event; verify event time and coordinates in their source period.
- Re-run all four Metrica team reports and existing dashboard/browser checks to detect regressions. Confirm Metrica-only installation, tests, and app launch work without Floodlight installed.
- Do not expose IDSSE in the dashboard until every match offered for selection satisfies the gates above.

## Attribution and limitations

Attribute the Deutsche Fußball Liga (DFL) and cite Bassek et al., “An integrated dataset of spatiotemporal and event data in elite soccer,” *Scientific Data* 12, 195 (2025). Retain the IDSSE dataset license and source links in the README, roadmap, and generated validation report. Keep raw source data external to Git.

IDSSE event coverage is limited to the documented mapped types. Reported evidence and suggestions remain descriptive. Any source synchronization, identity, or visibility limitations found during validation must be shown with the affected match and may disqualify that match from catalogue use.
