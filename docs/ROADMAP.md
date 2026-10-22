# Touchline roadmap

The first release is **Touchline**, a local post-match dashboard. Its question is: “What patterns contributed to the result, and what adjustments might have helped?”

## V1 — Evidence before explanation

**Completed:** 2026-10-20 at 12:07:49 PM EDT — release checkpoint `c8d032b`, pushed to `codex/tactical-analyst-test` after the 243-test suite and real-match release validation passed.

- Metrica sample games 1–2; both home and away perspectives.
- Complete goal context alongside penalty-excluded modeling events.
- Held-out xT, buildup turnover patterns, direct final-third entries, and opponent danger channels.
- Up to three deterministic findings, with source event references, hypotheses, and tradeoffs.
- Pitch plots, player summaries, and period-aware tracking reconstruction.
- Real-data regression tests, checked example report, screenshots, and a captioned walkthrough.

The safeguards of five denominator events and three supporting events are display rules, not confidence intervals. Reports can contain fewer than three findings. No finding establishes causation or promises a different result.

## V2 — More matches

The IDSSE implementation is limited to an **origin-only IDSSE catalogue**. Seven real XML matches have a recorded source-quality audit, and the dashboard exposes only origin views when all seven candidates are present in the manifest with current XML fingerprints.

**V2 origin-only release completed: 2026-10-21 11:30 EDT.** The final verification passed 337 tests, validated all four real Metrica team reports, and freshly audited IDSSE at 7/7 accepted matches with 21/21 current XML fingerprints, zero errors, and an exact parsed match to the checked-in manifest. The IDSSE selector and source-origin view were manually checked in the dashboard.

This release preserves IDSSE source event coordinates, timestamps, and event-code provenance. IDSSE tracking synchronization, attack direction, pass endpoints, IDSSE xT and progression, and held-out IDSSE evaluation remain deferred. Metrica retains its existing endpoint-based analysis and training. IDSSE data is attributed to the Deutsche Fußball Liga (DFL) under CC BY 4.0. Citation: Bassek et al., “An integrated dataset of spatiotemporal and event data in elite soccer,” *Scientific Data* 12, 195 (2025), [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). See the [setup and source instructions](../README.md#optional-idsse-match-catalogue) and [validation notes](VALIDATION.md).

## V3 — Local question-answering assistant

**Completed: 2026-10-21 12:50 PM EDT.** The optional Streamlit assistant uses local Ollama inference, defaults to `qwen3:8b`, supports `TOUCHLINE_OLLAMA_MODEL`, and never downloads models or calls a paid provider. Context-bound tools replace shared global agent state; Metrica responses use selected-report facts and citations, IDSSE responses remain limited to source event origins, and chat resets when the selected source, match, or team changes. See the [local assistant setup](../README.md#optional-local-match-assistant).

Owner-machine acceptance used the real Metrica and IDSSE files and installed Qwen3 8B: the losing Metrica side's buildup result returned **22/76 (28.9%)** with a resolvable match-event citation; a guaranteed-win question was declined; IDSSE returned a factual source-origin answer with its source ID and declined pass-destination and event-to-tracking synchronization questions. The Streamlit app was exercised with the real catalogue, including a Metrica-to-IDSSE match change and chat-history reset. The full real-data suite passed **368 tests**, and release validation exported all four Metrica team reports with held-out matches used for xT training. Local model response quality remains sensitive to hardware and the selected model.

## V4 — Richer tactical analysis

**Completed: October 21, 2026.** Games 1–3 support observed possession sequences, clipped 10-second transitions, explicitly recognized defensive actions, and attacking-channel shot counts/outcomes. V4 does not fit learned shot-quality models. See [V4 validation](VALIDATION.md#v4-held-out-validation-2026-10-21) and [held-out counts](validation/v4/cross-match-validation.json).

The V4 release gate is complete. Source and descriptive audits remain separate from tactical suggestion effectiveness; validation cannot establish that a suggested adjustment changes a result. Local Ollama response time still depends on the selected model and hardware.

## V5 — Video-derived player tracking

**In progress: local-file player tracking first; real-footage acceptance gate pending.** The local review/export workspace and evaluator are implemented. Inference runs locally with no required paid API. See the [V5 design](superpowers/specs/2026-10-21-v5-local-player-tracking-design.md), [optional setup and model terms](../README.md#optional-local-video-player-tracking-v5), and [evaluation recipe and evidence status](validation/v5/README.md).

### First milestone: player tracking

- The **Video tracking** workspace accepts a short, continuous local MP4 with one home attack direction. Explicit optional CV installation and local player/pitch weights are required; no startup or inference download occurs.
- Player detection, clip-local ByteTrack IDs and independent frame calibration produce centered 105 × 68 m pitch locations. Manual home/away mapping and optional roster names accompany canonical tracking export; automatic jersey/name recognition is deferred.
- Failed calibration, rejected detections and off-camera players leave gaps without interpolation. Saved runs contain tracking, diagnostics, metadata and overlay under ignored `data/processed/video_runs/<run_id>/`; raw videos/annotations and weights remain outside Git under ignored `data/raw/` and `data/models/`, or external local storage.
- The evaluator accepts local SoccerNet/MOT annotations and independent manual pitch points. HOTA/DetA/AssA, misses, identity switches and annotation-derived location/calibration errors still need a real run. Saved pipeline runs do not retain the image boxes or per-frame transforms needed for all metrics; the evaluation notes specify those separate inputs.
- Annotated-footage evaluation, representative visual review, real browser overlay playback and native Windows Python 3.15 inference remain pending. Automated fake-model tests and cross-target dependency resolution do not satisfy those gates. Exact Google Drive checkpoint terms remain uncertain as documented in the setup notes; upstream code licensing is not checkpoint-license verification.

### Follow-on milestones

- After local-file processing works, add best-effort URL import for sources the app can access. Feed retrieved video through the same processing path and show an actionable error with local-file fallback when retrieval fails.
- Add ball tracking and then event extraction as later V5 milestones. Keep live ingestion and incremental analysis in V6.

URL import, ball tracking, event extraction, and live analysis are not implemented. The first player-tracking milestone is complete only after the local-file canonical export, manual mapping, annotated-footage evaluation and visual review are recorded. No accuracy baseline has been measured yet. Video-only runs have no event table and cannot supply score/tactical match reports or assistant evidence. Follow-on V5 milestones do not block this first gate; live analysis stays in V6.

## V6 — Live analysis

Stream validated data through incremental analytics. Measure latency and data coverage. Use past-only motion features: the current centered smoothing is suitable for post-match analysis and must not be reused as a live feature. Validate without future frames.

Public hosting follows a local portfolio demo and a dataset redistribution review. No hosting, authentication, API costs, live video, or learned forecast is required for V1.

V1 portfolio artifacts: [featured example](demo/featured-analysis.md), [four screenshots and capture notes](demo/README.md), and a [80-second edited browser walkthrough](demo/walkthrough.mp4). The walkthrough packages 240 sequential real browser captures at 3 fps; it is edited low-frame-rate browser footage, not match video or seamless high-frame-rate screen recording. Reproducible Windows setup is in the [main README](../README.md).
