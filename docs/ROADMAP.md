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

## V3 — Question-answering agent

Use the same report engine and evidence IDs as tools. Questions must be scoped to the selected match and team. Evaluate answers for numeric fidelity, evidence references, refusal of unsupported claims, and correct limits. Choose model provider and budget then; the dashboard remains usable without an API key. Replace shared global agent state with an explicit per-session analysis context before exposing chat.

## V4 — Richer tactical analysis

Add possession sequences, transitions, defensive actions, and better shot models only when source data supports them. Compare each addition against simpler baselines on held-out matches. Validate tactical suggestions separately from descriptive metric accuracy.

## V5 — Recorded video

Integrate a video pipeline such as Roboflow Sports through the existing canonical adapter. Evaluate identity switches, missing players, pitch calibration, event extraction, and location error against ground truth. Mark off-camera players as missing; never manufacture a complete pitch.

## V6 — Live analysis

Stream validated data through incremental analytics. Measure latency and data coverage. Use past-only motion features: the current centered smoothing is suitable for post-match analysis and must not be reused as a live feature. Validate without future frames.

Public hosting follows a local portfolio demo and a dataset redistribution review. No hosting, authentication, API costs, live video, or learned forecast is required for V1.

V1 portfolio artifacts: [featured example](demo/featured-analysis.md), [four screenshots and capture notes](demo/README.md), and a [80-second edited browser walkthrough](demo/walkthrough.mp4). The walkthrough packages 240 sequential real browser captures at 3 fps; it is edited low-frame-rate browser footage, not match video or seamless high-frame-rate screen recording. Reproducible Windows setup is in the [main README](../README.md).
