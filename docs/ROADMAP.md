# Touchline roadmap

The first release is **Touchline**, a local post-match dashboard. Its question is: “What patterns contributed to the result, and what adjustments might have helped?”

## V1 — Evidence before explanation

- Metrica sample games 1–2; both home and away perspectives.
- Complete goal context alongside penalty-excluded modeling events.
- Held-out xT, buildup turnover patterns, direct final-third entries, and opponent danger channels.
- Up to three deterministic findings, with source event references, hypotheses, and tradeoffs.
- Pitch plots, player summaries, and period-aware tracking reconstruction.
- Real-data regression tests, checked example report, screenshots, and a captioned walkthrough.

The safeguards of five denominator events and three supporting events are display rules, not confidence intervals. Reports can contain fewer than three findings. No finding establishes causation or promises a different result.

## V2 — More matches

Validate the existing IDSSE loader against actual XML. Check Floodlight compatibility, orientation, period clocks, coordinate units, event semantics, team sheets, and actual sample rates. Add bundle metadata and complete goals, then run leave-one-match-out evaluation with per-match output. Do not put IDSSE into the dashboard until these checks pass. Its download is not a blocker for V1.

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
