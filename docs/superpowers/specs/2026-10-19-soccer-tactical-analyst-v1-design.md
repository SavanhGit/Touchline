# Soccer Tactical Analyst V1 Design

## Status and origin

This file records the scope already agreed in the conversation under **Soccer Tactical Analyst: First Release and Roadmap**, plus the concrete behavior in the implementation that was started before work paused. It is a reviewable companion to the detailed implementation plan, not a claim that the release is finished. Changes to these decisions must be identified during review rather than silently introduced by an implementer.

The user wants a soccer portfolio project built around post-match analysis of whatever teams have usable data. The eventual question is why a team lost or is losing, and what adjustments might help. V1 addresses measurable patterns and hypotheses; it cannot establish causes or simulate the outcome of a tactical change.

## Product and scope

**Product name:** Touchline.

**Question:** “What patterns contributed to this result, and what adjustments might have helped?”

**Audience:** Recruiters and people exploring a personal soccer analytics project.

**V1 delivery:** A local interactive Streamlit dashboard, a source-checked losing-team example, screenshots, a short recorded walkthrough, and reproducible Windows instructions.

Start with Metrica sample games 1–2 and support either team. The featured example is game 1's away team, which lost 3–0. Game 2 finished Home 3–2 Away. Keep anonymized team/player identifiers; do not invent real names or a competition.

V1 excludes chat, model-provider selection, public hosting, live tracking, arbitrary match downloads, learned forecasting, video inference, and expanded tactical models. IDSSE is the next dataset work package and must not block V1.

## Architecture and contracts

Data flow: **source CSV → canonical tracking/modeling events + complete goals → MatchBundle → held-out analytics → MatchReport → dashboard and audit artifacts**.

Keep existing tuple-returning loaders compatible. The new `load_metrica_bundle(root, game)` adds complete match context without changing `load_metrica_game`.

`MatchBundle` owns the match ID, tracking, modeling events, complete goal records, source, sampling rate, score availability, anonymous team labels, and quality results. A known goalless match has score 0–0; an unknown score is `None`. Goal records contain team, period, frame, source seconds, player ID, penalty flag, and optional original subtype.

`analyze_match(match_bundle, team, training_matches) -> MatchReport` stays independent of Streamlit and the global agent registry. The report owns result context, both-team comparisons, channel tables, selected-team player metrics, pass values, findings, evidence, limitations, and xT provenance.

Each finding owns a category, title, measured observation, adjustment to investigate, tradeoff, numeric counts/denominators, and evidence. Evidence owns the match, period, frame, source time, team/player, event type, and event coordinates. Event IDs need to be deterministic for an unchanged ordered event table and disambiguate events sharing a frame; V1 does not promise identity across rewritten source files.

## Data correctness

- Canonical tracking is in metres, centered on a nominal 105 × 68 m pitch. Home attacks +x after orientation in every period; away attacks −x.
- Rotate tracking and event coordinates using the same original orientation decisions. Multiple matches must be oriented independently.
- Object identity is `(match_id, period, frame, team, player_id)`; temporal joins must include period and match.
- Compute kinematics from source time differences, grouping by match, period, team, and player. Invalid/nonpositive time differences cannot produce infinite speeds.
- Use the source sampling rate for tracked minutes. Default player eligibility is at least 15 tracked minutes; missing observations must not be counted as observed time.
- Keep complete scored goals, including penalties, separate from penalty-excluded modeling shots. Never infer the displayed score from filtered modeling events.
- Quality checks report duplicates, conflicting clocks, retained missing/nonfinite positions, nominal-bound violations, global frame gaps, event/tracking coverage, positional alignment, and shot orientation. Outside-pitch observations can be valid out-of-play locations: report them, do not silently delete them.
- Refuse ambiguous duplicate object keys or conflicting frame clocks before report calculations. Missing evidence tracking yields an explicit unavailable state; do not borrow a frame from another period.

Fit xT on other matches only. Reject the held-out match in the training set; record training IDs. Training without shots must produce a clear calibration error, not a fabricated surface. For Metrica, analyze game 1 with game 2's fitted surface, and vice versa.

## Analysis rules

Use three equal-width channels in each team's attacking frame: right for `y < -68/6`, centre for `-68/6 <= y < 68/6`, and left for `y >= 68/6`. Opponent danger labels explicitly refer to the opponent's attacking channels.

Show at most one finding in each category and at most three total. A finding needs a denominator of at least five relevant events and at least three supporting events. These are display safeguards, not significance tests.

| Category | Measurement and selection | Supporting evidence |
|---|---|---|
| Buildup | Pass starts at attacking `x0 < 0`; loss means unsuccessful pass. Compare channel loss rates with denominators; choose highest qualifying rate, then highest summed turnover cost. | Up to three losses with highest turnover cost. |
| Progression | An entry is a successful pass with `x0 < 17.5` and `x1 >= 17.5`. Denominator is all passes originating outside the final third in that channel. Choose lowest qualifying entry rate, then largest denominator. At least three nonentries must support the finding. | Up to three nonentries sampled across the match chronology. |
| Danger | Count opponent modeling shots by origin channel. Use completed opponent entries by origin channel only if no shot channel qualifies. Choose largest qualifying concentration. | Up to three shots, goals first; otherwise entries sampled across chronology. |

Use right, centre, left as the final deterministic tie order. Keep zero-denominator rates unavailable, not zero percent. Keep count, rate, pass value and turnover cost separate. Failed pass endpoints are interception/loss locations, not intended targets. Nonentries can be useful circulation; they do not prove bad decisions. Channel counts do not identify defensive responsibility.

Suggested adjustments are deterministic hypotheses: review closer support or another buildup route; review forward support and other progression routes; review earlier pressure or cover in the opponent's highlighted channel. Each explains the associated tradeoff. Do not generate formation changes, causal explanations, or predicted score improvements.

## Dashboard and evidence

Keep the existing four views:

1. **Overview:** complete result and goal timeline, selected-team summary, both-team comparisons, and findings.
2. **Risk & progression:** turnover endpoints, completed entry routes, and channel comparisons.
3. **Players:** pass impact, buildup turnover count/cost, minutes and nearest-opponent context; inferred goalkeepers separate by default.
4. **Evidence:** finding/moment selectors, event ID and source clock, pitch positions, fixed source-event arrow, movement trails, and a time slider.

Keep match/team/moment selection session-local. Cache source bundles without mutating them and cache derived reports using data/code revision keys. The app must not use the agent registry's global state.

For evidence, use the selected period only, a ±3.4-second source window, a slider from −3 to +3 seconds in 0.2-second steps, and up to two preceding seconds of trails. Return an unavailable frame when the nearest timestamp is farther than `max(1.5/fps, 0.06)` seconds. Never connect trails across a missing segment longer than that tolerance. The event arrow remains fixed while player positions follow the slider.

Optional pitch control is calculated only for the displayed frame when both teams are present. Explain that it omits ball flight and acceleration and is not pass-success probability. Label the view **tracking reconstruction**, not match video.

No data, partial files, unknown scores, no qualifying findings, empty player tables, and unavailable evidence should have readable empty/error states. Show limitations beside the affected results. Match/team switching must not preserve an invalid previous evidence selection.

## Verification and delivery

Release acceptance requires all four real match/team reports, original source-event checks for every featured moment, no skipped real-data release gate, passing synthetic and dashboard checks, manual inspection of the four views, and no raw data committed.

Reproduce xT split-match agreement, pitch control, zone threat, and player-impact checks. Store the observed outputs and environment instead of treating earlier handover numbers as requirements to force tests to meet.

Deliver `docs/demo/featured-analysis.md`, `release-validation.json`, screenshots of all four views, a roughly 45–90-second browser walkthrough with captions, and a single documented launch command. A slideshow assembled from still captures must be labeled as a slideshow rather than a continuous screen recording. Attribute Metrica Sports in the app and artifacts. Keep the source datasets external to Git.

## Later milestones

V2 validates IDSSE/XML synchronization, event semantics and sampling rates before catalogue integration. V3 chooses model access and adds evidence-citing chat over explicit per-session tool context. V4 adds richer models only with suitable data and held-out baseline comparisons. V5 measures video tracking/calibration against reference data. V6 adds streaming latency checks and past-only features. Each needs its own reviewed spec and implementation plan; none is silently authorized by completion of V1.
