"""Validate all six real team reports and export source-checked artifacts."""
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_tactics.analytics.report import Evidence, analyze_match
from soccer_tactics.loaders import load_metrica_bundle
from soccer_tactics.match import MatchBundle

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
_NOT_A_PASS = "THEFT|FORCED|HAND BALL|OFFSIDE|REFEREE|WOODWORK|END HALF"


def _close(left, right):
    try:
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-8))
    except (TypeError, ValueError):
        return left == right


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_orientation_flips(raw_events: pd.DataFrame) -> dict:
    """Infer each period's source orientation from the two teams' shot locations."""
    shots = raw_events[(raw_events.Type == "SHOT") & raw_events["Start X"].notna()]
    means = shots.assign(team=shots.Team.str.lower()).groupby(["Period", "team"])["Start X"].mean().unstack()
    flips = {}
    for period, row in means.iterrows():
        _require(pd.notna(row.get("home")) and pd.notna(row.get("away")),
                 f"cannot independently infer source orientation for period {period}")
        # Home's source-frame attack direction is toward the side with the more advanced shots.
        flips[int(period)] = bool(row["home"] < row["away"])
    return flips


def _verify_modeling_shots(bundle: MatchBundle, raw_events: pd.DataFrame) -> None:
    """Require canonical shots to equal every raw shot outside a penalty set piece."""
    after_penalty = (raw_events.Type.shift() == "SET PIECE") & (raw_events.Subtype.shift() == "PENALTY")
    source = raw_events[(raw_events.Type == "SHOT") & ~after_penalty].copy()
    canonical = bundle.events[bundle.events.event == "shot"]

    def source_identity(row):
        return (int(row.Period), int(row["Start Frame"]), round(float(row["Start Time [s]"]), 8),
                str(row.Team).lower(), str(row.From).replace(" ", ""),
                str(row.Subtype or "").endswith("GOAL"))

    def canonical_identity(row):
        return (int(row.period), int(row.frame), round(float(row.time_s), 8),
                str(row.team), str(row.player_id), bool(row.success))

    expected = sorted(source_identity(row) for _, row in source.iterrows())
    actual = sorted(canonical_identity(row) for row in canonical.itertuples(index=False))
    _require(actual == expected,
             f"canonical modeling shots differ from penalty-excluded raw source for {bundle.match_id}")


def verify_evidence(bundle: MatchBundle, evidence: Evidence, raw_events: pd.DataFrame) -> None:
    """Check every displayed field against canonical data and its original Metrica row.

    Metrica coordinates are normalized, top-left-origin values. The source orientation for
    each period is independently inferred from raw shot locations, then applied after unit
    conversion before comparing all coordinates.
    """
    _require(evidence.match_id == bundle.match_id, "evidence match does not match bundle")
    defensive = evidence.event in {"tackle", "interception", "recovery"}
    canonical = bundle.events.reset_index(drop=True)
    selected = canonical[
        (canonical.match_id.astype(str) == str(evidence.match_id))
        & (canonical.period == evidence.period) & (canonical.frame == evidence.frame)
        & (canonical.team == evidence.team) & (canonical.player_id == evidence.player_id)
    ]
    _require(not selected.empty, "evidence has no canonical event at its match/frame/player")
    exact = selected[
        (selected.event == evidence.event)
        & np.isclose(selected.time_s, evidence.time_s, rtol=0, atol=1e-8)
        & np.isclose(selected.x0, evidence.x0, rtol=0, atol=1e-8)
        & np.isclose(selected.y0, evidence.y0, rtol=0, atol=1e-8)
        & np.isclose(selected.x1, evidence.x1, rtol=0, atol=1e-8, equal_nan=defensive)
        & np.isclose(selected.y1, evidence.y1, rtol=0, atol=1e-8, equal_nan=defensive)
    ]
    _require(len(exact) == 1, "evidence fields do not identify exactly one canonical event")
    event = exact.iloc[0]
    event_index = int(exact.index[0])
    expected_event_id = f"{bundle.match_id}:{int(event.period)}:{int(event.frame)}:{event_index}"
    _require(evidence.event_id == expected_event_id, "evidence ID does not identify its canonical event row")

    source_type = "SHOT" if event.event == "shot" else ("PASS" if bool(event.success) else "BALL LOST")
    expected_subtype = None
    if defensive:
        source_type, expected_subtype = {
            ("tackle", True): ("CHALLENGE", "TACKLE-WON"),
            ("tackle", False): ("CHALLENGE", "TACKLE-LOST"),
            ("interception", True): ("RECOVERY", "INTERCEPTION"),
            ("recovery", True): ("RECOVERY", "THEFT"),
        }.get((event.event, bool(event.success)), (None, None))
        _require(source_type is not None, "unsupported defensive action outcome")
    raw = raw_events.copy()
    subtype = raw.Subtype.fillna("")
    mask = ((raw.Period == evidence.period) & (raw["Start Frame"] == evidence.frame)
            & (raw.Team.str.lower() == evidence.team)
            & (raw.From.fillna("").str.replace(" ", "", regex=False) == evidence.player_id)
            & (raw.Type == source_type))
    if expected_subtype is not None:
        mask &= subtype == expected_subtype
    if source_type == "BALL LOST":
        mask &= raw["End X"].notna() & ~subtype.str.contains(_NOT_A_PASS, regex=True)
    if source_type == "SHOT":
        after_penalty = (raw.Type.shift(1) == "SET PIECE") & (subtype.shift(1) == "PENALTY")
        mask &= ~after_penalty
    source_rows = raw.loc[mask]
    _require(len(source_rows) == 1, f"expected one original {source_type} source row, found {len(source_rows)}")
    source = source_rows.iloc[0]
    _require(_close(source["Start Time [s]"], event.time_s), "source timestamp differs from canonical event")
    if source_type == "SHOT":
        source_goal = str(source.Subtype or "").endswith("GOAL")
        _require(bool(event.success) == source_goal, "shot goal status differs from original source subtype")

    converted = np.array([
        (float(source["Start X"]) - .5) * PITCH_LENGTH,
        (.5 - float(source["Start Y"])) * PITCH_WIDTH,
        (float(source["End X"]) - .5) * PITCH_LENGTH,
        (.5 - float(source["End Y"])) * PITCH_WIDTH,
    ])
    canonical_xy = np.array([event.x0, event.y0, event.x1, event.y1], dtype=float)
    _require(np.isfinite(converted[:2]).all(), "source event has unavailable origin coordinates")
    if defensive:
        _require((np.isfinite(converted[2:]) | np.isnan(converted[2:])).all(),
                 "source defensive action has invalid endpoints")
    else:
        _require(np.isfinite(converted[2:]).all(), "source event has unavailable coordinates")
    flips = _source_orientation_flips(raw_events)
    _require(int(evidence.period) in flips,
             f"cannot independently infer source orientation for evidence period {evidence.period}")
    if flips[int(evidence.period)]:
        converted = -converted
    _require(np.allclose(converted, canonical_xy, rtol=0, atol=1e-7, equal_nan=defensive),
             "source coordinates do not match canonical coordinates after normalized conversion/orientation")
    _require(str(source.Team).lower() == evidence.team, "source team differs from displayed evidence")
    _require(str(source.From).replace(" ", "") == evidence.player_id, "source player differs from displayed evidence")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _featured_markdown(featured: dict) -> str:
    score = featured.get("score") or {}
    outcome = featured.get("outcome")
    result_text = {"win": "The away team won.", "draw": "The match was a draw.",
                   "loss": "The away team lost."}.get(outcome)
    if result_text is None:
        away, home = score.get("away"), score.get("home")
        result_text = ("The result is unavailable." if away is None or home is None else
                       "The away team won." if away > home else
                       "The away team lost." if away < home else "The match was a draw.")
    coverage = featured["quality"].get("event_tracking_coverage")
    median = featured["quality"].get("alignment_median_m")
    p90 = featured["quality"].get("alignment_p90_m")
    coverage_text = "unavailable" if coverage is None else f"{coverage:.1%}"
    median_text = "unavailable" if median is None else f"{median:.3f} m"
    p90_text = "unavailable" if p90 is None else f"{p90:.3f} m"
    home_score, away_score = score.get("home"), score.get("away")
    score_text = "Score unavailable." if home_score is None or away_score is None else f"Home {home_score}–{away_score} Away."
    lines = ["# Featured analysis: Metrica game 1, away team", "",
             f"{score_text} {result_text}", "",
             "These patterns may have contributed; this report does not establish causation or promise that a suggested change would alter the result.", "",
             "xT is trained on games 2 and 3 only. Teams and player IDs are anonymous. Data: [Metrica Sports](https://github.com/metrica-sports/sample-data).", ""]
    for finding in featured["findings"]:
        lines += [f"## {finding['title']}", "", finding["observation"], "",
                  f"**Adjustment to investigate:** {finding['suggestion']}", "",
                  f"**Tradeoff:** {finding['tradeoff']}", "",
                  "| Evidence ID | Period | Frame | Source seconds | Player |",
                  "|---|---:|---:|---:|---|"]
        for e in finding["evidence"]:
            lines.append(f"| {e['event_id']} | {e['period']} | {e['frame']} | {e['time_s']:.2f} | {e['player_id']} |")
        lines.append("")
    lines += ["## Verification and interpretation", "",
              f"The source audit compared all displayed event fields, including type, player, team, period, source seconds and coordinates, with the original event source. Event-to-player tracking coverage is {coverage_text}; median event alignment is {median_text} and p90 alignment is {p90_text}. See `release-validation.json` for all six team reports and measured quality results.", "",
              "Counts exclude carries. Shot-pattern findings exclude penalties, while the result includes every recorded goal. Turnover cost and player detriment are exploratory xT quantities. Pitch control is a simplified spatial model. Channel concentrations describe locations, not defensive responsibility.", ""]
    return "\n".join(lines)


def source_events(root: Path, game: int, bundle: MatchBundle) -> pd.DataFrame:
    """Read original local CSV/JSON directly, independently of Kloppy filtering."""
    folder = root / "data" / f"Sample_Game_{game}"
    if game != 3:
        return pd.read_csv(folder / f"Sample_Game_{game}_RawEventsData.csv")
    with (folder / "Sample_Game_3_events.json").open(encoding="utf-8") as stream:
        events = json.load(stream)["data"]
    rows = []
    teams = bundle.metadata["source_team_ids"]
    for event in events:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append(dict(Team=teams[event["team"]["id"]], Type=event["type"]["name"],
            Subtype="-".join(item["name"] for item in subtypes), Period=event["period"],
            **{"Start Frame": event["start"]["frame"], "Start Time [s]": event["start"]["time"],
               "From": event["from"]["id"], "Start X": event["start"]["x"],
               "Start Y": event["start"]["y"], "End X": event["end"]["x"], "End Y": event["end"]["y"]}))
    return pd.DataFrame(rows)


def verify_sequences(report):
    """Audit membership, counts, period/team integrity and clipped transition clocks."""
    events = {e.event_id: e for e in report.event_evidence}
    for label, table in (("possession", report.possessions), ("transition", report.transitions)):
        for row in table.itertuples():
            _require(bool(row.event_ids), f"empty {label}")
            _require(all(eid in events for eid in row.event_ids), f"unknown {label} event")
            selected = [events[eid] for eid in row.event_ids]
            _require(all(e.period == row.period and e.match_id == row.match_id and e.team == row.team
                         for e in selected), f"{label} period/match/team boundary crossed")
            _require(_close(selected[0].time_s, row.start_time_s), f"{label} start clock mismatch")
            _require(all(row.start_time_s <= e.time_s <= row.end_time_s for e in selected),
                     f"{label} event outside clock window")
            _require(int(row.shots) == sum(e.event == "shot" for e in selected), f"{label} shot count mismatch")
            if label == "transition":
                _require(0 <= row.end_time_s-row.start_time_s <= 10 and row.window_s == 10,
                         "transition exceeds 10-second window")
                source = events.get(row.source_turnover_event_id)
                _require(source is not None and source.period == row.period and source.match_id == row.match_id,
                         "transition source period/match mismatch")
                _require(0 <= row.start_time_s-source.time_s <= 10, "transition trigger outside window")


def audit_suggestions(report, source_events=None):
    """Check each observation's support and citations, independently of ranking.

    Tactical benefit is deliberately not scored: source moments cannot validate a
    counterfactual adjustment. A visible tradeoff is a separate required field.
    """
    from soccer_tactics.analytics.report import channel
    from soccer_tactics.analytics.xt import attacking
    from soccer_tactics.analytics.sequences import DEFENSE
    if not report.findings:
        return []
    by_id = {e.event_id: e for e in report.event_evidence}
    events = pd.DataFrame([asdict(e) for e in report.event_evidence])
    events["channel"] = channel(attacking(events).y0)
    audits = []
    for finding in report.findings:
        _require(bool(finding.suggestion.strip()) and bool(finding.tradeoff.strip()), "suggestion needs visible tradeoff")
        own = report.passes[report.passes.team == report.team]
        category = finding.category
        if category == "transition":
            population = report.transitions[report.transitions.team == report.team]
            support = population[(population.final_third_entries == 0) & (population.shots == 0)]
            support_ids = {ids[0] for ids in support.event_ids}
            numerator, denominator = len(support), len(population)
        elif category == "defense":
            _require(source_events is not None, "defense suggestion audit requires canonical source outcomes")
            canonical = source_events.reset_index(drop=True).copy()
            canonical["event_id"] = [f"{report.match_id}:{int(row.period)}:{int(row.frame)}:{i}"
                                     for i, row in canonical.iterrows()]
            actions = canonical[(canonical.team == report.team) & canonical.event.isin(DEFENSE)]
            support = actions[~actions.success.astype(bool)]
            numerator, denominator = len(support), len(actions)
            support_ids = set(support.event_id)
        elif category in {"buildup", "progression"}:
            # Derive the channel from the complete title (the generated title names it).
            names = [name for name in ("left", "centre", "right") if finding.title.endswith(name)]
            _require(len(names) == 1, "finding channel unavailable")
            population = own[own.channel == names[0]]
            population = population[population.buildup] if category == "buildup" else population[population.outside_third]
            support = population[population.buildup_loss] if category == "buildup" else population[~population.entry]
            numerator, denominator = len(support), len(population)
            support_ids = set(support.event_id)
            if category == "progression":
                _require(finding.metrics["entries"] == int(population.entry.sum()), "progression entry count mismatch")
        elif category == "danger":
            opponent = "away" if report.team == "home" else "home"
            population = events[(events.team == opponent) & (events.event == "shot")] if finding.metrics["event_type"] == "shots" else report.passes[(report.passes.team == opponent) & report.passes.entry]
            names = [name for name in ("left", "centre", "right") if finding.title.endswith(name)]
            _require(len(names) == 1, "finding channel unavailable")
            support = population[population.channel == names[0]]
            numerator, denominator = len(support), len(population)
            support_ids = set(support.event_id)
        else:
            raise ValueError(f"unaudited suggestion category: {category}")
        _require(finding.metrics["numerator"] == numerator and finding.metrics["denominator"] == denominator,
                 "finding support count mismatch")
        _require(numerator >= 3 and denominator >= 5, "finding below display safeguard")
        _require(bool(finding.evidence) and all(e.event_id in support_ids for e in finding.evidence),
                 "citation does not support observation")
        for citation in finding.evidence:
            original = by_id.get(citation.event_id)
            _require(original is not None and all(
                _close(getattr(citation, key), getattr(original, key))
                or (pd.isna(getattr(citation, key)) and pd.isna(getattr(original, key)))
                for key in Evidence.__dataclass_fields__), "cited fields differ from audited source event")
        audits.append(dict(category=category, title=finding.title, numerator=numerator, denominator=denominator,
            citation_ids=[e.event_id for e in finding.evidence], descriptive_accuracy="pass",
            tradeoff=finding.tradeoff, tradeoff_visible=True,
            tactical_effect="Not evaluated: no causal or counterfactual validation"))
    return audits


def run(root: Path, output: Path) -> None:
    if __package__:
        from scripts.validate_cross_match import summarize_v4_fold
    else:
        from validate_cross_match import summarize_v4_fold
    bundles = [load_metrica_bundle(root, game) for game in (1, 2, 3)]
    results, reports = [], []
    for bundle in bundles:
        game = int(bundle.match_id.rsplit("_", 1)[1])
        raw = source_events(root, game, bundle)
        _verify_modeling_shots(bundle, raw)
        source_goals = raw[(raw.Type == "SHOT") & raw.Subtype.fillna("").str.endswith("GOAL")]
        expected_score = {team: int((source_goals.Team.str.lower() == team).sum()) for team in ("home", "away")}
        _require(bundle.score == expected_score, f"complete source score mismatch for {bundle.match_id}")
        if game == 3:
            _require(expected_score == bundle.metadata["source_score"] == {"home": 0, "away": 2}, "Game 3 metadata score mismatch")
        source_penalty = (raw.Type.shift() == "SET PIECE") & (raw.Subtype.shift() == "PENALTY")
        expected_goals = source_goals[["Team", "Period", "Start Frame", "Start Time [s]", "From"]].copy()
        expected_goals.columns = ["team", "period", "frame", "time_s", "player_id"]
        expected_goals["team"] = expected_goals.team.str.lower()
        expected_goals["player_id"] = expected_goals.player_id.str.replace(" ", "", regex=False)
        expected_goals["penalty"] = source_penalty.loc[source_goals.index].to_numpy(bool)
        keys = ["team", "period", "frame", "time_s", "player_id", "penalty"]
        _require(bundle.goals[keys].sort_values(keys).reset_index(drop=True).equals(
            expected_goals.sort_values(keys).reset_index(drop=True)), "source goal records mismatch")
        subtype = raw.Subtype.fillna("")
        expected_counts = dict(pass_count=int(((raw.Type == "PASS") | ((raw.Type == "BALL LOST") & raw["End X"].notna() & ~subtype.str.contains(_NOT_A_PASS))).sum()),
            shot=int(((raw.Type == "SHOT") & ~source_penalty).sum()),
            tackle=int(((raw.Type == "CHALLENGE") & subtype.isin(["TACKLE-WON", "TACKLE-LOST"])).sum()),
            interception=int(((raw.Type == "RECOVERY") & (subtype == "INTERCEPTION")).sum()),
            recovery=int(((raw.Type == "RECOVERY") & (subtype == "THEFT")).sum()))
        expected_counts["pass"] = expected_counts.pop("pass_count")
        actual_counts = {kind: int((bundle.events.event == kind).sum()) for kind in expected_counts}
        _require(actual_counts == expected_counts, f"source event totals mismatch: {actual_counts} != {expected_counts}")
        for key in ("duplicate_objects", "conflicting_clocks", "frame_gaps"):
            _require(bundle.quality[key] == 0, f"{bundle.match_id} quality check {key} failed")
        _require(bundle.quality["alignment_p90_m"] < 2, f"{bundle.match_id} p90 alignment exceeds 2 m")
        _require(bundle.quality["event_tracking_coverage"] is not None and bundle.quality["event_tracking_coverage"] > .99,
                 f"{bundle.match_id} event-to-player tracking coverage is below 99%")
        periods = []
        for period, clocks in bundle.tracking[["period", "frame", "time_s"]].drop_duplicates().groupby("period"):
            clocks = clocks.sort_values("frame")
            _require(np.allclose(clocks.time_s.diff().dropna(), .04, atol=1e-7, rtol=0), "source sample rate is not 25 Hz")
            selected = bundle.events[bundle.events.period == period]
            _require(selected.frame.between(clocks.frame.min(), clocks.frame.max()).all() and
                     selected.time_s.between(clocks.time_s.min(), clocks.time_s.max()).all(), "event outside period tracking bounds")
            periods.append(dict(period=int(period), first_frame=int(clocks.frame.min()), last_frame=int(clocks.frame.max()),
                first_time_s=float(clocks.time_s.min()), last_time_s=float(clocks.time_s.max()), fps=bundle.fps))
        _require({p["period"] for p in periods} == set(bundle.events.period) == {1, 2}, "period coverage mismatch")
        training = [other for other in bundles if other.match_id != bundle.match_id]
        for team in ("home", "away"):
            report = analyze_match(bundle, team, training)
            _require(report.provenance["training_match_ids"] == [other.match_id for other in training], "xT training provenance mismatch")
            verify_sequences(report)
            # Audit every retained source event, not only the three sampled citations.
            if team == "home":
                for evidence in report.event_evidence:
                    verify_evidence(bundle, evidence, raw)
            audit = audit_suggestions(report, bundle.events)
            reports.append(report)
            results.append(dict(match_id=report.match_id, team=team, score=report.score, outcome=report.outcome,
                training_match_ids=report.provenance["training_match_ids"], quality=bundle.quality,
                source_event_counts=actual_counts, periods=periods, source_orientation_flips=_source_orientation_flips(raw),
                comparisons=report.comparisons.to_dict("records"), findings=[asdict(f) for f in report.findings],
                suggestion_audit=audit))
    folds = [summarize_v4_fold(report, [(other.match_id, other.defensive_actions) for other in reports
              if other.team == "home" and other.match_id != report.match_id]) for report in reports]
    output.mkdir(parents=True, exist_ok=True)
    clean_results = _json_safe(results)
    (output / "release-validation.json").write_text(json.dumps(clean_results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "cross-match-validation.json").write_text(json.dumps(_json_safe(folds), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    featured = next(r for r in clean_results if r["match_id"] == "metrica_1" and r["team"] == "away")
    (output / "featured-analysis.md").write_text(_featured_markdown(featured), encoding="utf-8")
    print(f"Validated {len(results)} real team reports, {len(folds)} held-out team folds, "
          f"{sum(len(r['suggestion_audit']) for r in results)} suggestions; exported {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/raw/metrica"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/v4"))
    args = parser.parse_args()
    run(args.root, args.output)
