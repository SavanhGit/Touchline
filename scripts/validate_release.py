"""Validate all four real team reports and export source-checked artifacts."""
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
        & np.isclose(selected.x1, evidence.x1, rtol=0, atol=1e-8)
        & np.isclose(selected.y1, evidence.y1, rtol=0, atol=1e-8)
    ]
    _require(len(exact) == 1, "evidence fields do not identify exactly one canonical event")
    event = exact.iloc[0]
    event_index = int(exact.index[0])
    expected_event_id = f"{bundle.match_id}:{int(event.period)}:{int(event.frame)}:{event_index}"
    _require(evidence.event_id == expected_event_id, "evidence ID does not identify its canonical event row")

    source_type = "SHOT" if event.event == "shot" else ("PASS" if bool(event.success) else "BALL LOST")
    raw = raw_events.copy()
    subtype = raw.Subtype.fillna("")
    mask = ((raw.Period == evidence.period) & (raw["Start Frame"] == evidence.frame)
            & (raw.Team.str.lower() == evidence.team)
            & (raw.From.fillna("").str.replace(" ", "", regex=False) == evidence.player_id)
            & (raw.Type == source_type))
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
    _require(np.isfinite(converted).all(), "source event has unavailable coordinates")
    flips = _source_orientation_flips(raw_events)
    _require(int(evidence.period) in flips,
             f"cannot independently infer source orientation for evidence period {evidence.period}")
    if flips[int(evidence.period)]:
        converted = -converted
    _require(np.allclose(converted, canonical_xy, rtol=0, atol=1e-7),
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
             "xT is trained on game 2 only. Teams and player IDs are anonymous. Data: [Metrica Sports](https://github.com/metrica-sports/sample-data).", ""]
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
              f"The source audit compared all displayed event fields, including type, player, team, period, source seconds and coordinates, with the original event CSV. Event-to-player tracking coverage is {coverage_text}; median event alignment is {median_text} and p90 alignment is {p90_text}. See `release-validation.json` for all four team reports and measured quality results.", "",
              "Counts exclude carries. Shot-pattern findings exclude penalties, while the result includes every recorded goal. Turnover cost and player detriment are exploratory xT quantities. Pitch control is a simplified spatial model. Channel concentrations describe locations, not defensive responsibility.", ""]
    return "\n".join(lines)


def run(root: Path, output: Path) -> None:
    bundles = [load_metrica_bundle(root, game) for game in (1, 2)]
    results = []
    for bundle, training in zip(bundles, bundles[::-1]):
        game = int(bundle.match_id.rsplit("_", 1)[1])
        raw = pd.read_csv(root / "data" / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv")
        _verify_modeling_shots(bundle, raw)
        source_goals = raw[(raw.Type == "SHOT") & raw.Subtype.fillna("").str.endswith("GOAL")]
        expected_score = {team: int((source_goals.Team.str.lower() == team).sum()) for team in ("home", "away")}
        _require(bundle.score == expected_score, f"complete source score mismatch for {bundle.match_id}")
        source_penalty = (raw.Type.shift() == "SET PIECE") & (raw.Subtype.shift() == "PENALTY")
        expected_goals = source_goals[["Team", "Period", "Start Frame", "Start Time [s]", "From"]].copy()
        expected_goals.columns = ["team", "period", "frame", "time_s", "player_id"]
        expected_goals["team"] = expected_goals.team.str.lower()
        expected_goals["player_id"] = expected_goals.player_id.str.replace(" ", "", regex=False)
        expected_goals["penalty"] = source_penalty.loc[source_goals.index].to_numpy(bool)
        actual_goals = bundle.goals[["team", "period", "frame", "time_s", "player_id", "penalty"]].copy()
        goal_keys = ["team", "period", "frame", "time_s", "player_id", "penalty"]
        _require(len(actual_goals) == len(expected_goals) and
                 actual_goals.sort_values(goal_keys).reset_index(drop=True).equals(
                     expected_goals.sort_values(goal_keys).reset_index(drop=True)),
                 f"complete goal records or penalty flags differ from raw source for {bundle.match_id}")
        for key in ("duplicate_objects", "conflicting_clocks"):
            _require(bundle.quality[key] == 0, f"{bundle.match_id} quality check {key} failed")
        _require(bundle.quality["alignment_p90_m"] < 2, f"{bundle.match_id} p90 alignment exceeds 2 m")
        _require(bundle.quality["event_tracking_coverage"] is not None and bundle.quality["event_tracking_coverage"] > .99,
                 f"{bundle.match_id} event-to-player tracking coverage is below 99%")
        for team in ("home", "away"):
            report = analyze_match(bundle, team, [training])
            _require(report.score == expected_score, f"{bundle.match_id} report score differs from complete goals")
            for finding in report.findings:
                _require(finding.metrics["denominator"] >= 5 and finding.metrics["numerator"] >= 3,
                         f"{bundle.match_id}/{team} finding lacks minimum evidence support")
                for evidence in finding.evidence:
                    verify_evidence(bundle, evidence, raw)
            results.append({"match_id": report.match_id, "team": team, "score": report.score,
                            "outcome": report.outcome, "training_match_ids": [training.match_id],
                            "quality": bundle.quality, "comparisons": report.comparisons.to_dict("records"),
                            "findings": [asdict(f) for f in report.findings]})
    output.mkdir(parents=True, exist_ok=True)
    clean_results = _json_safe(results)
    (output / "release-validation.json").write_text(json.dumps(clean_results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    featured = next(r for r in clean_results if r["match_id"] == "metrica_1" and r["team"] == "away")
    (output / "featured-analysis.md").write_text(_featured_markdown(featured), encoding="utf-8")
    print(f"Validated {len(results)} real team reports; exported {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/raw/metrica"))
    parser.add_argument("--output", type=Path, default=Path("docs/demo"))
    args = parser.parse_args()
    run(args.root, args.output)
