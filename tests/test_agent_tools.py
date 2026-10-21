import pandas as pd
import pytest

from soccer_tactics.agent.tools import AnalysisContext, call_tool, tool_specs
from soccer_tactics.analytics.report import Evidence, Finding, MatchReport


def _report(team="home", match_id="m1"):
    evidence = Evidence("m1:1:12:0", match_id, 1, 12, 12.0, team, "p1", "pass", 0., 0., 1., 1.)
    finding = Finding("progression", "Limited entries", "Observation", "Suggestion", "Tradeoff",
                      {"rate": .25}, [evidence])
    return MatchReport(match_id, team, {"home": 1, "away": 0}, "win",
                       pd.DataFrame(), pd.DataFrame([{"team": team, "passes": 4}]),
                       pd.DataFrame(), pd.DataFrame([{"match_id": match_id, "team": team,
                           "player_id": "p1", "impact_p90": .4}]),
                       pd.DataFrame([{"event_id": evidence.event_id, "match_id": match_id,
                           "period": 1, "frame": 12, "time_s": 12.0, "team": team,
                           "player_id": "p1", "event": "pass", "x0": 0., "y0": 0.,
                           "x1": 1., "y1": 1.}]),
                       [finding], {"source": "Metrica"}, ["single match is noisy"])


def _context(team="home", match_id="m1"):
    report = _report(team, match_id)
    return AnalysisContext(source="Metrica", match_id=match_id, selected_team=team, report=report,
                           team_names={"home": "Home FC", "away": "Away FC"},
                           limitations=("single match is noisy",),
                           allowed_citation_ids=("m1:1:12:0",))


def test_metrica_specs_and_tools_are_context_bound_and_cited():
    context = _context()
    names = {spec["name"] for spec in tool_specs(context)}
    assert names == {"match_summary", "team_findings", "player_metrics", "evidence_lookup", "sequence_summary", "defensive_summary", "shot_summary"}
    summary = call_tool(context, "match_summary", {})
    assert summary["match_id"] == "m1"
    assert summary["citation_ids"] == ["m1:1:12:0"]
    findings = call_tool(context, "team_findings", {})
    assert findings["findings"][0]["evidence_ids"] == ["m1:1:12:0"]
    players = call_tool(context, "player_metrics", {"metric": "impact_p90"})
    assert players["rows"][0]["player_id"] == "p1"
    assert players["citation_ids"] == ["m1:1:12:0"]
    with pytest.raises(ValueError, match="supported values"):
        call_tool(context, "player_metrics", {"metric": "match_id"})
    evidence = call_tool(context, "evidence_lookup", {"citation_id": "m1:1:12:0"})
    assert evidence["evidence"]["team"] == "home"


def test_invalid_and_cross_context_requests_are_rejected():
    context = _context()
    with pytest.raises(KeyError, match="unknown"):
        call_tool(context, "threat_map", {})
    with pytest.raises(ValueError, match="citation"):
        call_tool(context, "evidence_lookup", {"citation_id": "m1:1:12:999"})
    with pytest.raises(ValueError, match="team"):
        AnalysisContext(source="Metrica", match_id="m1", selected_team="away", report=_report(),
                       team_names={}, limitations=(), allowed_citation_ids=())
    with pytest.raises(ValueError, match="match"):
        AnalysisContext(source="Metrica", match_id="m2", selected_team="home", report=_report(),
                       team_names={}, limitations=(), allowed_citation_ids=())
    report = _report()
    report.findings[0].evidence.append(
        Evidence("m1:1:13:0", "m1", 1, 13, 13.0, "away", "a1", "pass", 0., 0., 1., 1.)
    )
    opponent_context = AnalysisContext(source="Metrica", match_id="m1", selected_team="home",
                                       report=report, team_names={}, limitations=(),
                                       allowed_citation_ids=("m1:1:13:0",))
    assert call_tool(opponent_context, "evidence_lookup", {
        "citation_id": "m1:1:13:0",
    })["evidence"]["team"] == "away"


def test_pass_and_player_metrics_remain_citeable_without_findings():
    report = _report()
    report.findings.clear()
    context = AnalysisContext(source="Metrica", match_id="m1", selected_team="home", report=report,
                              team_names={}, limitations=(), allowed_citation_ids=("m1:1:12:0",))
    player_result = call_tool(context, "player_metrics", {"metric": "impact_p90"})
    assert player_result["rows"][0]["citation_ids"] == ["m1:1:12:0"]
    assert player_result["citation_ids"] == ["m1:1:12:0"]
    assert call_tool(context, "evidence_lookup", {"citation_id": "m1:1:12:0"})[
        "evidence"]["player_id"] == "p1"


def test_idsse_exposes_origin_facts_only_and_scopes_to_selected_team():
    origins = pd.DataFrame([
        {"match_id": "i1", "period": 1, "team": "home", "event": "pass", "success": True,
         "player_id": "p1", "source_code": "E1", "origin_id": "IDSSE:i1:000000", "source_timestamp": "2020-01-01T00:00:00Z",
         "gameclock_s": 12., "x_m": 2., "y_m": 3., "x1": 99., "attack_direction": "right",
         "tracking_time_s": 10., "xt": .2},
        {"match_id": "i1", "period": 1, "team": "away", "event": "shot", "success": False,
         "player_id": "p2", "source_code": "E2", "origin_id": "IDSSE:i1:000001", "source_timestamp": "2020-01-01T00:00:01Z",
         "gameclock_s": 13., "x_m": 4., "y_m": 5.},
    ])
    context = AnalysisContext(source="IDSSE", match_id="i1", selected_team="home", report=None,
                              event_origins=origins, team_names={"home": "Home", "away": "Away"},
                              limitations=(), allowed_citation_ids=("IDSSE:i1:000000",))
    assert [spec["name"] for spec in tool_specs(context)] == ["origin_lookup"]
    result = call_tool(context, "origin_lookup", {"event": "pass"})
    assert len(result["origins"]) == 1
    row = result["origins"][0]
    assert row["team"] == "home" and row["x_m"] == 2.
    assert row["citation_id"] == "IDSSE:i1:000000"
    forbidden = {"x1", "attack_direction", "tracking_time_s", "xt", "destination", "progression"}
    assert forbidden.isdisjoint(row)


def test_source_contract_rejects_mixed_match_rows_and_invalid_arguments():
    context = _context()
    with pytest.raises(ValueError, match="arguments"):
        call_tool(context, "match_summary", {"match_id": "other"})
    origins = pd.DataFrame([{"match_id": "other", "team": "home"}])
    with pytest.raises(ValueError, match="match"):
        AnalysisContext(source="IDSSE", match_id="i1", selected_team="home", report=None,
                       event_origins=origins, team_names={}, limitations=(),
                       allowed_citation_ids=())


def test_v4_summaries_are_context_bound_without_findings():
    from factories import make_bundle
    from soccer_tactics.analytics.report import analyze_match
    from soccer_tactics.dashboard.chat import build_metrica_context, citation_description
    bundle = make_bundle("one")
    bundle.events.loc[bundle.events.index[-2:], "team"] = "home"
    bundle.events.loc[bundle.events.index[-2], "event"] = "tackle"
    report = analyze_match(bundle, "home", [make_bundle("training")])
    report.findings.clear()
    context = build_metrica_context(report)
    for name, field in [("sequence_summary", "possessions"), ("defensive_summary", "defensive_actions"), ("shot_summary", "shot_summaries")]:
        result = call_tool(context, name, {})
        assert result["match_id"] == "one" and result["team"] == "home"
        assert all(row["team"] == "home" for row in result[field])
        assert result["citation_ids"]
        assert set(result["citation_ids"]) <= set(context.allowed_citation_ids)
        for cid in result["citation_ids"]:
            assert call_tool(context, "evidence_lookup", {"citation_id": cid})["evidence"]["match_id"] == "one"
            assert citation_description(context, cid)
    assert call_tool(context, "sequence_summary", {})["sample_counts"]["possessions"] == len(report.possessions[report.possessions.team == "home"])


def test_idsse_stays_origin_only_after_v4():
    origins = pd.DataFrame([{"match_id": "i1", "team": "home", "origin_id": "IDSSE:i1:000001"}])
    context = AnalysisContext("IDSSE", "i1", "home", event_origins=origins,
                              allowed_citation_ids=("IDSSE:i1:000001",))
    for name in ("sequence_summary", "defensive_summary", "shot_summary"):
        with pytest.raises(KeyError):
            call_tool(context, name, {})
