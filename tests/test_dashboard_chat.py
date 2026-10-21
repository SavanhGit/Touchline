import pandas as pd

from factories import make_bundle
from soccer_tactics.agent.chat import ChatAnswer
from soccer_tactics.analytics.report import analyze_match
from soccer_tactics.dashboard.chat import (
    build_idsse_context,
    build_metrica_context,
    citation_description,
    ensure_chat_state,
    render_chat_panel,
)


def test_metrica_context_uses_selected_report_passes_only():
    bundle = make_bundle("1")
    report = analyze_match(bundle, "home", [make_bundle("training")])

    context = build_metrica_context(report, {"home": "Home", "away": "Away"})

    assert context.match_id == "1"
    assert context.selected_team == "home"
    assert context.report is report
    expected = report.passes.loc[
        (report.passes.match_id.astype(str) == "1") & (report.passes.team == "home"), "event_id"
    ].astype(str).tolist()
    expected.extend(
        str(evidence.event_id) for finding in report.findings for evidence in finding.evidence
        if str(evidence.match_id) == str(report.match_id)
    )
    assert context.allowed_citation_ids == tuple(dict.fromkeys(expected))


def test_idsse_context_canonicalizes_loaded_rows_and_limits_citations_to_selected_side():
    origins = pd.DataFrame([
        {"origin_id": "IDSSE:DFL-MAT-A1:000001", "match_id": "DFL-MAT-A1",
         "period": 1, "team": "Home Label", "event": "pass", "source_code": "Play_Pass",
         "source_timestamp": "2026-10-21T12:00:00Z", "gameclock_s": 4.0,
         "source_x_m": 50.0, "source_y_m": 20.0},
        {"origin_id": "IDSSE:DFL-MAT-A1:000002", "match_id": "DFL-MAT-A1",
         "period": 1, "team": "Away Label", "event": "shot", "source_code": "ShotAtGoal_SavedShot",
         "source_timestamp": "2026-10-21T12:01:00Z", "gameclock_s": 64.0,
         "source_x_m": 30.0, "source_y_m": 10.0},
    ])

    context = build_idsse_context(
        origins, "DFL-MAT-A1", "home", {"home": "Home Label", "away": "Away Label"}
    )

    assert context.event_origins.team.tolist() == ["home", "away"]
    assert context.event_origins.match_id.eq("DFL-MAT-A1").all()
    assert context.allowed_citation_ids == ("IDSSE:DFL-MAT-A1:000001",)
    assert "tracking synchronization" in " ".join(context.limitations).lower()
    assert "IDSSE:DFL-MAT-A1:000001" in citation_description(context, "IDSSE:DFL-MAT-A1:000001")
    assert "source 50.0, 20.0 m" in citation_description(context, "IDSSE:DFL-MAT-A1:000001")


def test_chat_state_resets_for_each_selection_tuple_change():
    state = {"_touchline_chat_selection": ("Metrica", "1", "home"),
             "_touchline_chat_history": [{"role": "user", "content": "old"}]}

    assert ensure_chat_state(state, ("Metrica", "1", "home")) == state["_touchline_chat_history"]
    assert ensure_chat_state(state, ("Metrica", "2", "home")) == []
    state["_touchline_chat_history"].append({"role": "user", "content": "new match"})
    assert ensure_chat_state(state, ("Metrica", "2", "away")) == []
    state["_touchline_chat_history"].append({"role": "user", "content": "new team"})
    assert ensure_chat_state(state, ("IDSSE", "DFL-MAT-A1", "away")) == []


def test_metrica_context_includes_opponent_evidence_from_selected_report_finding():
    report = analyze_match(make_bundle("1"), "home", [make_bundle("training")])
    evidence_type = type(report.findings[0].evidence[0])
    report.findings[0].evidence.append(evidence_type(
        "opponent-shot", "1", 1, 90, 90., "away", "keeper", "shot", 1., 2., 3., 4.
    ))
    context = build_metrica_context(report)

    assert "opponent-shot" in context.allowed_citation_ids
    assert "keeper" in citation_description(context, "opponent-shot")


class _ContextManager:
    def __enter__(self): return self
    def __exit__(self, *args): return False


class _FakeStreamlit:
    def __init__(self):
        self.session_state = {}

    def container(self, **kwargs): return _ContextManager()
    def subheader(self, value): pass
    def caption(self, value): pass
    def text_input(self, label, key): return "What happened?"
    def button(self, *args, **kwargs): return True
    def markdown(self, value): pass
    def write(self, value): pass
    def error(self, value): pass


def test_dashboard_withholds_answer_without_resolvable_citation():
    report = analyze_match(make_bundle("1"), "home", [make_bundle("training")])
    context = build_metrica_context(report)
    st = _FakeStreamlit()

    render_chat_panel(st, context, ("Metrica", "1", "home"),
                      answerer=lambda *_: ChatAnswer("answered", "Unsupported claim.", ["bad-id"]))

    response = st.session_state["_touchline_chat_history"][-1]
    assert response["status"] == "withheld"
    assert not response["citation_ids"]
    assert "can't verify" in response["content"]

