from pathlib import Path
from dataclasses import replace
import pickle

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from conftest import GAMES
from factories import make_bundle
from soccer_tactics.analytics.report import analyze_match
from soccer_tactics.loaders.catalog import CatalogEntry


@pytest.fixture(autouse=True)
def clear_dashboard_caches():
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def synthetic_app(tmp_path, *, sources=6, empty=False, unavailable=False, known_score=False,
                 answer_error=False):
    """Substitute source/analysis boundaries; render the real main() unchanged."""
    root = tmp_path / "sources"
    paths = [root / "data" / f"Sample_Game_{g}" / f"Sample_Game_{g}_Raw{suffix}.csv"
             for g in (1, 2) for suffix in ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team")]
    if sources >= 0:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        for path in paths[:sources]:
            path.write_text("source placeholder")
    if sources == 7:
        paths[-1].write_text("")
    bundles, reports = {}, {}
    for game in (1, 2):
        b = make_bundle(str(game))
        if empty or game == 2:
            b = replace(b, events=b.events.iloc[:0], tracking=b.tracking[b.tracking.time_s < 10])
        b = replace(b, score_known=known_score)
        bundles[game] = b
        for team in ("home", "away"):
            reports[game, team] = analyze_match(b, team, [make_bundle("training")])
            if empty:
                reports[game, team].comparisons.loc[0, "completion_rate"] = float("nan")
        if unavailable:
            bundles[game] = replace(b, tracking=b.tracking.iloc[:0])
    payload = tmp_path / "fixtures.pkl"
    payload.write_bytes(pickle.dumps((bundles, reports)))
    script = f'''
import sys, pickle
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import app as dashboard
import importlib
importlib.reload(dashboard)
import streamlit as st
dashboard.DATA = Path({str(root)!r})
dashboard.IDSSE_DATA = Path({str(tmp_path / "idsse")!r})
dashboard.IDSSE_MANIFEST = Path({str(tmp_path / "idsse-manifest.json")!r})
if "loader_calls" not in st.session_state:
    st.session_state["loader_calls"] = 0
bundles, reports = pickle.loads(Path({str(payload)!r}).read_bytes())
def bundle(game, revision):
    st.session_state["loader_calls"] = st.session_state.get("loader_calls", 0) + 1
    return bundles[game]
dashboard.load_bundle = bundle
dashboard.load_report = lambda game, team, revision: reports[game, team]
from soccer_tactics.agent.chat import ChatAnswer
raise_answer_error = {answer_error!r}
def fake_answer(context, question, history):
    if raise_answer_error:
        raise RuntimeError("Ollama is unavailable at localhost:11434")
    citations = list(context.allowed_citation_ids[:1])
    return ChatAnswer("answered", "fake answer", citations)
dashboard.answer_question = fake_answer
dashboard.main()
'''
    return AppTest.from_string(script, default_timeout=30)


@pytest.mark.parametrize("sources", [-1, 0])
def test_missing_sources_stop_before_modeling(tmp_path, sources):
    app = synthetic_app(tmp_path, sources=sources).run()
    assert not app.exception
    assert app.session_state["loader_calls"] == 0
    assert any("source" in item.value.lower() for item in app.sidebar.warning)
    assert not app.title


@pytest.mark.parametrize("broken", ["tracking", "events"])
def test_nonempty_malformed_sources_show_actionable_unavailable_state(tmp_path, broken):
    root = tmp_path / "sources"
    tracking = ",,,Home,,,\n,,,1,,,\nPeriod,Frame,Time [s],Player1,,Ball,\n1,1,0.04,0.2,0.3,0.4,0.5\n"
    for game in (1, 2):
        directory = root / "data" / f"Sample_Game_{game}"
        directory.mkdir(parents=True)
        for team in ("Home", "Away"):
            (directory / f"Sample_Game_{game}_RawTrackingData_{team}_Team.csv").write_text(tracking)
        (directory / f"Sample_Game_{game}_RawEventsData.csv").write_text("Team,Type\nHome,PASS\n")
    broken_name = ("Sample_Game_1_RawTrackingData_Home_Team.csv" if broken == "tracking"
                   else "Sample_Game_1_RawEventsData.csv")
    if broken == "tracking":
        (root / "data" / "Sample_Game_1" / broken_name).write_text(",,,Home,,\n,,,1,,\n")
    script = f'''
import sys, importlib
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import app as dashboard
# Earlier synthetic AppTests replace module loaders; restore the real ingestion boundary.
importlib.reload(dashboard)
dashboard.DATA = Path({str(root)!r})
dashboard.main()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert not app.title
    assert len(app.error) == 1
    message = app.error[0].value
    assert "Match analysis unavailable" in message
    assert broken_name in message
    assert "restore" in message.lower() or "download" in message.lower()


def test_empty_report_unknown_score_and_unavailable_rates(tmp_path):
    app = synthetic_app(tmp_path, empty=True).run()
    assert not app.exception
    assert any("unknown" in item.value.lower() for item in app.markdown)
    assert any("No pattern" in item.value for item in app.info)
    assert any("No eligible outfield" in item.value for item in app.info)
    assert any("No eligible goalkeeper" in item.value for item in app.info)
    comparison, channels = app.dataframe[0].value, app.dataframe[1].value
    assert comparison.loc["Completion rate", "home"] == "Unavailable"
    assert channels.entry_rate.eq("Unavailable").all()
    assert channels.buildup_loss_rate.eq("Unavailable").all()


def test_known_goalless_score_is_distinct_from_unknown(tmp_path):
    app = synthetic_app(tmp_path, empty=True, known_score=True).run()
    assert not app.exception
    assert any("Home 0 – 0 Away" in item.value for item in app.markdown)
    assert any("No goals recorded" in item.value for item in app.markdown)


def test_unavailable_evidence_preserves_source_context(tmp_path):
    app = synthetic_app(tmp_path, unavailable=True).run()
    app.selectbox(key="selected_team").select("home").run()
    assert not app.exception
    assert any("Tracking is unavailable" in item.value for item in app.warning)
    assert any("Evidence ID: 1:" in item.value for item in app.caption)
    app.slider(key="offset_1_home_0_0").set_value(-3.).run()
    app.checkbox(key="show_control").check().run()
    app.slider(key="offset_1_home_0_0").set_value(3.).run()
    assert not app.exception


def test_match_team_switch_does_not_reuse_evidence(tmp_path):
    app = synthetic_app(tmp_path).run()
    app.selectbox(key="selected_team").select("home").run()
    assert len(app.selectbox(key="finding_1_home").options) == 3
    app.selectbox(key="finding_1_home").select(2).run()
    assert any("Evidence ID: 1:" in item.value for item in app.caption)
    app.selectbox(key="selected_match").select("metrica_2").run()
    assert not app.exception
    assert any("No qualifying evidence" in item.value for item in app.info)
    assert not any("Evidence ID:" in item.value for item in app.caption)
    app.selectbox(key="selected_match").select("metrica_1").run()
    app.selectbox(key="selected_team").select("away").run()
    assert not app.exception
    assert not any("Evidence ID:" in item.value for item in app.caption)
    app.selectbox(key="selected_team").select("home").run()
    assert not app.exception
    assert any("Evidence ID: 1:" in item.value for item in app.caption)


def test_chat_citations_render_and_history_resets_when_match_or_team_changes(tmp_path):
    app = synthetic_app(tmp_path).run()
    app.selectbox(key="selected_team").select("home").run()
    app.text_input(key="touchline_chat_question").set_value("What shaped the match?").run()
    app.button(key="touchline_chat_submit").click().run()
    assert not app.exception
    assert any("Touchline · answered" in item.value for item in app.markdown)
    assert any("source" in item.value and "frame" in item.value for item in app.caption)
    assert len(app.session_state["_touchline_chat_history"]) == 2

    app.selectbox(key="selected_match").select("metrica_2").run()
    assert app.session_state["_touchline_chat_history"] == []
    app.selectbox(key="selected_team").select("home").run()
    assert app.session_state["_touchline_chat_history"] == []


def test_local_chat_failure_keeps_dashboard_available(tmp_path):
    app = synthetic_app(tmp_path, answer_error=True).run()
    app.text_input(key="touchline_chat_question").set_value("Question while Ollama is offline").run()
    app.button(key="touchline_chat_submit").click().run()

    assert not app.exception
    assert app.title[0].value == "What shaped the result?"
    assert any("ollama serve" in item.value.lower() for item in app.caption)
    assert any("Ollama is unavailable" in item.value for item in app.error)


def test_idsse_selection_renders_source_origins_and_limitations(tmp_path):
    script = f'''
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import app as dashboard
from soccer_tactics.loaders.catalog import CatalogEntry
dashboard.DATA = Path({str(tmp_path / "metrica")!r})
dashboard.IDSSE_DATA = Path({str(tmp_path / "idsse")!r})
dashboard.IDSSE_MANIFEST = Path({str(tmp_path / "manifest.json")!r})
entry = CatalogEntry("IDSSE", "idsse_DFL-MAT-A1", str(dashboard.IDSSE_DATA), "IDSSE · DFL-MAT-A1", {{"home":"Home Label", "away":"Away Label"}}, True)
dashboard.load_catalogue = lambda revision: [entry]
origins = pd.DataFrame([{{"origin_id":"IDSSE:DFL-MAT-A1:000001","match_id":"DFL-MAT-A1","period":1,"team":"Away Label","event":"pass","source_code":"Play_Pass","source_timestamp":"2026-01-01T00:00:00Z","gameclock_s":2.0,"source_x_m":50.0,"source_y_m":20.0}}])
dashboard.load_idsse_origin_report = lambda *args: (origins, {{"home":1,"away":0}}, entry.team_names, {{"event_capabilities":{{"pass_endpoints":False}}}}, {{"event_origin_count":1}})
dashboard.main()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "What the source records show"
    assert any("IDSSE/DFL source events" in item.value for item in app.caption)
    assert any("synchronization are unavailable" in item.value for item in app.warning)
    assert len(app.dataframe) == 1
    assert not any("Risk & progression" in item.label for item in app.tabs)
    assert not any("Seconds relative" in item.value for item in app.slider)


def test_provider_switch_resets_team_and_idsse_skips_analysis_and_tracking(tmp_path):
    bundle = make_bundle("1")
    report = analyze_match(bundle, "home", [make_bundle("2")])
    payload = tmp_path / "fixtures.pkl"
    payload.write_bytes(pickle.dumps((bundle, report)))
    script = f'''
import sys, pickle
from pathlib import Path
import pandas as pd
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import app as dashboard
from soccer_tactics.loaders.catalog import CatalogEntry
dashboard.DATA = Path({str(tmp_path / "metrica")!r})
dashboard.IDSSE_DATA = Path({str(tmp_path / "idsse")!r})
dashboard.IDSSE_MANIFEST = Path({str(tmp_path / "manifest.json")!r})
entries = [
    CatalogEntry("Metrica", "metrica_1", str(dashboard.DATA), "Metrica · Sample game 1", {{"home":"Home (anonymous)","away":"Away (anonymous)"}}, True),
    CatalogEntry("IDSSE", "idsse_DFL-MAT-A1", str(dashboard.IDSSE_DATA), "IDSSE · DFL-MAT-A1", {{"home":"Home Label","away":"Away Label"}}, True),
]
dashboard.load_catalogue = lambda revision: entries
bundle, report = pickle.loads(Path({str(payload)!r}).read_bytes())
dashboard.load_bundle = lambda game, revision: bundle
dashboard.load_report = lambda game, team, revision: report
origins = pd.DataFrame([{{"origin_id":"IDSSE:DFL-MAT-A1:000001","match_id":"DFL-MAT-A1","period":1,"team":"Away Label","event":"pass","source_code":"Play_Pass","source_timestamp":"2026-01-01T00:00:00Z","gameclock_s":2.0,"source_x_m":50.0,"source_y_m":20.0}}])
dashboard.load_idsse_origin_report = lambda *args: (origins, {{"home":1,"away":0}}, entries[1].team_names, {{}}, {{}})
from soccer_tactics.agent.chat import ChatAnswer
def fake_answer(context, question, history):
    citations = list(context.allowed_citation_ids[:1])
    return ChatAnswer("answered", "fake answer", citations)
dashboard.answer_question = fake_answer
dashboard.analyze_match = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("IDSSE called analyze_match"))
def forbidden_during_idsse(name, original):
    def check(*args, **kwargs):
        if str(dashboard.st.session_state.get("selected_match", "")).startswith("idsse_"):
            raise AssertionError("IDSSE invoked " + name)
        return original(*args, **kwargs)
    return check
dashboard.tracking_window = forbidden_during_idsse("tracking_window", dashboard.tracking_window)
dashboard.snapshot = forbidden_during_idsse("snapshot", dashboard.snapshot)
dashboard.evidence_pitch = forbidden_during_idsse("evidence_pitch", dashboard.evidence_pitch)
dashboard.main()
'''
    app = AppTest.from_string(script, default_timeout=45).run()
    assert not app.exception
    app.selectbox(key="selected_team").select("home").run()
    assert not app.exception and app.selectbox(key="selected_team").value == "home"
    app.text_input(key="touchline_chat_question").set_value("List a source event.").run()
    app.button(key="touchline_chat_submit").click().run()
    assert len(app.session_state["_touchline_chat_history"]) == 2
    app.selectbox(key="selected_match").select("idsse_DFL-MAT-A1").run()
    assert not app.exception
    assert app.session_state["_touchline_chat_history"] == []
    assert app.selectbox(key="selected_team").value == "away"
    assert app.title[0].value == "What the source records show"
    assert any("tracking synchronization" in item.value.lower() for item in app.caption)
    assert not any("Evidence ID:" in item.value for item in app.caption)
    assert not app.slider and not app.checkbox
    app.selectbox(key="selected_team").select("home").run()
    app.selectbox(key="selected_match").select("metrica_1").run()
    assert not app.exception
    assert app.selectbox(key="selected_team").value == "away"
    assert app.title[0].value == "What shaped the result?"
    app.selectbox(key="selected_team").select("home").run()
    assert not app.exception
    assert any(f"Evidence ID: {report.findings[0].evidence[0].event_id}" in item.value for item in app.caption)


@pytest.mark.skipif(len(GAMES) < 2, reason="requires both real Metrica games")
def test_dashboard_match_team_and_evidence_navigation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
    assert not app.exception
    assert app.title[0].value == "What shaped the result?"
    app.selectbox(key="selected_team").select("away").run()
    assert not app.exception
    app.selectbox(key="finding_1_away").select(1).run()
    app.selectbox(key="moment_1_away_1").select(1).run()
    app.slider(key="offset_1_away_1_1").set_value(1.).run()
    assert not app.exception
    app.checkbox(key="show_control").check().run()
    assert not app.exception
    app.selectbox(key="selected_match").select("metrica_2").run()
    assert not app.exception


def test_dashboard_v4_sections_render_for_metrica(tmp_path):
    app = synthetic_app(tmp_path).run()
    assert not app.exception
    headings = {item.value for item in app.subheader}
    assert {"Possessions and transitions", "Defensive actions", "Shot locations and outcomes"} <= headings
    assert any("10 seconds" in item.value for item in app.caption)
    assert any("observed tracking minutes" in item.value for item in app.caption)
    assert any("shot quality" in item.value for item in app.caption)
    assert any("5 shots" in item.value for item in app.caption)


def test_dashboard_v4_empty_summaries_are_explained(tmp_path):
    app = synthetic_app(tmp_path, empty=True).run()
    assert not app.exception
    text = " ".join(item.value for item in app.info)
    for expected in ("No observed possessions", "No qualifying transitions", "No recognized defensive actions", "No shots"):
        assert expected in text



@pytest.mark.parametrize("sources", [5, 7])
def test_single_complete_catalogued_match_is_selectable(tmp_path, sources):
    app = synthetic_app(tmp_path, sources=sources).run()
    assert not app.exception
    assert app.title[0].value == "What shaped the result?"
    assert app.selectbox(key="selected_match").value == "metrica_1"
    assert len(app.selectbox(key="selected_match").options) == 1


def test_three_match_dashboard_cache_retains_training_sources(monkeypatch):
    import importlib
    import app as dashboard
    importlib.reload(dashboard)
    calls = []
    def load(root, game):
        calls.append(game)
        return {"game": game}
    monkeypatch.setattr(dashboard, "load_metrica_bundle", load)
    for _ in range(2):
        for game in (1, 2, 3):
            assert dashboard.load_bundle(game, ("v4-cache-test",))["game"] == game
    assert calls == [1, 2, 3]

@pytest.mark.parametrize("selected_game", [1, 2])
def test_malformed_game3_heldout_source_shows_dashboard_error(tmp_path, selected_game):
    pytest.importorskip("kloppy")
    root = tmp_path / "sources"
    for game in (1, 2):
        folder = root / "data" / f"Sample_Game_{game}"
        folder.mkdir(parents=True)
        for suffix in ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team"):
            (folder / f"Sample_Game_{game}_Raw{suffix}.csv").write_text("source placeholder")
    folder = root / "data" / "Sample_Game_3"
    folder.mkdir()
    (folder / "Sample_Game_3_metadata.xml").write_text("<broken")
    (folder / "Sample_Game_3_tracking.txt").write_text("nonempty")
    (folder / "Sample_Game_3_events.json").write_text('{"data": []}')
    script = f'''
import sys, importlib
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
import app as dashboard
importlib.reload(dashboard)
import streamlit as st
from factories import make_bundle
from soccer_tactics.loaders.bundle import load_metrica_bundle
dashboard.DATA = Path({str(root)!r})
dashboard.IDSSE_DATA = Path({str(tmp_path / "idsse")!r})
dashboard.IDSSE_MANIFEST = Path({str(tmp_path / "manifest.json")!r})
st.session_state["selected_match"] = "metrica_{selected_game}"
def load(root, game):
    if game == 3:
        return load_metrica_bundle(root, game)
    return make_bundle("metrica_" + str(game))
dashboard.load_metrica_bundle = load
dashboard.main()
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="selected_match").value == f"metrica_{selected_game}"
    assert not app.title
    assert len(app.error) == 1
    message = app.error[0].value
    assert "Match analysis unavailable" in message
    assert "Game 3" in message and "metadata" in message
    assert str(folder) in message
    assert "restore" in message.lower() or "download" in message.lower()
