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


@pytest.fixture(autouse=True)
def clear_dashboard_caches():
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def synthetic_app(tmp_path, *, sources=6, empty=False, unavailable=False, known_score=False):
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
import streamlit as st
dashboard.DATA = Path({str(root)!r})
if "loader_calls" not in st.session_state:
    st.session_state["loader_calls"] = 0
bundles, reports = pickle.loads(Path({str(payload)!r}).read_bytes())
def bundle(game, revision):
    st.session_state["loader_calls"] = st.session_state.get("loader_calls", 0) + 1
    return bundles[game]
dashboard.load_bundle = bundle
dashboard.load_report = lambda game, team, revision: reports[game, team]
dashboard.main()
'''
    return AppTest.from_string(script, default_timeout=30)


@pytest.mark.parametrize("sources", [-1, 0, 5, 7])
def test_missing_sources_stop_before_modeling(tmp_path, sources):
    app = synthetic_app(tmp_path, sources=sources).run()
    assert not app.exception
    assert app.session_state["loader_calls"] == 0
    assert any("source" in item.value.lower() for item in app.warning)
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
    app.selectbox(key="selected_match").select(2).run()
    assert not app.exception
    assert any("No qualifying evidence" in item.value for item in app.info)
    assert not any("Evidence ID:" in item.value for item in app.caption)
    app.selectbox(key="selected_match").select(1).run()
    app.selectbox(key="selected_team").select("away").run()
    assert not app.exception
    assert not any("Evidence ID:" in item.value for item in app.caption)
    app.selectbox(key="selected_team").select("home").run()
    assert not app.exception
    assert any("Evidence ID: 1:" in item.value for item in app.caption)


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
    app.selectbox(key="selected_match").select(2).run()
    assert not app.exception
