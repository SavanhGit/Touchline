from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from conftest import GAMES


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
