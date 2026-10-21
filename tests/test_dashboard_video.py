"""Video UI contracts use fake inference; decoding and persistence stay real."""
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.fixture
def preload_parquet_engine():
    # AppTest rolls back newly imported modules, while Arrow's C registry survives.
    # Initialize pandas' Arrow extensions outside its script before save/read tests.
    pytest.importorskip("pyarrow")
    pd.DataFrame({"warmup": [1]}).to_parquet(BytesIO(), index=False)


def video_app(tmp_path, *, invalid=False, failure=False, missing_cv=False, track_ids=("7",)):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    clip = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    payload = b"invalid" if invalid else clip.read_bytes()
    detections = [{"frame": 0, "track_id": track_id, "x": 2., "y": 3.}
                  for track_id in track_ids]
    script = f'''
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import streamlit as st
from soccer_tactics.dashboard.video_tracking import render_video_tracking
from soccer_tactics.cv.video_types import VideoTrackingResult
upload = BytesIO({payload!r})
upload.name = "private-source.mp4"
def processor(source, config, **kwargs):
    st.session_state["temporary_source"] = str(source)
    st.session_state["calls"] = st.session_state.get("calls", 0) + 1
    if {failure!r}:
        raise RuntimeError("missing dependency at C:/private/weights.pt")
    overlay = kwargs["work_dir"] / "overlay.mp4"
    overlay.write_bytes(source.read_bytes())
    kwargs["on_progress"](1, 1)
    return VideoTrackingResult(
        pd.DataFrame({detections!r}),
        pd.DataFrame([{{"frame": 0, "calibration_valid": True}}]),
        {{"match_id": config.match_id, "fps":25., "source_sha256":"a"*64,
          "home_attacks_positive_x":config.home_attacks_positive_x}}, overlay)
real_editor = st.data_editor
def editor(*args, **kwargs):
    table = real_editor(*args, **kwargs)
    # AppTest cannot interact with data_editor yet; substitute only its edited output.
    for row, changes in st.session_state.get("_mapping_edits", {{}}).items():
        for column, value in changes.items():
            table.loc[row, column] = value
    return table
with patch.object(st, "file_uploader", return_value=upload), patch.object(st, "data_editor", side_effect=editor), patch.dict("sys.modules", {{"cv2": None}} if {missing_cv!r} else {{}}):
    render_video_tracking(st, processor=processor, output_root=Path({str(tmp_path / 'runs')!r}))
'''
    return AppTest.from_string(script, default_timeout=30)


def process(app):
    app.run()
    app.radio(key="video_direction").set_value("Home attacks +x").run()
    app.button(key="process_video").click().run()
    return app


def test_fake_processing_review_mapping_and_export(tmp_path, preload_parquet_engine):
    app = process(video_app(tmp_path))
    assert not app.exception
    assert len(app.get("video")) == 1
    assert any(item.label == "Download review overlay" for item in app.get("download_button"))
    assert app.dataframe[-1].value.track_id.tolist() == ["7"]
    assert any("Track mapping" in item.value for item in app.subheader)
    assert not Path(app.session_state["temporary_source"]).exists()
    assert app.button(key="save_video_run").disabled
    app.session_state["_mapping_edits"] = {0: {"team": "home", "roster_name": "Ada"}}
    app.run()
    app.button(key="save_video_run").click().run()
    assert not app.exception
    assert any(item.label == "Download run" for item in app.get("download_button"))
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert set(p.name for p in runs[0].iterdir()) == {"tracking.parquet", "frame_diagnostics.parquet", "metadata.json", "overlay.mp4"}
    tracking = pd.read_parquet(runs[0] / "tracking.parquet")
    assert tracking.player_id.tolist() == ["7"]
    assert tracking.team.tolist() == ["home"]


def test_invalid_upload_is_actionable_and_never_processed(tmp_path):
    app = process(video_app(tmp_path, invalid=True))
    assert not app.exception
    assert app.error
    assert "MP4" in app.error[0].value
    assert not (tmp_path / "runs").exists()


def test_missing_optional_setup_is_contained_and_hides_source_paths(tmp_path):
    app = process(video_app(tmp_path, failure=True))
    assert not app.exception
    assert "soccer-tactics[cv]" in app.error[0].value
    assert "C:/private" not in app.error[0].value
    assert not Path(app.session_state["temporary_source"]).exists()
    assert not (tmp_path / "runs").exists()


def test_missing_opencv_keeps_workspace_controls_available(tmp_path):
    app = process(video_app(tmp_path, missing_cv=True))
    assert not app.exception
    assert app.title[0].value == "Local video tracking"
    assert app.radio(key="video_direction")
    assert "soccer-tactics[cv]" in app.error[0].value
    assert not (tmp_path / "runs").exists()


def test_attack_direction_required_and_changing_it_clears_review(tmp_path):
    app = video_app(tmp_path).run()
    app.button(key="process_video").click().run()
    assert app.error
    assert "attack direction" in app.error[0].value
    app.radio(key="video_direction").set_value("Home attacks +x").run()
    app.button(key="process_video").click().run()
    assert len(app.get("video")) == 1
    app.radio(key="video_direction").set_value("Home attacks −x").run()
    assert not app.get("video")


def test_mapping_edit_invalidates_previous_export(tmp_path, preload_parquet_engine):
    app = process(video_app(tmp_path))
    app.session_state["_mapping_edits"] = {0: {"team": "away"}}
    app.run()
    app.button(key="save_video_run").click().run()
    assert any(item.label == "Download run" for item in app.get("download_button"))
    app.session_state["_mapping_edits"] = {0: {"team": "home"}}
    app.run()
    assert not any(item.label == "Download run" for item in app.get("download_button"))


def test_excluded_track_has_no_persisted_rows_or_assignment(tmp_path, preload_parquet_engine):
    app = process(video_app(tmp_path, track_ids=("7", "8")))
    assert not app.exception
    assert app.dataframe[-1].value.track_id.tolist() == ["7", "8"]
    app.session_state["_mapping_edits"] = {
        0: {"team": "home", "roster_name": "Ada"},
        1: {"team": "exclude", "roster_name": "Excluded player"},
    }
    app.run()
    assert not app.button(key="save_video_run").disabled
    app.button(key="save_video_run").click().run()
    assert not app.exception
    assert any(item.label == "Download run" for item in app.get("download_button"))
    run, = (tmp_path / "runs").iterdir()
    tracking = pd.read_parquet(run / "tracking.parquet")
    assert tracking.player_id.tolist() == ["7"]
    assert tracking.team.tolist() == ["home"]
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["assignments"] == {"7": {"team": "home", "roster_name": "Ada"}}


def test_video_workspace_bypasses_missing_match_catalogue(tmp_path):
    from test_dashboard import synthetic_app
    app = synthetic_app(tmp_path, sources=0).run()
    app.sidebar.selectbox(key="workspace").select("Video tracking").run()
    assert not app.exception
    assert app.title[0].value == "Local video tracking"
    assert not app.sidebar.warning
    assert app.session_state["loader_calls"] == 0
