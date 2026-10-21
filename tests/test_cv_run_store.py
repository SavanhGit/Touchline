import json

import pandas as pd
import pytest

from soccer_tactics.cv.adapter import TrackAssignment
from soccer_tactics.cv.run_store import save_video_run, source_sha256
from soccer_tactics.cv.video_types import VideoTrackingResult
from soccer_tactics.schema import validate


def video_result(tmp_path):
    overlay = tmp_path / "temporary-overlay.mp4"
    overlay.write_bytes(b"annotated video")
    return VideoTrackingResult(
        detections=pd.DataFrame({"frame": [15], "track_id": ["track_7"], "x": [10.0], "y": [4.0]}),
        frame_diagnostics=pd.DataFrame({"frame": [15], "calibration_valid": [True]}),
        video_metadata={"match_id": "clip", "fps": 10.0, "home_attacks_positive_x": False,
                        "source_sha256": "a" * 64, "source_path": "private/input.mp4",
                        "model_path": "private/weights.pt", "width": 1920,
                        "provenance": {
                            "models": {"player": {"id": "yolo11n", "version": "1"},
                                       "pitch": {"id": "pitch-keypoints", "version": "2.1"}},
                            "packages": {"ultralytics": "8.3.0", "supervision": "0.25.0"},
                        }},
        overlay_path=overlay,
    )


def test_source_hashes_content(tmp_path):
    source = tmp_path / "private.mp4"
    source.write_bytes(b"abc")
    assert source_sha256(source) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("version", ["1", "2.1", "8.3.0+cpu", "0.25.0rc1"])
def test_save_roundtrips_canonical_rows_and_only_safe_metadata(tmp_path, version):
    result = video_result(tmp_path)
    result.video_metadata["provenance"]["models"]["player"]["version"] = version
    root = tmp_path / "runs"
    run = save_video_run(result, assignments={"track_7": TrackAssignment("home", "Alex")},
                         output_root=root)
    assert run.parent == root
    tracking = validate(pd.read_parquet(run / "tracking.parquet"))
    assert tracking.player_id.tolist() == ["track_7"]
    assert tracking[["x", "y", "time_s"]].values.tolist() == [[-10.0, -4.0, 1.5]]
    pd.testing.assert_frame_equal(pd.read_parquet(run / "frame_diagnostics.parquet"),
                                  result.frame_diagnostics)
    assert (run / "overlay.mp4").read_bytes() == b"annotated video"
    metadata_text = (run / "metadata.json").read_text()
    metadata = json.loads(metadata_text)
    assert metadata["source_sha256"] == "a" * 64
    assert metadata["assignments"] == {"track_7": {"team": "home", "roster_name": "Alex"}}
    assert metadata["run_id"] == run.name
    assert metadata["home_attacks_positive_x"] is False
    assert metadata["provenance"] == {
        "models": {"player": {"id": "yolo11n", "version": version},
                   "pitch": {"id": "pitch-keypoints", "version": "2.1"}},
        "packages": {"ultralytics": "8.3.0", "supervision": "0.25.0"},
    }
    assert "private" not in metadata_text
    assert "source_path" not in metadata
    assert "model_path" not in metadata
    assert sorted(p.name for p in run.iterdir()) == [
        "frame_diagnostics.parquet", "metadata.json", "overlay.mp4", "tracking.parquet"]
    second = save_video_run(result, assignments={}, output_root=root)
    assert second != run


def test_failed_run_leaves_no_published_or_partial_directory(tmp_path, monkeypatch):
    result = video_result(tmp_path)
    root = tmp_path / "runs"

    def failed_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failed_write)
    with pytest.raises(OSError, match="disk full"):
        save_video_run(result, assignments={}, output_root=root)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("provenance", [
    {"models": {"player": {"id": "C:\\private\\weights.pt", "version": "1"}}, "packages": {}},
    {"models": {"player": {"id": "/private/weights.pt", "version": "1"}}, "packages": {}},
    {"models": {"player": {"id": "weights.pt", "version": "1"}}, "packages": {}},
    {"models": {}, "packages": {"weights.pt": "1.2.3"}},
    {"models": {"player": {"id": "yolo11n", "version": "../private"}}, "packages": {}},
    {"models": {"player": {"id": "yolo11n", "version": "1", "path": "private.pt"}}, "packages": {}},
    {"models": {}, "packages": {"ultralytics": {"version": "8.3.0", "path": "private"}}},
    {"models": {}, "packages": {"ultralytics": "file:///private/package"}},
    {"models": {}, "packages": {}, "source_path": "private.mp4"},
    {"models": [], "packages": {}},
    None,
])
def test_unsafe_provenance_is_rejected_before_creating_run(tmp_path, provenance):
    result = video_result(tmp_path)
    result.video_metadata["provenance"] = provenance
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="provenance"):
        save_video_run(result, assignments={}, output_root=root)
    assert not root.exists()
