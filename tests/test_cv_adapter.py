import pandas as pd
import pytest

from soccer_tactics.cv.adapter import (
    TrackAssignment, TrackingAdapter, detections_to_tracking, video_tracks_to_tracking,
)


def tracks():
    return pd.DataFrame({"frame": [15, 15, 15], "track_id": ["track_7", "track_8", "track_9"],
                         "x": [10.0, -8.0, 0.0], "y": [4.0, -3.0, 0.0]})


def test_mapping_keeps_track_identity_and_omits_unassigned_tracks():
    result = video_tracks_to_tracking(
        tracks(), match_id="clip", fps=10,
        assignments={"track_7": TrackAssignment("home", "Alex"),
                     "track_8": TrackAssignment("away", "Alex")},
        home_attacks_positive_x=True,
    )
    assert result.player_id.tolist() == ["track_7", "track_8"]
    assert result.team.tolist() == ["home", "away"]
    assert result.time_s.tolist() == [1.5, 1.5]
    assert result.x.tolist() == [10.0, -8.0]
    assert result.y.tolist() == [4.0, -3.0]


def test_mapping_rotates_both_coordinates_and_allows_empty_assignment():
    result = video_tracks_to_tracking(
        tracks(), match_id="clip", fps=10, assignments={"track_7": TrackAssignment("home")},
        home_attacks_positive_x=False, period=2,
    )
    assert result[["x", "y"]].values.tolist() == [[-10.0, -4.0]]
    assert result.period.tolist() == [2]
    assert video_tracks_to_tracking(tracks(), match_id="clip", fps=10, assignments={},
                                    home_attacks_positive_x=True).empty


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_mapping_rejects_invalid_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        video_tracks_to_tracking(tracks(), match_id="clip", fps=fps, assignments={},
                                 home_attacks_positive_x=True)


def test_assignment_rejects_unknown_teams():
    with pytest.raises(ValueError, match="team"):
        TrackAssignment("ball")


def test_legacy_adapter_and_conversion_remain_compatible():
    detections = pd.DataFrame({"frame": [25], "team": ["home"], "player_id": ["p1"],
                               "x": [1.0], "y": [2.0]})
    expected = detections_to_tracking(detections, match_id="video")
    pd.testing.assert_frame_equal(TrackingAdapter(lambda _: detections)("video"), expected)
    assert expected.time_s.tolist() == [1.0]
