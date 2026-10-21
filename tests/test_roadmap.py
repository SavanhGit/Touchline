import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from soccer_tactics.analytics.impact import pass_values, player_impact_with_actions
from soccer_tactics.analytics.trajectory import constant_velocity_forecast
from soccer_tactics.cv import detections_to_tracking
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS, FPS


ROOT = Path(__file__).resolve().parents[1]


def test_roadmap_records_origin_only_idsse_scope_and_completed_release():
    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")
    assert "origin-only IDSSE" in roadmap
    assert "tracking synchronization" in roadmap
    assert "attack direction" in roadmap
    assert "pass endpoints" in roadmap
    assert "IDSSE xT" in roadmap
    assert "V2 origin-only release completed: 2026-10-21 11:30 EDT" in roadmap
    assert "held-out IDSSE evaluation" in roadmap


def _tracking():
    rows = []
    for frame in range(1, 31):
        for team, player, x in [
            ("home", "h1", frame / FPS * 2),
            ("away", "a1", 10.0),
        ]:
            rows.append(("m", 1, frame, frame / FPS, team, player, x, 0.0))
    return pd.DataFrame(rows, columns=COLUMNS)


def test_constant_velocity_forecast_returns_metre_prediction():
    out = constant_velocity_forecast(_tracking(), horizon_s=1.0)
    h1 = out[out["player_id"] == "h1"].iloc[0]
    assert np.isclose(h1["x_pred"], 30 / FPS * 2 + 2.0, atol=0.1)
    assert h1["horizon_s"] == 1.0


def test_video_adapter_emits_canonical_tracking():
    detections = pd.DataFrame(
        {"frame": [1], "team": ["home"], "player_id": ["h1"], "x": [3.0], "y": [2.0]}
    )
    out = detections_to_tracking(detections, match_id="video-1", fps=25)
    assert list(out.columns) == COLUMNS
    assert out.iloc[0]["time_s"] == 1 / 25


def test_video_adapter_rejects_unmapped_detections():
    with pytest.raises(ValueError, match="detections missing"):
        detections_to_tracking(pd.DataFrame({"frame": [1]}), match_id="video-1")


def test_action_impact_adds_defensive_and_shot_totals():
    pass_event = pd.DataFrame(
        [("m", 1, "home", "pass", True, "h1", "h2", 1, 0.04, -10, 0, 10, 0)],
        columns=EVENT_COLUMNS,
    )
    actions = pd.DataFrame(
        [
            ("m", 1, "home", "shot", True, "h1", None, 2, 0.08, 10, 0, 10, 0),
            ("m", 1, "home", "tackle", True, "h1", None, 3, 0.12, -10, 0, -10, 0),
        ],
        columns=EVENT_COLUMNS,
    )
    events = pd.concat([pass_event, actions], ignore_index=True)
    rows = [
        ("m", 1, frame, frame / FPS, "home", "h1", 0.0, 0.0)
        for frame in range(FPS * 60 * 15)
    ]
    rows += [
        ("m", 1, frame, frame / FPS, "away", "a1", 10.0, 0.0)
        for frame in range(FPS * 60 * 15)
    ]
    tracking = pd.DataFrame(rows, columns=COLUMNS)
    grid = np.array([[0.1], [0.4]])
    out = player_impact_with_actions(
        pass_values(pass_event, grid), events, tracking, grid, min_minutes=15
    ).set_index("player_id")
    assert out.loc["h1", "shots"] == 1
    assert out.loc["h1", "defensive_actions"] == 1
    assert out.loc["h1", "defensive_value_p90"] > 0
