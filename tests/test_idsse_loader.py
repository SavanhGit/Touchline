"""Compatibility checks for the public IDSSE loader."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from soccer_tactics.loaders import idsse
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS


def test_public_loader_returns_two_canonical_tables(tmp_path, monkeypatch):
    match = "DFL-MAT-9"
    (tmp_path / f"matchinformation_{match}.xml").write_text("<root/>")
    (tmp_path / f"events_raw_{match}.xml").write_text("<root/>")
    (tmp_path / f"positions_raw_observed_{match}.xml").write_text("""<root>
      <FrameSet TeamId="ball" GameSection="firstHalf">
        <Frame N="42" T="2025-01-01T12:00:00.000Z"/>
        <Frame N="43" T="2025-01-01T12:00:00.040Z"/>
      </FrameSet>
    </root>""")
    xy = SimpleNamespace(x=np.array([[1.0], [2.0]]), y=np.array([[3.0], [4.0]]), framerate=25)
    positions = {"firstHalf": {"Home": xy, "Away": xy, "Ball": xy}}
    sheets = {team: SimpleNamespace(teamsheet=pd.DataFrame({"xID": [0], "pID": [f"{team}1"]}))
              for team in ("Home", "Away")}
    events = {"firstHalf": {"Home": SimpleNamespace(events=pd.DataFrame([
        {"eID": "Play_Pass", "gameclock": .04, "timestamp": pd.Timestamp("2025-01-01T12:00:00.040Z"),
         "at_x": 1., "at_y": 2., "to_x": 3., "to_y": 4., "outcome": 1,
         "pID": "Home1", "qualifier": {}}
    ]))}}
    monkeypatch.setattr(idsse, "_require_floodlight", lambda: (
        lambda *_: (events, sheets, None),
        lambda *_: (positions, None, None, sheets, None),
    ))
    tracking, modeled = idsse.load_idsse_game(tmp_path, match)
    assert tracking.columns.tolist() == COLUMNS
    assert modeled.columns.tolist() == EVENT_COLUMNS
    assert tracking.frame.unique().tolist() == [42, 43]
    assert modeled[["period", "team", "event", "frame", "time_s"]].to_dict("records") == [
        {"period": 1, "team": "home", "event": "pass", "frame": -1, "time_s": .04}
    ]


def test_sparse_source_frames_index_floodlight_xy_by_n_offset(tmp_path, monkeypatch):
    match = "DFL-MAT-10"
    (tmp_path / f"matchinformation_{match}.xml").write_text("<root/>")
    (tmp_path / f"events_raw_{match}.xml").write_text("<root/>")
    (tmp_path / f"positions_raw_observed_{match}.xml").write_text("""<root>
      <FrameSet TeamId="ball" GameSection="firstHalf">
        <Frame N="42" T="2025-01-01T12:00:00.000Z"/>
        <Frame N="45" T="2025-01-01T12:00:00.120Z"/>
      </FrameSet>
    </root>""")
    # Floodlight allocates N=42..45, including missing 43 and 44.
    xy = SimpleNamespace(x=np.array([[1.], [np.nan], [np.nan], [4.]]),
                         y=np.array([[2.], [np.nan], [np.nan], [5.]]), framerate=25)
    positions = {"firstHalf": {"Home": xy, "Away": xy, "Ball": xy}}
    sheets = {team: SimpleNamespace(teamsheet=pd.DataFrame({"xID": [0], "pID": [f"{team}1"]}))
              for team in ("Home", "Away")}
    monkeypatch.setattr(idsse, "_require_floodlight", lambda: (
        lambda *_: ({}, sheets, None),
        lambda *_: (positions, None, None, sheets, None),
    ))
    tracking, events = idsse.load_idsse_game(tmp_path, match)
    assert events.empty
    assert tracking[tracking.team == "ball"][["frame", "time_s", "x", "y"]].to_dict("records") == [
        {"frame": 42, "time_s": 0., "x": 1., "y": 2.},
        {"frame": 45, "time_s": .12, "x": 4., "y": 5.},
    ]
