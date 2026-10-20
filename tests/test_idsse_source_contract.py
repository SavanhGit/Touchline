"""Synthetic contracts at the optional Floodlight/IDSSE boundary."""
from pathlib import Path
from types import SimpleNamespace
import builtins

import pandas as pd
import pytest

from soccer_tactics.loaders import idsse
from soccer_tactics.schema import EVENT_COLUMNS


MATCH = "DFL-MAT-123"


def triplet(root: Path, match_id: str = MATCH) -> dict[str, Path]:
    files = {
        "info": root / f"matchinformation_{match_id}.xml",
        "events": root / f"events_raw_{match_id}.xml",
        "positions": root / f"positions_raw_observed_{match_id}.xml",
    }
    for path in files.values():
        path.write_text("<root/>", encoding="utf-8")
    return files


def source_events(*rows, segment="firstHalf", team="Home"):
    frame = pd.DataFrame(rows)
    return {segment: {team: SimpleNamespace(events=frame)}}


def event(eid, *, clock=1.25, x0=10.0, y0=20.0, x1=30.0, y1=40.0,
          outcome=1, player="p1", qualifier=None):
    return dict(eID=eid, gameclock=clock, timestamp=pd.Timestamp("2025-01-01T12:00:01.250Z"),
                at_x=x0, at_y=y0, to_x=x1, to_y=y1, outcome=outcome,
                pID=player, qualifier=qualifier or {})


def test_triplets_require_one_file_per_family(tmp_path):
    paths = triplet(tmp_path)
    assert idsse.list_idsse_matches(tmp_path) == [MATCH]
    paths["events"].unlink()
    assert idsse.list_idsse_matches(tmp_path) == []
    with pytest.raises(FileNotFoundError, match="complete IDSSE XML set"):
        idsse._read_idsse_source(tmp_path, MATCH)


def test_duplicate_file_family_is_rejected(tmp_path):
    triplet(tmp_path)
    (tmp_path / f"other_events_raw_{MATCH}.xml").write_text("<root/>")
    assert idsse.list_idsse_matches(tmp_path) == []
    with pytest.raises(ValueError, match="duplicate.*events"):
        idsse._read_idsse_source(tmp_path, MATCH)


def test_source_reader_keeps_distinct_period_objects_and_parser_outputs(tmp_path, monkeypatch):
    paths = triplet(tmp_path)
    positions = {"firstHalf": {"Home": object()}, "secondHalf": {"Home": object()}}
    events = {"firstHalf": {"Home": object()}, "secondHalf": {"Home": object()}}
    teamsheets, pitch = object(), object()
    calls = []

    def read_positions(pos, info):
        calls.append(("positions", pos, info))
        return positions, None, None, teamsheets, pitch

    def read_events(evt, info):
        calls.append(("events", evt, info))
        return events, teamsheets, pitch

    monkeypatch.setattr(idsse, "_require_floodlight", lambda: (read_events, read_positions))
    source = idsse._read_idsse_source(tmp_path, MATCH)
    assert source.match_id == MATCH
    assert source.files == paths
    assert source.positions is positions and source.events is events
    assert source.teamsheets is teamsheets and source.pitch is pitch
    assert source.match_information.tag == "root"
    assert source.positions["firstHalf"] is not source.positions["secondHalf"]
    assert calls == [("positions", paths["positions"], paths["info"]),
                     ("events", paths["events"], paths["info"])]


def test_floodlight_is_required_only_after_file_validation(tmp_path, monkeypatch):
    paths = triplet(tmp_path)
    monkeypatch.setattr(idsse, "_require_floodlight", lambda: (_ for _ in ()).throw(
        ImportError("IDSSE loading requires pip install -e '.[idsse]'")))
    with pytest.raises(ImportError, match=r"pip install -e '.\[idsse\]'"):
        idsse._read_idsse_source(tmp_path, MATCH)
    paths["events"].unlink()
    with pytest.raises(FileNotFoundError):
        idsse._read_idsse_source(tmp_path, MATCH)


def test_missing_floodlight_error_gives_optional_install_command(monkeypatch):
    original_import = builtins.__import__

    def missing_floodlight(name, *args, **kwargs):
        if name == "floodlight.io.dfl":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_floodlight)
    with pytest.raises(ImportError, match=r"pip install -e '.\[idsse\]'"):
        idsse._require_floodlight()


def test_exact_codes_and_bad_spatial_or_time_keys_are_counted():
    source = source_events(
        event("Play_Pass"),
        event("Play_Cross", x0=11),
        event("MyPassImposter"),
        event("Play_Pass", x1=None),
        event("ShotAtGoal_SavedShot", clock=None),
    )
    selection = idsse.select_idsse_events(source, MATCH)
    assert selection.events.columns.tolist() == EVENT_COLUMNS
    assert selection.events.event.tolist() == ["pass", "pass"]
    assert selection.events.frame.tolist() == [-1, -1]
    assert selection.events.time_s.tolist() == [1.25, 1.25]
    assert selection.total == 5
    assert selection.exclusions.reason.value_counts().to_dict() == {
        "unsupported_event_code": 1,
        "missing_pass_endpoint": 1,
        "invalid_event_time": 1,
    }
    assert selection.exclusions.source_code.tolist() == [
        "MyPassImposter", "Play_Pass", "ShotAtGoal_SavedShot"]


def test_goal_records_credit_own_goal_to_opponent_and_keep_penalty_separate():
    source = source_events(
        event("ShotAtGoal_SuccessfulShot", clock=1),
        event("Penalty_ShotAtGoal_SuccessfulShot", clock=2),
        event("OwnGoal", clock=3, outcome=None),
        event("FreeKick", clock=4, outcome=None),
        event("ShotAtGoal_BlockedShot", clock=5, outcome=0),
    )
    selection = idsse.select_idsse_events(source, MATCH)
    assert selection.goals[["team", "penalty", "source_code"]].to_dict("records") == [
        {"team": "home", "penalty": False, "source_code": "ShotAtGoal_SuccessfulShot"},
        {"team": "home", "penalty": True, "source_code": "Penalty_ShotAtGoal_SuccessfulShot"},
        {"team": "away", "penalty": False, "source_code": "OwnGoal"},
    ]
    assert selection.events.event.tolist() == ["shot", "shot"]
    assert selection.events.success.tolist() == [True, False]
    assert selection.exclusions.reason.value_counts().to_dict() == {
        "penalty_shot_excluded": 1,
        "own_goal_not_modeling_shot": 1,
        "unsupported_event_code": 1,
    }


def test_second_half_events_keep_period_and_original_timestamp():
    source = source_events(event("Play_Pass", clock=0.5), segment="secondHalf", team="Away")
    selection = idsse.select_idsse_events(source, MATCH)
    row = selection.events.iloc[0]
    assert (row.period, row.team, row.time_s) == (2, "away", 0.5)
    assert selection.source_events.iloc[0].timestamp == pd.Timestamp("2025-01-01T12:00:01.250Z")
    assert selection.source_events.iloc[0].canonical_index == 0


def test_pass_without_documented_outcome_is_excluded():
    selection = idsse.select_idsse_events(source_events(
        event("Play_Pass", outcome=None)), MATCH)
    assert selection.events.empty
    assert selection.exclusions.reason.tolist() == ["missing_pass_outcome"]


def test_floodlight_pass_evaluation_supplies_completed_and_failed_outcomes():
    selection = idsse.select_idsse_events(source_events(
        event("Play_Pass", outcome=float("nan"),
              qualifier={"Evaluation": "successfullyCompleted"}),
        event("Play_Cross", outcome=float("nan"),
              qualifier={"Evaluation": "unsuccessful"}),
    ), MATCH)
    assert selection.events.success.tolist() == [True, False]
    assert selection.source_events.canonical_index.tolist() == [0, 1]
    assert selection.exclusions.empty


def test_ambiguous_or_unknown_pass_evaluations_are_excluded():
    selection = idsse.select_idsse_events(source_events(
        event("Play_Pass", outcome=float("nan"),
              qualifier={"Evaluation": "successful"}),
        event("Play_Pass", outcome=float("nan"),
              qualifier={"Evaluation": "unknownValue"}),
    ), MATCH)
    assert selection.events.empty
    assert selection.exclusions.reason.tolist() == [
        "unknown_pass_evaluation", "unknown_pass_evaluation"]


def test_canonical_pass_outcomes_require_binary_values_and_agree_with_evaluation():
    selection = idsse.select_idsse_events(source_events(
        event("Play_Pass", outcome=1),
        event("Play_Pass", outcome=0),
        event("Play_Pass", outcome=1, qualifier={"Evaluation": "unsuccessful"}),
        event("Play_Pass", outcome=2, qualifier={"Evaluation": "successfullyCompleted"}),
    ), MATCH)
    assert selection.events.success.tolist() == [True, False]
    assert selection.exclusions.reason.tolist() == [
        "conflicting_pass_outcome", "invalid_pass_outcome"]


def test_scored_goal_survives_missing_modeling_coordinates():
    selection = idsse.select_idsse_events(source_events(
        event("ShotAtGoal_SuccessfulShot", x0=None)), MATCH)
    assert selection.goals.team.tolist() == ["home"]
    assert selection.events.empty
    assert selection.exclusions.reason.tolist() == ["missing_event_origin"]


def test_nested_documented_scored_shots_are_goals_even_when_not_modeling_codes():
    selection = idsse.select_idsse_events(source_events(
        event("FreeKick_ShotAtGoal_SuccessfulShot", clock=1),
        event("CornerKick_Play_ShotAtGoal_SuccessfulShot", clock=2),
        event("KickOff_Play_ShotAtGoal_SuccessfulShot", clock=3),
        event("Mystery_ShotAtGoal_SuccessfulShot", clock=4),
        event("FreeKick", clock=5),
    ), MATCH)
    assert selection.goals.source_code.tolist() == [
        "FreeKick_ShotAtGoal_SuccessfulShot",
        "CornerKick_Play_ShotAtGoal_SuccessfulShot",
        "KickOff_Play_ShotAtGoal_SuccessfulShot",
    ]
    assert selection.events.empty
    assert selection.exclusions.reason.tolist() == [
        "unsupported_modeling_goal_code", "unsupported_modeling_goal_code",
        "unsupported_modeling_goal_code", "unsupported_event_code",
        "unsupported_event_code",
    ]


def test_position_clock_reads_frame_attributes_before_clearing_parent(tmp_path):
    xml = tmp_path / "positions.xml"
    xml.write_text("""<root><FrameSet GameSection="firstHalf" TeamId="ball">
      <Frame N="17" T="2025-01-01T00:00:00.000Z"/>
      <Frame N="19" T="2025-01-01T00:00:00.080Z"/>
    </FrameSet></root>""")
    assert idsse._position_clocks(xml)["firstHalf"].to_dict("records") == [
        {"frame": 17, "timestamp": "2025-01-01T00:00:00.000Z"},
        {"frame": 19, "timestamp": "2025-01-01T00:00:00.080Z"},
    ]


def test_position_clocks_preserve_repeated_raw_frame_ids_and_timestamps(tmp_path, monkeypatch):
    paths = triplet(tmp_path)
    paths["positions"].write_text("""<root>
      <FrameSet TeamId="ball" GameSection="firstHalf">
        <Frame N="900" T="2025-01-01T12:00:00.000Z" X="1" Y="2"/>
        <Frame N="901" T="2025-01-01T12:00:00.040Z" X="2" Y="3"/>
      </FrameSet>
      <FrameSet TeamId="ball" GameSection="secondHalf">
        <Frame N="900" T="2025-01-01T13:00:00.000Z" X="4" Y="5"/>
      </FrameSet>
    </root>""")
    monkeypatch.setattr(idsse, "_require_floodlight", lambda: (
        lambda *_: ({}, {}, None),
        lambda *_: ({}, None, None, {}, None),
    ))
    source = idsse._read_idsse_source(tmp_path, MATCH)
    assert source.position_clocks["firstHalf"].to_dict("records") == [
        {"frame": 900, "timestamp": "2025-01-01T12:00:00.000Z"},
        {"frame": 901, "timestamp": "2025-01-01T12:00:00.040Z"},
    ]
    assert source.position_clocks["secondHalf"].to_dict("records") == [
        {"frame": 900, "timestamp": "2025-01-01T13:00:00.000Z"},
    ]
