"""Bundle contracts from a small two-period DFL/Floodlight-shaped source."""
from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from soccer_tactics import loaders
from soccer_tactics.loaders import idsse
from soccer_tactics.match import validate_bundle


MATCH = "DFL-MAT-FIXTURE"


def load_idsse_bundle(root, match_id):
    return loaders.load_idsse_bundle(root, match_id)


def _event(code, timestamp, *, clock=0.05, x=80., y=10., to_x=82., to_y=12.,
           player="h1", outcome=1):
    return dict(eID=code, timestamp=pd.Timestamp(timestamp), gameclock=clock,
                at_x=x, at_y=y, to_x=to_x, to_y=to_y, pID=player,
                outcome=outcome, qualifier={})


@pytest.fixture
def source(monkeypatch):
    metadata = ET.fromstring("""<PutDataRequest><MatchInformation>
      <General MatchId="DFL-MAT-FIXTURE" KickoffTime="2025-04-05T12:00:00Z"
       CompetitionName="Bundesliga" HomeTeamName="Home FC" HomeTeamId="home-id"
       GuestTeamName="Visitors" GuestTeamId="away-id" Result="2:1"/>
      <Environment PitchX="110.00" PitchY="70.00"/>
    </MatchInformation></PutDataRequest>""")
    def xy(x, y):
        return SimpleNamespace(x=np.array([[x], [x], [x]], dtype=float),
                               y=np.array([[y], [y], [y]], dtype=float), framerate=20)
    positions = {
        "firstHalf": {"Home": xy(20, -24), "Away": xy(-20, 24), "Ball": xy(0, 0)},
        "secondHalf": {"Home": xy(-20, 24), "Away": xy(20, -24), "Ball": xy(0, 0)},
    }
    clocks = {
        period: pd.DataFrame({"frame": [100, 101, 102],
                              "timestamp": [f"2025-04-05T{hour}:00:00.000Z",
                                            f"2025-04-05T{hour}:00:00.050Z",
                                            f"2025-04-05T{hour}:00:00.100Z"]})
        for period, hour in (("firstHalf", "12"), ("secondHalf", "13"))
    }
    sheets = {team: SimpleNamespace(teamsheet=pd.DataFrame({"xID": [0], "pID": [pid]}))
              for team, pid in (("Home", "h1"), ("Away", "a1"))}
    events = {
        "firstHalf": {"Home": SimpleNamespace(events=pd.DataFrame([
            _event("Play_Pass", "2025-04-05T12:00:00.015Z", clock=.015),
            _event("ShotAtGoal_SuccessfulShot", "2025-04-05T12:00:00.050Z"),
            _event("UnknownKick", "2025-04-05T12:00:00.000Z"),
            _event("Play_Pass", "2025-04-05T12:00:00.000Z", to_x=None),
            _event("Play_Pass", "2025-04-05T13:00:00.000Z", clock=0.),
        ]))},
        "secondHalf": {"Home": SimpleNamespace(events=pd.DataFrame([
            _event("Penalty_ShotAtGoal_SuccessfulShot", "2025-04-05T13:00:00.050Z",
                   x=30., y=60.),
            _event("OwnGoal", "2025-04-05T13:00:00.000Z", clock=0.),
            _event("FreeKick_ShotAtGoal_SavedShot", "2025-04-05T13:00:00.100Z"),
        ]))},
    }
    match = idsse.IdsseSourceMatch(MATCH, {}, positions, events, sheets, None, metadata, clocks)
    monkeypatch.setattr(idsse, "_read_idsse_source", lambda root, match_id: match)
    return match


def test_bundle_retains_source_metadata_and_unoriented_tracking(source):
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.match_id == MATCH and bundle.source == "IDSSE/DFL"
    assert bundle.team_names == {"home": "Home FC", "away": "Visitors"}
    assert bundle.metadata["match_date"] == "2025-04-05"
    assert bundle.metadata["competition"] == "Bundesliga"
    assert bundle.metadata["source_team_ids"] == {"home": "home-id", "away": "away-id"}
    assert bundle.metadata["source_team_labels"] == bundle.team_names
    assert bundle.metadata["source_pitch_dimensions_m"] == {"length": 110., "width": 70.}
    assert "no attack-direction rotation" in bundle.metadata["transforms"]["tracking"]
    assert bundle.fps == 20.
    home = bundle.tracking.query("team == 'home'").drop_duplicates("period")
    assert home[["period", "x", "y"]].to_dict("records") == [
        {"period": 1, "x": 20., "y": -24.}, {"period": 2, "x": -20., "y": 24.}]
    first_pass = bundle.event_origins.query("period == 1 and event == 'pass'").iloc[0]
    assert (first_pass.x_m, first_pass.y_m) == (25., -25.)
    second_shot = bundle.goals.query("period == 2 and penalty").iloc[0]
    assert (second_shot.period, second_shot.frame, second_shot.time_s) == (2, -1, .05)
    validate_bundle(bundle)


def test_source_event_clocks_remain_separate_from_observed_tracking_clocks(source):
    bundle = load_idsse_bundle("unused", MATCH)
    passes = bundle.event_origins.query("event == 'pass'")
    assert passes.gameclock_s.tolist() == [.015, .05, 0.]
    assert passes.source_timestamp.iloc[-1] == pd.Timestamp("2025-04-05T13:00:00Z")
    assert bundle.tracking.query("period == 2 and frame == 100").shape[0] == 3
    assert "event_alignment" not in bundle.quality
    assert bundle.metadata["clock_provenance"]["gameclock_s"].startswith("source period-relative")
    assert bundle.quality["event_exclusion_reasons"]["unsupported_event_code"] == 2
    assert bundle.quality["source_event_codes"]["UnknownKick"] == 1


def test_complete_score_includes_penalty_and_own_goal_but_model_excludes_them(source):
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.score_known and bundle.score == {"home": 2, "away": 1}
    assert bundle.goals[["team", "period", "frame", "penalty"]].to_dict("records") == [
        {"team": "home", "period": 1, "frame": -1, "penalty": False},
        {"team": "home", "period": 2, "frame": -1, "penalty": True},
        {"team": "away", "period": 2, "frame": -1, "penalty": False},
    ]
    assert bundle.event_origins.event.tolist() == ["pass", "shot", "pass", "pass"]
    assert bundle.quality["event_exclusion_reasons"]["penalty_shot_excluded"] == 1
    assert bundle.quality["event_exclusion_reasons"]["own_goal_not_modeling_shot"] == 1


def test_invalid_origin_team_is_rejected(source):
    bundle = load_idsse_bundle("unused", MATCH)
    invalid = replace(bundle, event_origins=bundle.event_origins.assign(team="Visitors"))
    with pytest.raises(ValueError, match="team"):
        validate_bundle(invalid)


def test_duplicate_origin_ids_are_rejected(source):
    bundle = load_idsse_bundle("unused", MATCH)
    origins = bundle.event_origins.copy()
    origins.loc[1, "origin_id"] = origins.loc[0, "origin_id"]
    with pytest.raises(ValueError, match="origin IDs"):
        validate_bundle(replace(bundle, event_origins=origins))


def test_observed_rate_is_diagnostic_only(source):
    source.position_clocks["secondHalf"]["timestamp"] = [
        "2025-04-05T13:00:00.000Z", "2025-04-05T13:00:00.100Z",
        "2025-04-05T13:00:00.200Z"]
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.quality["source_fps_by_period"] == {1: 20., 2: 10.}
    penalty = bundle.goals.query("penalty").iloc[0]
    assert penalty.frame == -1
    assert "event_alignment" not in bundle.quality


def test_source_missing_positions_are_counted_without_invented_tracking_rows(source):
    source.positions["firstHalf"]["Home"].x[1, 0] = np.nan
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.quality["source_missing_positions"] == 1
    assert bundle.quality["tracking_rows"] == 17
    assert bundle.tracking.query("period == 1 and frame == 101 and team == 'home'").empty


def test_source_missing_positions_count_never_observed_player(source):
    source.positions["firstHalf"]["Home"].x[:, 0] = np.nan
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.quality["source_missing_positions"] == 3
    assert bundle.quality["tracking_rows"] == 15
    assert bundle.tracking.query("period == 1 and team == 'home'").empty


@pytest.mark.parametrize("timestamp", [pd.NaT, "not a timestamp", None])
def test_origin_source_timestamp_must_be_valid(source, timestamp):
    bundle = load_idsse_bundle("unused", MATCH)
    origins = bundle.event_origins.copy()
    origins["source_timestamp"] = origins["source_timestamp"].astype(object)
    origins.loc[0, "source_timestamp"] = timestamp
    with pytest.raises(ValueError, match="source timestamp"):
        validate_bundle(replace(bundle, event_origins=origins))


def test_zero_raw_events_have_deterministic_empty_origin_quality(source):
    source.events.clear()
    source.match_information.find("MatchInformation/General").set("Result", "0:0")
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.event_origins.empty and bundle.events.empty and bundle.goals.empty
    assert bundle.quality["source_events_total"] == 0
    assert bundle.quality["source_event_codes"] == {}
    assert bundle.quality["event_exclusion_reasons"] == {}
    assert bundle.quality["event_origin_count"] == 0


def test_metadata_match_identity_is_checked_before_canonical_loading(source):
    source.match_information.find("MatchInformation/General").set("MatchId", "other")
    with pytest.raises(ValueError, match="metadata match ID"):
        load_idsse_bundle("unused", MATCH)


def test_sparse_observed_frames_preserve_source_rate_and_never_fill_gap(source):
    source.position_clocks["firstHalf"] = source.position_clocks["firstHalf"].iloc[[0, 2]]
    events = source.events["firstHalf"]["Home"].events
    events.loc[1, "timestamp"] = pd.Timestamp("2025-04-05T12:00:00.100Z")
    events.loc[1, "gameclock"] = .1
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.fps == 20.
    assert bundle.quality["source_fps_by_period"] == {1: 20., 2: 20.}
    assert bundle.quality["frame_gaps"] == 1
    assert bundle.event_origins.query("period == 1").shape[0] == 4
    assert "frame" not in bundle.event_origins
    assert 101 not in bundle.tracking.query("period == 1").frame.tolist()


def test_unknown_attack_direction_is_left_unasserted(source):
    source.positions["firstHalf"]["Away"].x[:] = 20.
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.metadata["event_capabilities"]["attack_direction"] is False


def test_idsse_bundle_exposes_countable_origins_without_destinations_or_frames(source):
    bundle = load_idsse_bundle("unused", MATCH)
    assert bundle.events.empty
    assert bundle.event_origins.event.tolist() == ["pass", "shot", "pass", "pass"]
    assert not {"frame", "x1", "y1", "to_x", "to_y"} & set(bundle.event_origins.columns)
    first = bundle.event_origins.iloc[0]
    assert first.source_code == "Play_Pass"
    assert first.source_timestamp == pd.Timestamp("2025-04-05T12:00:00.015Z")
    assert first.gameclock_s == .015
    assert (first.source_x_m, first.source_y_m) == (80., 10.)
    assert (first.x_m, first.y_m) == (25., -25.)
    assert bundle.metadata["event_capabilities"] == {
        "origin_coordinates": True,
        "pass_endpoints": False,
        "attack_direction": False,
        "tracking_sync": False,
    }
    assert "event_alignment" not in bundle.quality
    assert bundle.quality["event_origin_count"] == 4


def test_origin_ids_follow_source_row_order_and_distinguish_rows(source):
    first = load_idsse_bundle("unused", MATCH).event_origins
    second = load_idsse_bundle("unused", MATCH).event_origins
    assert first.origin_id.tolist() == second.origin_id.tolist()
    assert first.origin_id.is_unique
    assert first.origin_id.str.match(r"^IDSSE:DFL-MAT-FIXTURE:[0-9]{6}$").all()
    assert len(set(first.origin_id)) == len(first)


def test_idsse_source_passes_cannot_enter_xt(source):
    from soccer_tactics.analytics.xt import fit_xt
    from soccer_tactics.analytics.report import analyze_match

    bundle = load_idsse_bundle("unused", MATCH)
    _, source_events = idsse.load_idsse_game("unused", MATCH)
    assert source_events.query("event == 'pass'")["x1"].isna().all()
    assert source_events.query("event == 'pass'")["y1"].isna().all()
    with pytest.raises(ValueError, match="pass endpoints"):
        fit_xt(source_events)
    with pytest.raises(ValueError, match="origin-only"):
        analyze_match(bundle, "home", [])
