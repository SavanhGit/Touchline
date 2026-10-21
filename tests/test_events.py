import numpy as np
import pytest

from conftest import GAMES
from soccer_tactics.schema import EVENT_COLUMNS

pytestmark = pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")


@pytest.fixture(params=GAMES)
def events(request, games):
    return games[request.param][1]


def test_schema(events):
    assert list(events.columns) == EVENT_COLUMNS
    assert {"pass", "shot", "tackle", "interception", "recovery"} == set(events["event"])


def test_failed_passes_are_present_and_rarer_than_completed(events):
    p = events[events["event"] == "pass"]
    assert 0.05 < (~p["success"]).mean() < 0.35
    assert p.loc[~p["success"], "receiver_id"].isna().all()
    assert p.loc[p["success"], "receiver_id"].notna().all()


def test_shots_and_goals_match_the_match_record_without_penalties(games):
    # raw SHOT rows: game 1 has 24 shots and 3 goals; game 2 has 24 shots and 5 goals, of which
    # one shot (a scored penalty) directly follows the PENALTY set piece and is dropped
    shots = {g: ev[ev["event"] == "shot"] for g, (_, ev) in games.items()}
    expected = {1: (24, 3), 2: (23, 4)}
    for g, s in shots.items():
        assert (len(s), int(s["success"].sum())) == expected[g]


def test_shots_start_in_the_half_the_shooter_attacks(events):
    """Independent check on orientation: events are flipped using flips derived from tracking,
    so home shots should sit in the +x half and away shots in the -x half in both periods."""
    shots = events[events["event"] == "shot"]
    home, away = shots[shots["team"] == "home"], shots[shots["team"] == "away"]
    assert (home["x0"] > 0).mean() >= 0.95
    assert (away["x0"] < 0).mean() >= 0.95


def test_events_line_up_with_tracking_frames(games):
    """The passer's tracked position at the event frame matches the event's start point
    (measured: median 0.34 m, p90 0.51 m, and worse in both directions when frames are shifted)."""
    for tracking, events in games.values():
        passes = events[events["event"] == "pass"]
        tp = tracking[tracking["team"] != "ball"][["frame", "team", "player_id", "x", "y"]]
        m = passes.merge(tp, on=["frame", "team", "player_id"], how="left", suffixes=("", "_t"))
        d = np.hypot(m["x0"] - m["x"], m["y0"] - m["y"])
        assert m["x"].notna().mean() > 0.99
        assert d.median() < 1.0 and d.quantile(0.9) < 2.0


def test_no_ids_with_spaces(events):
    assert not events["player_id"].str.contains(" ").any()
    assert not events["receiver_id"].dropna().str.contains(" ").any()
