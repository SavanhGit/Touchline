"""Source clocks, observed minutes and object identity survive reused frame/player IDs."""
import numpy as np
import pandas as pd
import pytest

from factories import make_bundle
from soccer_tactics.agent import tools
from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.impact import add_pressure, pass_values, player_impact
from soccer_tactics.schema import COLUMNS

GRID = np.array([[0.1], [0.4]])


def _runner(fps=25, *, match_id="m", period=1, speed=2.0, offset=0.0):
    return pd.DataFrame(
        [(match_id, period, frame, frame / fps, "home", "h", offset + speed * frame / fps, 0.0)
         for frame in range(100)], columns=COLUMNS,
    )


@pytest.mark.parametrize("fps", [10, 25, 30])
def test_sample_rate_minutes_and_cutoff(fps):
    bundle = make_bundle(fps=fps)
    table = player_impact(pass_values(bundle.events, GRID), bundle.tracking)
    assert set(table.player_id) == {"h", "g", "a"}
    assert np.allclose(table.minutes, 15.0)


@pytest.mark.parametrize("fps", [10, 25, 30])
def test_dropped_samples_reduce_observed_minutes(fps):
    bundle = make_bundle(fps=fps)
    tracking = bundle.tracking
    tracking = tracking[~((tracking.player_id == "h") & tracking.time_s.between(100, 110, inclusive="left"))]
    passes = pass_values(bundle.events, GRID)
    table = player_impact(passes, tracking, min_minutes=0)
    assert np.isclose(table.set_index("player_id").loc["h", "minutes"], 15 - 10 / 60)
    assert set(player_impact(passes, tracking).player_id) == {"g", "a"}


@pytest.mark.parametrize("fps", [10, 25, 30])
def test_source_clock_sets_runner_speed_even_with_dropped_samples(fps):
    tracking = _runner(fps)
    tracking = tracking[~tracking.frame.between(40, 49)]
    # window=1 isolates the derivative from smoothing across a sampling gap.
    out = add_kinematics(tracking, fps=1, window=1)
    assert np.allclose(out.speed.iloc[1:], 2.0)
    smoothed = add_kinematics(_runner(fps))
    assert np.allclose(smoothed.speed.iloc[10:-10], 2.0)


@pytest.mark.parametrize("delta", [0.0, -0.04])
def test_nonpositive_source_intervals_have_unavailable_motion(delta):
    tracking = _runner().iloc[:3].copy()
    tracking.loc[1, "time_s"] = tracking.loc[0, "time_s"] + delta
    out = add_kinematics(tracking, window=1)
    assert out.loc[1, ["vx", "vy", "speed", "accel"]].isna().all()
    assert not np.isinf(out[["vx", "vy", "speed", "accel"]].to_numpy()).any()


@pytest.mark.parametrize("identity", ["match_id", "period"])
def test_kinematics_does_not_mix_reused_player_or_frame_ids(identity):
    first = _runner()
    second = _runner(speed=6.0, offset=20.0)
    second[identity] = "other" if identity == "match_id" else 2
    out = add_kinematics(pd.concat([second, first], ignore_index=True))
    groups = list(out.groupby(identity))
    assert len(groups) == 2
    for key, group in groups:
        expected = 2.0 if key == first[identity].iloc[0] else 6.0
        assert group.speed.iloc[0:1].isna().all()
        assert np.allclose(group.speed.iloc[10:-10], expected)


@pytest.mark.parametrize("identity", ["match_id", "period"])
def test_pressure_joins_the_same_match_and_period(identity):
    bundle = make_bundle()
    event = bundle.events.iloc[:1].copy()
    event[["x0", "y0"]] = 0.0
    correct = bundle.tracking.iloc[:1].copy()
    correct[["team", "player_id", "x", "y"]] = ["away", "a", 3.0, 4.0]
    other = correct.copy()
    other[identity] = "other" if identity == "match_id" else 2
    other[["x", "y"]] = 0.0
    out = add_pressure(event, pd.concat([other, correct], ignore_index=True))
    assert out.pressure_m.iloc[0] == 5.0


def test_impact_infers_each_match_rate_with_shared_player_ids():
    first, second = make_bundle("ten", fps=10), make_bundle("thirty", fps=30)
    events = pd.concat([first.events, second.events], ignore_index=True)
    # Match thirty has no successful passes; another match's values must not leak in.
    events.loc[events.match_id == "thirty", "success"] = False
    table = player_impact(pass_values(events, GRID), pd.concat([first.tracking, second.tracking]))
    assert len(table) == 6
    assert np.allclose(table.minutes, 15.0)
    players = table.set_index(["match_id", "player_id"])
    assert players.loc[("ten", "h"), "completed"] == 4
    assert players.loc[("thirty", "h"), "completed"] == 0


def test_explicit_fps_overrides_inference():
    bundle = make_bundle(fps=10)
    table = player_impact(pass_values(bundle.events, GRID), bundle.tracking, min_minutes=0, fps=20)
    assert np.allclose(table.minutes, 7.5)


@pytest.mark.parametrize("invalid_fps", [0, -1, np.nan, np.inf])
def test_invalid_explicit_fps(invalid_fps):
    bundle = make_bundle()
    passes = pass_values(bundle.events, GRID)
    with pytest.raises(ValueError, match="fps"):
        player_impact(passes, bundle.tracking, fps=invalid_fps)


def test_top_speeds_keeps_matches_with_shared_player_ids_separate(monkeypatch):
    tracking = pd.concat([_runner(match_id="one"), _runner(match_id="two", speed=6.0)])
    monkeypatch.setitem(tools.STATE, "tracking", tracking)
    results = tools.top_speeds(n=2)
    assert len(results) == 2
    speeds = {row["match_id"]: row["peak_speed"] for row in results}
    assert np.isclose(speeds["one"], 2.0)
    assert np.isclose(speeds["two"], 6.0)
