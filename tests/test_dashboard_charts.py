"""Geometry and observation availability for dashboard pitch figures."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from soccer_tactics.analytics.report import Evidence
from soccer_tactics.dashboard.charts import evidence_pitch, pass_map, pitch, snapshot, tracking_window
from soccer_tactics.match import MatchBundle


def moment():
    return Evidence("e", "match", 1, 50, 2., "away", "a", "pass", -10., 4., -25., 8.)


def tracking(times, *, teams=("home", "away", "ball")):
    return pd.DataFrame([
        dict(match_id="match", period=1, frame=round(t * 25), time_s=t,
             team=team, player_id=team, x=t + i, y=float(i), vx=0., vy=0.)
        for t in times for i, team in enumerate(teams)
    ])


def bundle(rows):
    return MatchBundle("match", rows, pd.DataFrame(), pd.DataFrame(), "test", 25.)


def lines(fig):
    return [trace for trace in fig.data if trace.mode == "lines"]


def test_tracking_window_does_not_borrow_from_another_half_or_match():
    rows = tracking([-1.4001, -1.4, 0., 2., 5.4, 5.4001])
    rows = pd.concat([rows, rows.assign(period=2), rows.assign(match_id="other")])
    window = tracking_window(bundle(rows), moment())
    assert set(window.period) <= {1}
    assert set(window.match_id) == {"match"}
    assert set(window.time_s) == {-1.4, 0., 2., 5.4}
    assert tracking_window(bundle(rows.assign(period=2)), moment()).empty


@pytest.mark.parametrize("fps", [10., 25., 50.])
def test_snapshot_uses_bounded_nearest_observation(fps):
    rows = tracking([0.])
    tolerance = max(1.5 / fps, .06)
    assert not snapshot(rows, tolerance, fps).empty
    assert snapshot(rows, tolerance + .0001, fps).empty
    assert snapshot(rows, 5000, fps).empty
    assert snapshot(rows.iloc[:0], 0, fps).empty


def test_away_maps_negate_both_axes_without_changing_source():
    rows = pd.DataFrame([dict(x0=-10., y0=4., x1=-25., y1=8., turnover_cost=.1,
                              player_id="a", period=1, time_s=2.)])
    fig = pass_map(rows, "away", "Passes")
    assert tuple(fig.data[0].x) == (10., 25., None)
    assert tuple(fig.data[0].y) == (-4., -8., None)
    loss = pass_map(rows, "away", "Losses", losses=True)
    assert tuple(loss.data[0].x) == (25.,)
    assert tuple(loss.data[0].y) == (-8.,)
    assert rows.x0.iloc[0] == -10.


def test_pitch_geometry_and_source_arrow_are_fixed_across_slider_changes():
    fig = pitch()
    assert tuple(fig.layout.xaxis.range) == (-57, 57)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1
    arrows = []
    for time in [1., 2., 3.]:
        fig = evidence_pitch(tracking([time]), moment())
        arrow = fig.layout.annotations[-1]
        arrows.append((arrow.ax, arrow.ay, arrow.x, arrow.y))
        home = next(trace for trace in fig.data if trace.name == "Home")
        assert tuple(home.x) == (time,)
    assert arrows == [(-10., 4., -25., 8.)] * 3


@pytest.mark.parametrize("teams", [(), ("home",), ("away",)])
def test_empty_and_one_team_frames_have_no_control_heatmap(teams):
    rows = tracking([2.])
    rows = rows[rows.team.isin(teams)]
    fig = evidence_pitch(rows, moment(), control=True)
    assert not any(trace.type == "heatmap" for trace in fig.data)
    assert len(fig.layout.annotations) == 1  # Source arrow survives absent tracking.
    assert sum(len(trace.x) for trace in fig.data) == len(rows)


def test_nonfinite_positions_cannot_create_observations_or_control():
    rows = tracking([2.])
    rows.loc[rows.team == "away", "x"] = np.inf
    fig = evidence_pitch(rows, moment(), control=True)
    assert not any(trace.type == "heatmap" for trace in fig.data)
    for trace in fig.data:
        assert np.isfinite(np.asarray(trace.x, float)).all()


def test_control_depends_only_on_displayed_frame():
    frame = tracking([2.])
    fig = evidence_pitch(frame, moment(), control=True, trail=tracking([0., 1.]))
    heatmap = next(trace for trace in fig.data if trace.type == "heatmap")
    without_trail = evidence_pitch(frame, moment(), control=True)
    assert np.array_equal(heatmap.z, without_trail.data[0].z)
    assert np.isfinite(heatmap.z).all()


def test_trail_gap_breaks_before_downsampling_and_keeps_normal_paths_visible():
    times = np.r_[np.arange(0, .8, .04), np.arange(1.2, 2.04, .04)]
    trail = tracking(times, teams=("home",))
    fig = evidence_pitch(tracking([2.]), moment(), trail=trail)
    segments = []
    for trace in lines(fig):
        segment = []
        for x in trace.x:
            if x is None or not np.isfinite(x):
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append(x)
        if segment:
            segments.append(segment)
    assert len(segments) >= 2
    assert all(max(segment) < 1. or min(segment) > 1. for segment in segments)
    assert all(len(segment) > 1 for segment in segments)


def test_trails_are_limited_to_preceding_two_seconds_and_current_identity():
    trail = tracking(np.arange(0, 4.04, .04), teams=("home",))
    trail = pd.concat([trail, trail.assign(period=2, x=1000.),
                       trail.assign(match_id="other", x=2000.)])
    fig = evidence_pitch(tracking([3.]), moment(), trail=trail)
    values = [x for trace in lines(fig) for x in trace.x if x is not None]
    assert values
    assert all(1. <= x <= 3. for x in values)


def test_nonfinite_trail_observation_breaks_path():
    trail = tracking(np.arange(0, 2.04, .04), teams=("home",))
    trail.loc[trail.frame == 24, "x"] = np.nan
    fig = evidence_pitch(tracking([2.]), moment(), trail=trail)
    for trace in lines(fig):
        finite = [x for x in trace.x if x is not None and np.isfinite(x)]
        if finite and min(finite) < .96 < max(finite):
            assert any(x is None or not np.isfinite(x) for x in trace.x)
