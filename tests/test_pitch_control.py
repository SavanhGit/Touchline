import numpy as np
import pandas as pd
import pytest

from conftest import GAMES
from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.pitch_control import control_at, pitch_control, zone_threat
from soccer_tactics.analytics.xt import centers


def _frame(home, away):
    """home/away: lists of (x, y, vx, vy)"""
    rows = [("home", *p) for p in home] + [("away", *p) for p in away]
    return pd.DataFrame(rows, columns=["team", "x", "y", "vx", "vy"])


def test_symmetric_setup_gives_mirrored_control():
    c = pitch_control(_frame([(-10, 0, 0, 0)], [(10, 0, 0, 0)]))
    assert np.allclose(c + c[::-1, ::-1], 1.0)
    assert c.min() >= 0 and c.max() <= 1


def test_each_team_controls_the_space_around_its_players():
    c = pitch_control(_frame([(-30, 0, 0, 0)], [(30, 0, 0, 0)]))
    assert c[0, 10] > 0.99 and c[-1, 10] < 0.01


def test_velocity_shifts_control_in_the_direction_of_motion():
    still = pitch_control(_frame([(-5, 0, 0, 0)], [(5, 0, 0, 0)]))
    running = pitch_control(_frame([(-5, 0, 6, 0)], [(5, 0, 0, 0)]))  # home runs toward away
    X, _ = centers(32, 20)
    assert (running - still)[(X > 0) & (X < 4)].mean() > 0.05


def test_more_players_never_reduce_a_teams_control():
    base = _frame([(-20, 0, 0, 0)], [(20, 0, 0, 0)])
    plus = _frame([(-20, 0, 0, 0), (0, 10, 0, 0)], [(20, 0, 0, 0)])
    assert (pitch_control(plus) >= pitch_control(base) - 1e-12).all()


def test_nan_velocity_is_treated_as_standing_still():
    f = _frame([(-10, 0, np.nan, np.nan)], [(10, 0, 0, 0)])
    assert np.isfinite(pitch_control(f)).all()


def test_control_at_agrees_with_the_grid_and_accepts_scalars():
    f = _frame([(-10, 3, 1, 0), (-25, -8, 0, 2)], [(12, 0, -1, 0), (30, 9, 0, 0)])
    X, Y = centers(32, 20)
    grid = pitch_control(f)
    assert np.allclose(control_at(f, X, Y), grid)
    assert np.isclose(control_at(f, X[5, 7], Y[5, 7]), grid[5, 7])  # scalar in, scalar out
    assert control_at(f, [0.0, 10.0], [0.0, 0.0]).shape == (2,)


def test_missing_team_raises():
    with pytest.raises(ValueError):
        pitch_control(_frame([(0, 0, 0, 0)], []))


def test_zone_threat_splits_by_who_controls_and_who_attacks_where():
    xt = np.zeros((16, 12))
    xt[-1, :] = 1.0  # only the last xT column is valuable (to a team attacking +x); it spans 2 control columns
    home, away = zone_threat(np.ones((32, 20)), xt)  # home controls everything
    assert home[-2:].min() == 1 and home[:-2].max() == 0 and away.max() == 0
    home, away = zone_threat(np.zeros((32, 20)), xt)  # away controls everything
    assert home.max() == 0 and away[:2].min() == 1 and away[2:].max() == 0  # away's goal end is -x


@pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")
def test_real_frame_is_sane(games):
    tracking = add_kinematics(games[GAMES[0]][0])
    frame = tracking[(tracking["period"] == 1) & (tracking["frame"] == 20000) & (tracking["team"] != "ball")]
    c = pitch_control(frame)
    assert np.isfinite(c).all() and c.min() >= 0 and c.max() <= 1
    assert 0.25 < c.mean() < 0.75  # neither team owns the whole pitch
