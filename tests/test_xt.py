import numpy as np
import pandas as pd
import pytest

from conftest import GAMES
from soccer_tactics.analytics.xt import (
    _goal_prob,
    add_xt,
    cell,
    centers,
    fit_xt,
    goal_angle,
    smooth_grid,
    xt_at,
)
from soccer_tactics.schema import EVENT_COLUMNS


def _ev(rows):
    """rows: (event, team, success, x0, y0, x1, y1)"""
    df = pd.DataFrame(rows, columns=["event", "team", "success", "x0", "y0", "x1", "y1"])
    for c in EVENT_COLUMNS:
        if c not in df:
            df[c] = 1 if c in ("period", "frame") else (0.0 if c == "time_s" else "x")
    return df[EVENT_COLUMNS]


def test_cell_lookup_corners_and_clamping():
    assert tuple(map(int, cell(-52.4, -33.9, 16, 12))) == (0, 0)
    assert tuple(map(int, cell(52.4, 33.9, 16, 12))) == (15, 11)
    assert tuple(map(int, cell(80, -50, 16, 12))) == (15, 0)  # clamps off-pitch points


def test_goal_angle_is_largest_in_front_of_goal_and_shrinks_with_distance():
    assert goal_angle(40, 0) > goal_angle(40, 20)
    assert goal_angle(40, 0) > goal_angle(20, 0) > goal_angle(0, 0)


def test_failed_passes_stay_in_the_denominator():
    """Two-cell pitch (nx=2, ny=1). From the own half: 4 passes, 2 complete into the attacking
    half, 2 fail. In the attacking half: 2 shots, no passes, goal probability 0.5.
    xT(attack) = 1 * 0.5 = 0.5 and xT(own) = (2/4) * 0.5 = 0.25 (not 0.5, which would ignore failures)."""
    rows = [("pass", "home", True, -30, 0, 30, 0)] * 2 + [("pass", "home", False, -30, 0, -10, 0)] * 2
    rows += [("shot", "home", False, 30, 0, 52, 0)] * 2
    grid = fit_xt(_ev(rows), nx=2, ny=1, goal_prob=np.array([0.0, 0.5]), smooth=0)
    assert grid.shape == (2, 1)
    assert np.allclose(grid[:, 0], [0.25, 0.5])


def test_away_events_are_mirrored_so_both_teams_share_one_grid():
    home = [("pass", "home", True, -30, 0, 30, 0)] * 2 + [("shot", "home", False, 30, 0, 52, 0)]
    away = [(e, "away", s, -x0, -y0, -x1, -y1) for e, _, s, x0, y0, x1, y1 in home]
    g = np.array([0.0, 0.5])
    fit = lambda ev: fit_xt(_ev(ev), 2, 1, g, smooth=0)
    assert np.allclose(fit(home), fit(away))
    assert np.allclose(fit(home + away), fit(home))


def test_calibrated_shot_value_reproduces_observed_goals():
    """Put every shot at a cell center, so the per-cell goal probability the fit returns can be
    summed over the shots directly: it must equal the number of goals."""
    X, Y = centers(16, 12)
    picks = [(15, 5), (14, 6), (13, 4), (13, 7), (12, 5), (11, 6), (14, 3), (12, 8)]
    shots = pd.DataFrame({"x0": [X[i, j] for i, j in picks], "y0": [Y[i, j] for i, j in picks]})
    shots["success"] = [True, True, False, False, False, False, False, False]
    g = _goal_prob(shots, 16, 12).reshape(16, 12)
    assert abs(sum(g[i, j] for i, j in picks) - 2) < 1e-6
    assert g.min() >= 0 and g.max() < 1
    assert g[15, 5] > g[12, 5] > g[8, 5]  # closer and more central is likelier


def test_smooth_grid_fills_holes_without_leaving_the_value_range():
    grid = np.zeros((6, 5))
    grid[3, 2] = 1.0
    out = smooth_grid(grid)
    assert out[3, 2] < 1.0 and out[2, 2] > 0 and out[3, 3] > 0  # spike spreads to neighbours
    assert out.min() >= 0 and out.max() <= 1.0
    assert np.allclose(smooth_grid(np.full((6, 5), 0.3), passes=3), 0.3)  # flat stays flat (edges too)
    assert smooth_grid(np.ones((2, 1))).shape == (2, 1)  # degenerate axes are fine


def test_add_xt_values_completed_and_failed_passes():
    grid = np.zeros((2, 1))
    grid[:, 0] = [0.25, 0.5]
    ev = _ev([
        ("pass", "home", True, -30, 0, 30, 0),   # gain 0.5 - 0.25
        ("pass", "home", False, -30, 0, -10, 0),  # loses the 0.25 it started with
        ("pass", "away", True, 30, 0, -30, 0),   # away attacks -x, so this is also own -> attack
        ("shot", "home", False, 30, 0, 52, 0),
    ])
    out = add_xt(ev, grid)
    assert np.allclose(out["xt_gain"].iloc[:3], [0.25, -0.25, 0.25])
    assert np.isnan(out["xt_gain"].iloc[3])


def test_xt_at_matches_cell_lookup():
    grid = np.arange(12.0).reshape(3, 4)
    assert xt_at(grid, 52, 33) == grid[2, 3]


@pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")
def test_real_surface_rises_toward_goal_and_is_a_probability(games):
    ev = pd.concat([e for _, e in games.values()], ignore_index=True)
    grid = fit_xt(ev)
    assert grid.shape == (16, 12)
    assert grid.min() >= 0 and grid.max() < 1
    by_x = grid.mean(axis=1)
    # xT is convex in x, so check monotonic rank order rather than a linear fit
    assert pd.Series(by_x).corr(pd.Series(np.arange(16.0)), method="spearman") > 0.95
    assert by_x[-4:].mean() > 3 * by_x[:4].mean()
    assert grid[-3:, 4:8].mean() > grid[-3:, :2].mean()  # central beats the touchline near goal
    assert (grid > 0).all()  # smoothing leaves no zero-value holes


@pytest.mark.skipif(len(GAMES) < 2, reason="needs both Metrica sample games")
def test_surfaces_fitted_on_separate_games_agree(games):
    """Split-half check: with ~24 shots per game the raw surfaces barely agree (0.65); one smoothing
    pass is what makes them usable (0.96). Shuffled cells give ~0.11 at the 95th percentile."""
    a, b = (fit_xt(games[g][1]) for g in (1, 2))
    assert pd.Series(a.ravel()).corr(pd.Series(b.ravel()), method="spearman") > 0.9
