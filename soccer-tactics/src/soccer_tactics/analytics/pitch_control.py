"""Time-to-intercept pitch control and threat-weighted zones.

For each cell, every player gets an arrival time: they keep their current velocity through a
reaction time, then run straight at top speed. A team's arrival time is its fastest player's,
and the home control probability is a logistic of the arrival-time gap (the form used in
Spearman 2016). This is the simplified version: it ignores ball flight time and acceleration,
so it describes who could *reach space first*, not who could receive a pass there.

Grids are indexed [x_cell, y_cell], matching xT; positions are the canonical metric pitch.
"""
import numpy as np
import pandas as pd

from soccer_tactics.analytics.xt import centers, xt_at

REACTION = 0.7  # s
VMAX = 5.0  # m/s, sustained running speed over the short distances that matter here
SIGMA = 0.45  # s, how sharply control flips with the arrival-time gap


def _arrival(team: pd.DataFrame, X, Y, reaction: float, vmax: float) -> np.ndarray:
    """Fastest arrival time of any player in `team` at each point; X, Y share any shape."""
    px = team["x"].to_numpy() + np.nan_to_num(team["vx"].to_numpy()) * reaction
    py = team["y"].to_numpy() + np.nan_to_num(team["vy"].to_numpy()) * reaction
    shape = (-1,) + (1,) * X.ndim
    dist = np.hypot(X[None] - px.reshape(shape), Y[None] - py.reshape(shape))
    return (reaction + dist / vmax).min(axis=0)


def control_at(
    frame: pd.DataFrame,
    x,
    y,
    reaction: float = REACTION,
    vmax: float = VMAX,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Home-team control probability at points (x, y) of any shape. `frame` is one frame of
    tracking with kinematics (x, y, vx, vy, team); the ball is ignored."""
    home, away = frame[frame["team"] == "home"], frame[frame["team"] == "away"]
    if home.empty or away.empty:
        raise ValueError("frame needs players from both teams")
    X, Y = np.asarray(x, float), np.asarray(y, float)
    gap = _arrival(away, X, Y, reaction, vmax) - _arrival(home, X, Y, reaction, vmax)
    return 0.5 * (1 + np.tanh(gap * np.pi / (np.sqrt(3) * sigma) / 2))  # stable logistic


def pitch_control(frame: pd.DataFrame, nx: int = 32, ny: int = 20, **params) -> np.ndarray:
    """Home-team control probability per cell, shape (nx, ny); see control_at for parameters."""
    return control_at(frame, *centers(nx, ny), **params)


def zone_threat(control: np.ndarray, xt_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(home_threat, away_threat) per cell: the controlled share of a cell times what that cell
    is worth to the team attacking it. Home attacks +x; away attacks -x, so its xT is mirrored."""
    X, Y = centers(*control.shape)
    return control * xt_at(xt_grid, X, Y), (1 - control) * xt_at(xt_grid, -X, -Y)
