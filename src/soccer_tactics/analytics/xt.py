"""Expected threat (xT, Singh 2018) fitted from pass and shot events.

xT(cell) is the probability that possession in that cell ends in a goal, found by iterating

    xT = s * g + m * (T @ xT)

per cell: s/m = shot/move share of actions, g = goal probability of a shot, T = transition
probabilities of completed passes (failed passes stay in the denominator, so turnovers drain
value). Grids are indexed [x_cell, y_cell] with the acting team attacking +x.

Limits of this data: Metrica events contain no carries, so progress by dribbling is invisible,
and two games hold ~48 shots, too few to estimate conversion per zone. Shot value is therefore
a one-parameter geometric model (angle the goal subtends), calibrated so predicted goals equal
observed goals. Fit on more matches and swap in a proper xG model by passing `goal_prob`.
"""
import numpy as np
import pandas as pd

from soccer_tactics.schema import PITCH_LENGTH as L
from soccer_tactics.schema import PITCH_WIDTH as W
from soccer_tactics.schema import validate_events

NX, NY = 16, 12
GOAL_WIDTH = 7.32
_XY = ["x0", "y0", "x1", "y1"]


def attacking(events: pd.DataFrame) -> pd.DataFrame:
    """Copy with coordinates rotated so each event's team attacks +x (away team negated)."""
    e = events.copy()
    e.loc[e["team"] == "away", _XY] *= -1
    return e


def cell(x, y, nx: int = NX, ny: int = NY) -> tuple[np.ndarray, np.ndarray]:
    xi = np.clip(((np.asarray(x) + L / 2) / L * nx).astype(int), 0, nx - 1)
    yi = np.clip(((np.asarray(y) + W / 2) / W * ny).astype(int), 0, ny - 1)
    return xi, yi


def centers(nx: int = NX, ny: int = NY) -> tuple[np.ndarray, np.ndarray]:
    """Cell-center coordinates as (X, Y) arrays of shape (nx, ny)."""
    xs = (np.arange(nx) + 0.5) / nx * L - L / 2
    ys = (np.arange(ny) + 0.5) / ny * W - W / 2
    return np.meshgrid(xs, ys, indexing="ij")


def goal_angle(x, y) -> np.ndarray:
    """Angle (radians) the goal mouth subtends from (x, y), attacking +x."""
    dx, dy = L / 2 - np.asarray(x, float), np.asarray(y, float)
    return np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)


def _goal_prob(shots: pd.DataFrame, nx: int, ny: int) -> np.ndarray:
    """g = 1 - exp(-k * angle), with k set so the shots' predicted goals match actual goals."""
    if shots.empty:
        raise ValueError("need at least one shot to calibrate shot value")
    ang, goals = goal_angle(shots["x0"], shots["y0"]), shots["success"].sum()
    lo, hi = 0.0, 100.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if (1 - np.exp(-k * ang)).sum() < goals else (lo, k)
    return (1 - np.exp(-k * goal_angle(*centers(nx, ny)))).ravel()


def fit_xt(
    events: pd.DataFrame,
    nx: int = NX,
    ny: int = NY,
    goal_prob: np.ndarray | None = None,
    smooth: int = 1,
) -> np.ndarray:
    """Fit an (nx, ny) xT grid. `goal_prob` is an optional flat (nx*ny,) goal probability per cell.

    `smooth` blurs the fitted surface that many passes (0 = raw). On the two Metrica games, 46 of
    192 cells have fewer than 5 passes, mostly in the final third, which leaves holes and spikes.
    Surfaces fitted on each game alone agree at Spearman 0.65 raw and 0.96 with one pass."""
    e = attacking(validate_events(events))
    mv, sh = e[e["event"] == "pass"], e[e["event"] == "shot"]
    n = nx * ny
    flat = lambda d, a, b: (lambda xi, yi: xi * ny + yi)(*cell(d[a], d[b], nx, ny))

    start, end, ok = flat(mv, "x0", "y0"), flat(mv, "x1", "y1"), mv["success"].to_numpy(bool)
    n_move = np.bincount(start, minlength=n).astype(float)
    n_shot = np.bincount(flat(sh, "x0", "y0"), minlength=n).astype(float)
    T = np.zeros((n, n))
    np.add.at(T, (start[ok], end[ok]), 1.0)
    T /= np.maximum(n_move, 1)[:, None]

    total = n_move + n_shot
    s = np.divide(n_shot, total, out=np.zeros(n), where=total > 0)
    m = np.divide(n_move, total, out=np.zeros(n), where=total > 0)
    g = _goal_prob(sh, nx, ny) if goal_prob is None else np.asarray(goal_prob, float)

    xt = np.zeros(n)
    for _ in range(500):
        new = s * g + m * (T @ xt)
        done = np.abs(new - xt).max() < 1e-10
        xt = new
        if done:
            break
    return smooth_grid(xt.reshape(nx, ny), smooth)


def smooth_grid(grid: np.ndarray, passes: int = 1) -> np.ndarray:
    """Blur a grid with a [1, 2, 1] / 4 kernel along both axes (edges renormalized, so values
    stay probabilities). Fills the holes that sparse zones leave in a fitted surface."""
    out = np.asarray(grid, float)
    for _ in range(passes):
        for ax in (0, 1):
            a = np.moveaxis(out, ax, 0)
            num, den = 2 * a, np.full_like(a, 2.0)
            num[1:] += a[:-1]
            den[1:] += 1
            num[:-1] += a[1:]
            den[:-1] += 1
            out = np.moveaxis(num / den, 0, ax)
    return out


def xt_at(grid: np.ndarray, x, y) -> np.ndarray:
    """Look up xT at coordinates (attacking +x); out-of-pitch points clamp to the edge cell."""
    return grid[cell(x, y, *grid.shape)]


def add_xt(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Add xt0, xt1 and xt_gain to pass rows (NaN for shots).

    A completed pass gains xT(end) - xT(start). A failed pass loses xT(start): the value of the
    possession that was thrown away."""
    e = events.copy()
    a = attacking(e)
    mv = (e["event"] == "pass").to_numpy()
    for col in ("xt0", "xt1", "xt_gain"):
        e[col] = np.nan
    e.loc[mv, "xt0"] = xt_at(grid, a.loc[mv, "x0"], a.loc[mv, "y0"])
    e.loc[mv, "xt1"] = xt_at(grid, a.loc[mv, "x1"], a.loc[mv, "y1"])
    ok = e["success"].to_numpy(bool)
    e["xt_gain"] = np.where(ok, e["xt1"] - e["xt0"], -e["xt0"])
    e.loc[~mv, "xt_gain"] = np.nan
    return e
