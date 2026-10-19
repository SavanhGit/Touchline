"""Per-object kinematics from the canonical tracking frame."""
import numpy as np
import pandas as pd

from soccer_tactics.schema import FPS

KEYS = ["match_id", "period", "team", "player_id"]


def add_kinematics(
    df: pd.DataFrame,
    fps: int = FPS,
    window: int = 5,
    max_speed: float = 12.0,
    ball_max_speed: float = 40.0,
) -> pd.DataFrame:
    """Add vx, vy, speed (m/s) and accel (m/s^2) from a centered rolling-mean smoothing of x, y.

    Open tracking data has teleport glitches (a player jumping 50 m in one frame). Frames whose
    speed exceeds the physical cap, and the frames around them, get NaN kinematics instead of
    poisoning downstream stats.
    """
    df = df.sort_values(KEYS + ["frame"]).copy()
    for c in ("x", "y"):
        df[f"{c}_s"] = df.groupby(KEYS)[c].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).mean()
        )
    g = df.groupby(KEYS)
    dt = g["time_s"].diff()  # source clock handles different sampling rates and dropped frames
    dt = dt.where(dt > 0)
    df["vx"] = g["x_s"].diff() / dt
    df["vy"] = g["y_s"].diff() / dt
    df["speed"] = np.hypot(df["vx"], df["vy"])
    over = (df["speed"] > np.where(df["team"] == "ball", ball_max_speed, max_speed)).astype(float)
    # a glitch also contaminates the smoothing window around it, so mask its neighbours too
    bad = over.groupby([df[k] for k in KEYS]).transform(
        lambda s: s.rolling(2 * window + 1, center=True, min_periods=1).max()
    ) > 0
    df.loc[bad, ["vx", "vy", "speed"]] = np.nan
    df["accel"] = df.groupby(KEYS)["speed"].diff() / dt
    return df.drop(columns=["x_s", "y_s"])
