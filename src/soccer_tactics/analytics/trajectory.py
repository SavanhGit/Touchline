"""Transparent trajectory forecasting baselines for canonical tracking data."""
import numpy as np
import pandas as pd

from soccer_tactics.analytics.features import add_kinematics


def constant_velocity_forecast(
    tracking: pd.DataFrame,
    horizon_s: float = 1.0,
    *,
    players: list[str] | None = None,
) -> pd.DataFrame:
    """Forecast each player's position by constant velocity.

    The output is a baseline for later learned models, not a claim of
    long-horizon physical realism.
    """
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    frame = add_kinematics(tracking)
    frame = frame[frame["team"] != "ball"].copy()
    if players is not None:
        frame = frame[frame["player_id"].isin(players)]
    frame = frame.sort_values(["match_id", "period", "team", "player_id", "frame"])
    groups = frame.groupby(["match_id", "period", "team", "player_id"])
    frame["raw_vx"] = groups["x"].diff() / groups["time_s"].diff()
    frame["raw_vy"] = groups["y"].diff() / groups["time_s"].diff()
    latest = frame.sort_values("frame").groupby(
        ["match_id", "period", "team", "player_id"], as_index=False
    ).tail(1)
    latest["x_pred"] = latest["x"] + latest["raw_vx"].fillna(latest["vx"]).fillna(0.0) * horizon_s
    latest["y_pred"] = latest["y"] + latest["raw_vy"].fillna(latest["vy"]).fillna(0.0) * horizon_s
    latest["horizon_s"] = horizon_s
    return latest[
        ["match_id", "period", "frame", "team", "player_id", "horizon_s", "x_pred", "y_pred"]
    ].reset_index(drop=True)


def forecast_error(predictions: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    """Return mean and final displacement error for matching forecast rows."""
    keys = ["match_id", "period", "frame", "team", "player_id"]
    merged = predictions.merge(actual, on=keys, suffixes=("_pred", "_actual"))
    if merged.empty:
        raise ValueError("no matching forecast and actual rows")
    error = np.hypot(merged["x_pred"] - merged["x"], merged["y_pred"] - merged["y"])
    return {"ade": float(error.mean()), "fde": float(error.iloc[-1])}
