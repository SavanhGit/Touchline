"""Canonical tracking format. Every source (open datasets, the CV pipeline) converts into this.

One row per object per frame (long format). Coordinates are meters, origin at pitch
center, x along the length, y along the width. After `orient`, the home team attacks
toward +x in every period.
"""
import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FPS = 25

COLUMNS = ["match_id", "period", "frame", "time_s", "team", "player_id", "x", "y"]
TEAMS = ("home", "away", "ball")


def _require_finite(df: pd.DataFrame, columns: list[str], label: str) -> None:
    if df.empty:
        return
    try:
        finite = np.isfinite(df[columns].to_numpy()).all()
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{label} must be finite numeric values: {columns}")


def _validate_identity(df: pd.DataFrame) -> None:
    if df[["match_id", "team", "player_id"]].isna().any().any():
        raise ValueError("match_id, team and player_id cannot be missing")
    _require_finite(df, ["period", "frame", "time_s"], "source clocks")
    if (df[["period", "frame"]] % 1 != 0).any().any():
        raise ValueError("period and frame must be integers")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    bad = set(df["team"].unique()) - set(TEAMS)
    if bad:
        raise ValueError(f"unknown team labels: {bad}")
    _validate_identity(df)
    _require_finite(df, ["x", "y"], "tracking positions")
    return df[COLUMNS]


def attack_flips(df: pd.DataFrame) -> dict:
    """Per period: True if the home team attacks toward -x there, so the period needs flipping."""
    players = df[df["team"] != "ball"]
    keys = ["match_id", "period", "team"] if df.match_id.nunique() > 1 else ["period", "team"]
    mean_x = players.groupby(keys)["x"].mean().unstack()
    return (mean_x["home"] > mean_x["away"]).to_dict()


def flip_periods(df: pd.DataFrame, flips: dict, cols: list[str]) -> pd.DataFrame:
    """Rotate the pitch 180 degrees (negate x and y) in the flagged periods."""
    df = df.copy()
    if flips and isinstance(next(iter(flips)), tuple):
        mask = pd.Series(list(zip(df.match_id, df.period)), index=df.index).map(flips)
    else:
        mask = df["period"].map(flips)
    df.loc[mask.fillna(False).astype(bool), cols] *= -1
    return df


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Flip each period where home attacks toward -x, so home always attacks +x."""
    return flip_periods(df, attack_flips(df), ["x", "y"])


# Canonical endpoint events used by Metrica analytics. IDSSE source origin rows
# stay outside this table because their endpoints and attack direction are unknown.
# `success` means a completed pass, or a goal.
EVENT_COLUMNS = [
    "match_id", "period", "team", "event", "success", "player_id", "receiver_id",
    "frame", "time_s", "x0", "y0", "x1", "y1",
]
EVENTS = ("pass", "shot", "tackle", "interception", "recovery")


def validate_events(df: pd.DataFrame, *, require_pass_endpoints: bool = True) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    bad = set(df["event"].unique()) - set(EVENTS)
    if bad:
        raise ValueError(f"unknown event labels: {bad}")
    bad = set(df["team"].unique()) - {"home", "away"}
    if bad:
        raise ValueError(f"unknown event team labels: {bad}")
    _validate_identity(df)
    _require_finite(df, ["x0", "y0"], "event origins")
    if require_pass_endpoints:
        _require_finite(df.loc[df["event"] == "pass"], ["x1", "y1"], "pass endpoints")
    return df[EVENT_COLUMNS]
