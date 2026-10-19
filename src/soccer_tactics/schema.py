"""Canonical tracking format. Every source (open datasets, the CV pipeline) converts into this.

One row per object per frame (long format). Coordinates are meters, origin at pitch
center, x along the length, y along the width. After `orient`, the home team attacks
toward +x in every period.
"""
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FPS = 25

COLUMNS = ["match_id", "period", "frame", "time_s", "team", "player_id", "x", "y"]
TEAMS = ("home", "away", "ball")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    bad = set(df["team"].unique()) - set(TEAMS)
    if bad:
        raise ValueError(f"unknown team labels: {bad}")
    return df[COLUMNS]


def attack_flips(df: pd.DataFrame) -> dict:
    """Per period: True if the home team attacks toward -x there, so the period needs flipping."""
    players = df[df["team"] != "ball"]
    mean_x = players.groupby(["period", "team"])["x"].mean().unstack()
    return (mean_x["home"] > mean_x["away"]).to_dict()


def flip_periods(df: pd.DataFrame, flips: dict, cols: list[str]) -> pd.DataFrame:
    """Rotate the pitch 180 degrees (negate x and y) in the flagged periods."""
    df = df.copy()
    df.loc[df["period"].map(flips).fillna(False).astype(bool), cols] *= -1
    return df


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Flip each period where home attacks toward -x, so home always attacks +x."""
    return flip_periods(df, attack_flips(df), ["x", "y"])


# Events: one row per pass or shot. Same units and orientation as tracking, so after orient the
# home team attacks +x and the away team attacks -x. `success` means a completed pass, or a goal.
EVENT_COLUMNS = [
    "match_id", "period", "team", "event", "success", "player_id", "receiver_id",
    "frame", "time_s", "x0", "y0", "x1", "y1",
]
EVENTS = ("pass", "shot", "tackle", "interception", "recovery")


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    bad = set(df["event"].unique()) - set(EVENTS)
    if bad:
        raise ValueError(f"unknown event labels: {bad}")
    if df.loc[df["event"] == "pass", ["x0", "y0", "x1", "y1"]].isna().any().any():
        raise ValueError("passes need start and end coordinates")
    return df[EVENT_COLUMNS]
