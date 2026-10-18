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


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Flip each period where the home team defends the left side, so home always attacks +x."""
    df = df.copy()
    players = df[df["team"] != "ball"]
    for period, g in players.groupby("period"):
        mean_x = g.groupby("team")["x"].mean()
        if mean_x["home"] > mean_x["away"]:
            df.loc[df["period"] == period, ["x", "y"]] *= -1
    return df
