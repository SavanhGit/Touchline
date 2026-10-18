"""Metrica Sports sample data (games 1-2, CSV format) -> canonical tracking schema.

Source files hold normalized coordinates (0-1, origin top-left, y down), one wide row per
frame with an x/y column pair per player. The ball is in both team files; we keep the home copy.
"""
from pathlib import Path

import pandas as pd

from soccer_tactics.schema import PITCH_LENGTH, PITCH_WIDTH, orient, validate


def _read_team(path: Path, team: str) -> pd.DataFrame:
    with open(path) as f:
        header = [next(f) for _ in range(3)][2].rstrip("\n").split(",")
    labels = [header[i] for i in range(3, len(header) - 1, 2)]  # x columns; y follows each
    cols = ["period", "frame", "time_s"] + [f"{l}__{a}" for l in labels for a in "xy"]
    raw = pd.read_csv(path, skiprows=3, header=None, names=cols)

    parts = []
    for label in labels:
        p = raw[["period", "frame", "time_s", f"{label}__x", f"{label}__y"]].copy()
        p.columns = ["period", "frame", "time_s", "x", "y"]
        is_ball = label == "Ball"
        p["team"] = "ball" if is_ball else team
        p["player_id"] = "ball" if is_ball else label.replace(" ", "")  # game 2 has "Player 26"
        parts.append(p.dropna(subset=["x", "y"]))
    return pd.concat(parts, ignore_index=True)


def _drop_handover_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """On a substitution frame both players carry coordinates; keep the incoming one so
    no team ever has more than 11 players in a frame."""
    last = df.groupby(["team", "player_id"])["frame"].transform("max")
    n = df.groupby(["team", "frame"])["player_id"].transform("size")
    overlap = (df["team"] != "ball") & (n > 11) & (df["frame"] == last) & (df["frame"] < df["frame"].max())
    return df[~overlap]


def load_metrica(
    root: str | Path,
    game: int = 1,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
) -> pd.DataFrame:
    """Load one sample game from a clone of metrica-sports/sample-data."""
    d = Path(root) / "data" / f"Sample_Game_{game}"
    home = _read_team(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "home")
    away = _read_team(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "away")
    df = pd.concat([home, away[away["team"] != "ball"]], ignore_index=True)
    df = _drop_handover_overlap(df)

    df["x"] = (df["x"] - 0.5) * length
    df["y"] = (0.5 - df["y"]) * width
    df["match_id"] = f"metrica_{game}"
    df = orient(validate(df))
    return df.sort_values(["period", "frame", "team", "player_id"], ignore_index=True)
