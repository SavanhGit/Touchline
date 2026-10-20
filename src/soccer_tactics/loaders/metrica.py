"""Metrica Sports sample data (games 1-2, CSV format) -> canonical tracking and event schemas.

Source files hold normalized coordinates (0-1, origin top-left, y down). Tracking is one wide
row per frame with an x/y column pair per player; the ball is in both team files and we keep
the home copy.
"""
from pathlib import Path

import pandas as pd

from soccer_tactics.schema import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    attack_flips,
    flip_periods,
    validate,
    validate_events,
)


def _read_team(path: Path, team: str) -> pd.DataFrame:
    with open(path) as f:
        metadata = [f.readline() for _ in range(3)]
    header = metadata[2].rstrip("\r\n").split(",")
    labels = [header[i] for i in range(3, len(header) - 1, 2)]  # x columns; y follows each
    if (not all(line.strip() for line in metadata)
            or header[:3] != ["Period", "Frame", "Time [s]"]
            or len(header) < 7 or (len(header) - 3) % 2
            or not all(labels) or len(set(labels)) != len(labels)
            or "Ball" not in labels or any(header[4::2])):
        raise ValueError(
            f"Invalid or incomplete Metrica tracking metadata/header in {path}. "
            "Expected two metadata rows and a Period,Frame,Time [s] header with player/ball x-y pairs. "
            "Download or restore the complete source CSV from Metrica Sports sample-data."
        )
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


def _load_tracking_raw(d: Path, game: int, length: float, width: float) -> pd.DataFrame:
    home = _read_team(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", "home")
    away = _read_team(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", "away")
    df = pd.concat([home, away[away["team"] != "ball"]], ignore_index=True)
    df = _drop_handover_overlap(df)
    df["x"] = (df["x"] - 0.5) * length
    df["y"] = (0.5 - df["y"]) * width
    df["match_id"] = f"metrica_{game}"
    return validate(df)


# BALL LOST rows that are not a failed pass: dispossessions (3-4 m), forced errors, stoppages.
_NOT_A_PASS = "THEFT|FORCED|HAND BALL|OFFSIDE|REFEREE|WOODWORK|END HALF"


def _load_events_raw(d: Path, game: int, length: float, width: float) -> pd.DataFrame:
    """Passes and shots, un-oriented. Metrica logs every PASS as completed (the receiver is a
    teammate); a failed pass is a later BALL LOST row from the intended receiver's side with
    the interception point as its end. Penalties are dropped: they would skew shot conversion."""
    path = d / f"Sample_Game_{game}_RawEventsData.csv"
    e = pd.read_csv(path)
    required = {"Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]",
                "From", "To", "Start X", "Start Y", "End X", "End Y"}
    missing = required - set(e.columns)
    if missing:
        raise ValueError(
            f"Invalid Metrica event header in {path}; missing columns: {', '.join(sorted(missing))}. "
            "Download or restore the complete source CSV from Metrica Sports sample-data."
        )
    sub = e["Subtype"].fillna("")
    after_penalty = (e["Type"].shift(1) == "SET PIECE") & (e["Subtype"].shift(1) == "PENALTY")
    done = e["Type"] == "PASS"
    lost = (e["Type"] == "BALL LOST") & e["End X"].notna() & ~sub.str.contains(_NOT_A_PASS)
    shot = (e["Type"] == "SHOT") & ~after_penalty

    e = e[done | lost | shot]
    out = pd.DataFrame({
        "match_id": f"metrica_{game}",
        "period": e["Period"],
        "team": e["Team"].str.lower(),
        "event": e["Type"].map(lambda t: "shot" if t == "SHOT" else "pass"),
        # a goal is a SHOT subtype ending in GOAL; "GOAL KICK" passes must not match
        "success": (e["Type"] == "PASS") | ((e["Type"] == "SHOT") & e["Subtype"].fillna("").str.contains("GOAL")),
        "player_id": e["From"].str.replace(" ", ""),
        "receiver_id": e["To"].where(e["Type"] == "PASS").str.replace(" ", ""),
        "frame": e["Start Frame"],
        "time_s": e["Start Time [s]"],
        "x0": (e["Start X"] - 0.5) * length,
        "y0": (0.5 - e["Start Y"]) * width,
        "x1": (e["End X"] - 0.5) * length,
        "y1": (0.5 - e["End Y"]) * width,
    })
    return validate_events(out).reset_index(drop=True)


def load_metrica_game(
    root: str | Path,
    game: int = 1,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tracking and events for one sample game from a clone of metrica-sports/sample-data.

    Both come back in the same orientation: the per-period flip is derived from the tracking
    (where the teams actually stand), then applied to the events too."""
    d = Path(root) / "data" / f"Sample_Game_{game}"
    tracking = _load_tracking_raw(d, game, length, width)
    events = _load_events_raw(d, game, length, width)
    flips = attack_flips(tracking)
    tracking = flip_periods(tracking, flips, ["x", "y"])
    events = flip_periods(events, flips, ["x0", "y0", "x1", "y1"])
    tracking = tracking.sort_values(["period", "frame", "team", "player_id"], ignore_index=True)
    return tracking, events


def load_metrica(root: str | Path, game: int = 1, **kwargs) -> pd.DataFrame:
    """Tracking only; see load_metrica_game."""
    return load_metrica_game(root, game, **kwargs)[0]
