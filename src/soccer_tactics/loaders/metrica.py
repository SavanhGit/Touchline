"""Local Metrica sample data (games 1-2 CSV; Game 3 EPTS/JSON) -> canonical schemas.

Source files hold normalized coordinates (0-1, origin top-left, y down). Tracking is one wide
row per frame with an x/y column pair per player; the ball is in both team files and we keep
the home copy.
"""
from pathlib import Path

import numpy as np
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
    """Passes, shots and explicitly recognized defensive events, un-oriented. Metrica logs every PASS as completed (the receiver is a
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
    return _canonical_events(e, game, length, width)


def _canonical_events(e: pd.DataFrame, game: int, length: float, width: float) -> pd.DataFrame:
    sub = e["Subtype"].fillna("").astype("string")
    after_penalty = (e["Type"].shift(1) == "SET PIECE") & (e["Subtype"].shift(1) == "PENALTY")
    done = e["Type"] == "PASS"
    lost = (e["Type"] == "BALL LOST") & e["End X"].notna() & ~sub.str.contains(_NOT_A_PASS)
    shot = (e["Type"] == "SHOT") & ~after_penalty

    mapping = {("CHALLENGE", "TACKLE-WON"): ("tackle", True),
               ("CHALLENGE", "TACKLE-LOST"): ("tackle", False),
               ("RECOVERY", "INTERCEPTION"): ("interception", True),
               ("RECOVERY", "THEFT"): ("recovery", True)}
    defensive = pd.Series([mapping.get(pair) for pair in zip(e["Type"], sub)], index=e.index)
    excluded = e["Type"].isin(["CHALLENGE", "RECOVERY"]) & defensive.isna()
    reasons = (e.loc[excluded, "Type"] + "/" + sub[excluded].replace("", "<blank>"))
    counts = {key: int(value) for key, value in reasons.value_counts().items()}
    e = e[done | lost | shot | defensive.notna()]
    defensive = defensive.loc[e.index]
    out = pd.DataFrame({
        "match_id": f"metrica_{game}",
        "period": e["Period"],
        "team": e["Team"].str.lower(),
        "event": [item[0] if item else ("shot" if kind == "SHOT" else "pass")
                  for kind, item in zip(e["Type"], defensive)],
        # a goal is a SHOT subtype ending in GOAL; "GOAL KICK" passes must not match
        "success": [item[1] if item else bool(kind == "PASS" or (kind == "SHOT" and "GOAL" in subtype))
                    for kind, subtype, item in zip(e["Type"], sub.loc[e.index], defensive)],
        "player_id": e["From"].astype("string").str.replace(" ", ""),
        "receiver_id": e["To"].where(e["Type"] == "PASS").astype("string").str.replace(" ", ""),
        "frame": e["Start Frame"],
        "time_s": e["Start Time [s]"],
        "x0": (e["Start X"] - 0.5) * length,
        "y0": (0.5 - e["Start Y"]) * width,
        "x1": (e["End X"] - 0.5) * length,
        "y1": (0.5 - e["End Y"]) * width,
    })
    out = validate_events(out).reset_index(drop=True)
    out.attrs["defensive_exclusions"] = counts
    return out


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
    if game == 3:
        tracking, raw, metadata = _load_epts_raw(d, length, width)
        events = _canonical_events(raw, game, length, width)
        tracking.attrs.update(metadata)
    else:
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



def _load_epts_raw(d: Path, length: float, width: float):
    """Read only local Game 3 feeds; Kloppy is optional for the CSV games."""
    try:
        from kloppy import metrica
    except ImportError as exc:
        raise ImportError("Game 3 needs Kloppy. Install it with: pip install -e '.[metrica-epts]'") from exc
    paths = [d / f"Sample_Game_3_{suffix}" for suffix in ("metadata.xml", "tracking.txt", "events.json")]
    for path in paths:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Complete local Metrica Game 3 source required: {path}")
    # Open binary streams explicitly: no URL or remote filesystem is handed to Kloppy.
    with paths[0].open("rb") as meta, paths[1].open("rb") as positions:
        dataset = metrica.load_tracking_epts(meta, positions, coordinates="metrica")
    rows = []
    for frame in dataset.records:
        time = (frame.period.start_timestamp + frame.timestamp).total_seconds()
        prefix = ("metrica_3", frame.period.id, frame.frame_id, time)
        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is not None and np.isfinite([point.x, point.y]).all():
                rows.append((*prefix, player.team.ground.value, player.player_id,
                             (point.x - .5) * length, (.5 - point.y) * width))
        point = frame.ball_coordinates
        if point is not None and np.isfinite([point.x, point.y]).all():
            rows.append((*prefix, "ball", "ball", (point.x - .5) * length, (.5 - point.y) * width))
    from soccer_tactics.schema import COLUMNS
    tracking = validate(pd.DataFrame(rows, columns=COLUMNS))
    score = dataset.metadata.score
    del rows, dataset
    with paths[0].open("rb") as meta, paths[2].open("rb") as source:
        event_dataset = metrica.load_event(source, meta, coordinates="metrica")
    # Kloppy may emit synthetic ball-out records sharing a raw event. Retain each source once.
    raw_events = {event.raw_event["index"]: event.raw_event for event in event_dataset.records}
    teams = {team.team_id: team.ground.value for team in event_dataset.metadata.teams}
    # Set pieces are folded into Kloppy passes/shots. Preserve them for penalty exclusion.
    import json
    with paths[2].open(encoding="utf-8") as stream:
        for event in json.load(stream)["data"]:
            if event["type"]["name"] == "SET PIECE":
                raw_events[event["index"]] = event
    records = []
    for _, event in sorted(raw_events.items()):
        subtype = event.get("subtypes") or []
        if isinstance(subtype, dict):
            subtype = [subtype]
        records.append({"Team": teams[event["team"]["id"]], "Type": event["type"]["name"],
                        "Subtype": "-".join(item["name"] for item in subtype), "Period": event["period"],
                        "Start Frame": event["start"]["frame"], "Start Time [s]": event["start"]["time"],
                        "From": event["from"]["id"], "To": (event.get("to") or {}).get("id"),
                        "Start X": event["start"]["x"], "Start Y": event["start"]["y"],
                        "End X": event["end"]["x"], "End Y": event["end"]["y"]})
    return tracking, pd.DataFrame(records), {
        "source_score": {"home": score.home, "away": score.away}, "source_team_ids": teams,
    }
