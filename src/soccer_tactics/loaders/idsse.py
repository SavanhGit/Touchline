"""Loader for the IDSSE Bundesliga XML dataset.

IDSSE is distributed as DFL XML and is parsed through the official
Floodlight reader. Floodlight remains optional so users who only use Metrica
do not need the XML stack installed.
"""
from pathlib import Path
import re

import pandas as pd

from soccer_tactics.schema import EVENT_COLUMNS, COLUMNS, PITCH_LENGTH, PITCH_WIDTH, validate, validate_events


def _require_floodlight():
    try:
        from floodlight.io.dfl import read_event_data_xml, read_position_data_xml
    except ImportError as exc:
        raise ImportError(
            "IDSSE loading requires the optional 'idsse' dependency: "
            "pip install -e '.[idsse]'"
        ) from exc
    return read_event_data_xml, read_position_data_xml


def _match_key(path: Path) -> str:
    match = re.search(r"(DFL-MAT-[A-Z0-9]+)", path.name)
    if not match:
        raise ValueError(f"cannot find IDSSE match id in {path.name}")
    return match.group(1)


def _files(root: Path) -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = {}
    for path in root.glob("*.xml"):
        name = path.name
        kind = (
            "info" if "matchinformation" in name else
            "events" if "events_raw" in name else
            "positions" if "positions_raw" in name else None
        )
        if kind:
            found.setdefault(_match_key(path), {})[kind] = path
    return found


def _tracking(xy_objects, teamsheets, match_id: str, fps: float) -> pd.DataFrame:
    rows = []
    period_ids = {"firstHalf": 1, "secondHalf": 2}
    for segment, period in period_ids.items():
        if segment not in xy_objects:
            continue
        for team_key, team in (("Home", "home"), ("Away", "away"), ("Ball", "ball")):
            xy = xy_objects[segment][team_key]
            if team_key == "Ball":
                ids = ["ball"]
            else:
                sheet = teamsheets[team_key].teamsheet.sort_values("xID")
                ids = sheet["pID"].astype(str).tolist()
            for player_index, player_id in enumerate(ids):
                x = xy.x[:, player_index]
                y = xy.y[:, player_index]
                for offset, (px, py) in enumerate(zip(x, y)):
                    if pd.isna(px) or pd.isna(py):
                        continue
                    rows.append((match_id, period, offset, offset / fps, team, player_id, px, py))
    return validate(pd.DataFrame(rows, columns=COLUMNS))


def _events(event_objects, match_id: str, fps: float) -> pd.DataFrame:
    rows = []
    for segment, period in (("firstHalf", 1), ("secondHalf", 2)):
        for team_key, events in event_objects.get(segment, {}).items():
            team = "home" if team_key == "Home" else "away" if team_key == "Away" else None
            if team is None:
                continue
            for _, event in events.events.iterrows():
                eid = str(event["eID"])
                kind = "pass" if "Pass" in eid or "Cross" in eid else "shot" if "ShotAtGoal" in eid else None
                if kind is None:
                    continue
                x0, y0 = event.get("at_x"), event.get("at_y")
                x1, y1 = event.get("to_x"), event.get("to_y")
                if pd.isna(x0) or pd.isna(y0):
                    continue
                if pd.isna(x1) or pd.isna(y1):
                    x1, y1 = x0, y0
                clock = float(event["gameclock"])
                rows.append((
                    match_id, period, team, kind,
                    bool(event["outcome"]) if pd.notna(event["outcome"]) else False,
                    str(event["pID"]), None, round(clock * fps), clock,
                    x0, y0, x1, y1,
                ))
    return validate_events(pd.DataFrame(rows, columns=EVENT_COLUMNS))


def load_idsse_game(root: str | Path, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one IDSSE match identified by its ``DFL-MAT-*`` id."""
    read_events, read_positions = _require_floodlight()
    files = _files(Path(root)).get(match_id)
    if not files or not {"info", "events", "positions"} <= files.keys():
        raise FileNotFoundError(f"complete IDSSE XML set not found for {match_id}")
    xy, _, _, teamsheets, pitch = read_positions(files["positions"], files["info"])
    event_objects, _, _ = read_events(files["events"], files["info"])
    fps = next(iter(xy.values()))[next(iter(next(iter(xy.values())).keys()))].framerate
    tracking = _tracking(xy, teamsheets, match_id, fps)
    events = _events(event_objects, match_id, fps)
    return tracking, events


def list_idsse_matches(root: str | Path) -> list[str]:
    """List IDSSE match ids with complete metadata, event, and tracking files."""
    return sorted(k for k, files in _files(Path(root)).items() if {"info", "events", "positions"} <= files.keys())
