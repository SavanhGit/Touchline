"""Source adapter for the IDSSE/DFL XML triplets.

Floodlight is optional and is imported only for a requested complete match. DFL
event coordinates are source coordinates; orientation and event/position clock
alignment belong to the bundle boundary, which has both source objects here.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import pandas as pd

from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS, validate, validate_events


_PERIODS = {"firstHalf": 1, "secondHalf": 2}
_FAMILIES = {"matchinformation": "info", "events_raw": "events",
             "positions_raw_observed": "positions"}
_PASS_CODES = {"Pass", "Cross", "Play_Pass", "Play_Cross"}
_PASS_EVALUATIONS = {"successfullyCompleted": True, "unsuccessful": False}
_SHOT_RESULTS = {"SuccessfulShot": True, "SavedShot": False,
                 "BlockedShot": False, "ShotWide": False,
                 "ShotWoodWork": False, "OtherShot": False}
_GOAL_WRAPPERS = {"Play", "FreeKick", "CornerKick", "Penalty",
                  "ThrowIn", "GoalKick", "KickOff"}
_GOAL_COLUMNS = ["match_id", "team", "period", "frame", "time_s", "player_id",
                 "penalty", "source_code", "timestamp"]
_EXCLUSION_COLUMNS = ["period", "team", "source_code", "reason", "timestamp"]


@dataclass(frozen=True)
class IdsseSourceMatch:
    match_id: str
    files: dict[str, Path]
    positions: object
    events: object
    teamsheets: object
    pitch: object
    match_information: object
    position_clocks: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class IdsseEventSelection:
    events: pd.DataFrame
    goals: pd.DataFrame
    exclusions: pd.DataFrame
    source_events: pd.DataFrame
    total: int


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


def _files(root: Path) -> dict[str, dict[str, list[Path]]]:
    found: dict[str, dict[str, list[Path]]] = {}
    for path in root.glob("*.xml"):
        family = next((kind for fragment, kind in _FAMILIES.items()
                       if fragment in path.name.lower()), None)
        if family is None:
            continue
        found.setdefault(_match_key(path), {}).setdefault(family, []).append(path)
    return found


def _one_triplet(root: Path, match_id: str) -> dict[str, Path]:
    families = _files(root).get(match_id, {})
    duplicate = sorted(kind for kind, paths in families.items() if len(paths) != 1)
    if duplicate:
        raise ValueError(f"duplicate IDSSE XML family for {match_id}: {', '.join(duplicate)}")
    if set(families) != {"info", "events", "positions"}:
        raise FileNotFoundError(f"complete IDSSE XML set not found for {match_id}")
    return {kind: paths[0] for kind, paths in families.items()}


def _position_clocks(path: Path) -> dict[str, pd.DataFrame]:
    """Read observed ball N/T keys without duplicating position coordinates.

    Floodlight's DFL position reader indexes XY arrays from the first N but
    discards both N and T. Ball FrameSets are the source clock for each period.
    T stays as the exact XML string until the bundle checks event alignment.
    """
    clocks: dict[str, pd.DataFrame] = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "FrameSet":
            continue
        if str(element.get("TeamId", "")).lower() == "ball":
            segment = element.get("GameSection")
            if segment in clocks:
                raise ValueError(f"duplicate ball FrameSet for {segment}")
            rows = []
            for frame in element.iter("Frame"):
                number, timestamp = frame.get("N"), frame.get("T")
                if number is None or timestamp is None:
                    raise ValueError(f"missing observed frame N/T for {segment}")
                rows.append((int(number), timestamp))
            clocks[segment] = pd.DataFrame(rows, columns=["frame", "timestamp"])
        element.clear()
    return clocks


def _read_idsse_source(root: str | Path, match_id: str) -> IdsseSourceMatch:
    """Discover one complete triplet and return unmodified Floodlight objects."""
    files = _one_triplet(Path(root), match_id)
    read_events, read_positions = _require_floodlight()
    positions, _, _, teamsheets, pitch = read_positions(files["positions"], files["info"])
    events, _, _ = read_events(files["events"], files["info"])
    match_information = ET.parse(files["info"]).getroot()
    return IdsseSourceMatch(match_id, files, positions, events, teamsheets, pitch,
                            match_information, _position_clocks(files["positions"]))


def _event_kind(code: str) -> tuple[str | None, bool, bool]:
    if code in _PASS_CODES:
        return "pass", False, False
    if code == "OwnGoal":
        return None, True, False
    parts = code.split("_")
    if len(parts) == 2 and parts[0] == "ShotAtGoal" and parts[1] in _SHOT_RESULTS:
        return "shot", _SHOT_RESULTS[parts[1]], False
    if len(parts) == 3 and parts[0] == "Penalty" and parts[1] == "ShotAtGoal" and parts[2] in _SHOT_RESULTS:
        return "shot", _SHOT_RESULTS[parts[2]], True
    return None, False, False


def _scored_goal(code: str) -> tuple[bool, bool]:
    """Recognize a scored DFL shot independently of the modeling whitelist."""
    if code == "OwnGoal":
        return True, False
    parts = code.split("_")
    wrappers = parts[:-2]
    if (len(parts) < 2 or parts[-2:] != ["ShotAtGoal", "SuccessfulShot"]
            or len(wrappers) > 2 or any(part not in _GOAL_WRAPPERS for part in wrappers)):
        return False, False
    return True, "Penalty" in wrappers


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pass_success(event: pd.Series) -> tuple[bool | None, str | None]:
    """Resolve a pass from exact DFL evaluation or canonical binary outcome."""
    raw = event.get("outcome")
    if raw is None or raw is pd.NA or (isinstance(raw, Real) and math.isnan(float(raw))):
        outcome = None
    elif isinstance(raw, Real) and _finite(raw) and float(raw) in (0.0, 1.0):
        outcome = bool(raw)
    else:
        return None, "invalid_pass_outcome"

    qualifier = event.get("qualifier")
    evaluation = qualifier.get("Evaluation") if isinstance(qualifier, Mapping) else None
    if evaluation is not None:
        if not isinstance(evaluation, str) or evaluation not in _PASS_EVALUATIONS:
            return None, "unknown_pass_evaluation"
        evaluated = _PASS_EVALUATIONS[evaluation]
        if outcome is not None and outcome != evaluated:
            return None, "conflicting_pass_outcome"
        return evaluated, None
    if outcome is None:
        return None, "missing_pass_outcome"
    return outcome, None


def select_idsse_events(event_objects: object, match_id: str) -> IdsseEventSelection:
    """Select documented modeling codes; retain every source exclusion.

    ``frame=-1`` means alignment is pending: multiplying an event gameclock by
    framerate would invent an observed frame. The original timestamp is kept
    with each source row for period-local alignment at the bundle boundary.
    """
    rows, goals, excluded, originals = [], [], [], []
    for segment, team_objects in event_objects.items():
        period = _PERIODS.get(segment)
        for team_key, object_ in team_objects.items():
            team = {"Home": "home", "Away": "away"}.get(team_key)
            for _, event in object_.events.iterrows():
                code = str(event.get("eID"))
                timestamp = event.get("timestamp")
                source_row = {**event.to_dict(), "period": period, "team": team,
                              "canonical_index": None, "goal_index": None}
                originals.append(source_row)
                reason = None
                kind, modeled_goal, modeling_penalty = _event_kind(code)
                scored, penalty = _scored_goal(code)
                if period is None or team is None:
                    reason = "unsupported_period_or_team"
                elif kind is None and not scored:
                    reason = "unsupported_event_code"
                elif not _finite(event.get("gameclock")) or pd.isna(timestamp):
                    reason = "invalid_event_time"
                elif pd.isna(event.get("pID")):
                    reason = "missing_player_id"
                if reason is not None:
                    excluded.append((period, team, code, reason, timestamp))
                    continue
                clock = float(event["gameclock"])
                player = str(event["pID"])
                if scored:
                    credited = ("away" if team == "home" else "home") if code == "OwnGoal" else team
                    source_row["goal_index"] = len(goals)
                    goals.append((match_id, credited, period, -1, clock, player,
                                  penalty, code, timestamp))
                if code == "OwnGoal":
                    excluded.append((period, team, code, "own_goal_not_modeling_shot", timestamp))
                    continue
                if kind is None:
                    excluded.append((period, team, code, "unsupported_modeling_goal_code", timestamp))
                    continue
                if not _finite(event.get("at_x")) or not _finite(event.get("at_y")):
                    excluded.append((period, team, code, "missing_event_origin", timestamp))
                    continue
                if kind == "pass" and (not _finite(event.get("to_x")) or not _finite(event.get("to_y"))):
                    excluded.append((period, team, code, "missing_pass_endpoint", timestamp))
                    continue
                if kind == "pass":
                    pass_success, pass_reason = _pass_success(event)
                    if pass_reason is not None:
                        excluded.append((period, team, code, pass_reason, timestamp))
                        continue
                if modeling_penalty:
                    excluded.append((period, team, code, "penalty_shot_excluded", timestamp))
                    continue
                x0, y0 = float(event["at_x"]), float(event["at_y"])
                x1, y1 = (float(event["to_x"]), float(event["to_y"])) if kind == "pass" else (
                    float(event["to_x"]) if _finite(event.get("to_x")) else x0,
                    float(event["to_y"]) if _finite(event.get("to_y")) else y0)
                success = pass_success if kind == "pass" else modeled_goal
                source_row["canonical_index"] = len(rows)
                rows.append((match_id, period, team, kind, success, player, None,
                             -1, clock, x0, y0, x1, y1))
    return IdsseEventSelection(
        validate_events(pd.DataFrame(rows, columns=EVENT_COLUMNS)),
        pd.DataFrame(goals, columns=_GOAL_COLUMNS),
        pd.DataFrame(excluded, columns=_EXCLUSION_COLUMNS),
        pd.DataFrame(originals), len(originals),
    )


def _tracking(source: IdsseSourceMatch) -> pd.DataFrame:
    rows = []
    for segment, period in _PERIODS.items():
        if segment not in source.positions:
            continue
        clock = source.position_clocks.get(segment)
        if clock is None or clock.empty:
            raise ValueError(f"observed ball frame clock missing for {segment}")
        times = (pd.to_datetime(clock["timestamp"], utc=True) -
                 pd.to_datetime(clock["timestamp"].iloc[0], utc=True)).dt.total_seconds()
        for team_key, team in (("Home", "home"), ("Away", "away"), ("Ball", "ball")):
            xy = source.positions[segment][team_key]
            offsets = clock.frame.to_numpy(dtype=int) - int(clock.frame.iloc[0])
            if (offsets < 0).any() or (offsets >= len(xy.x)).any():
                raise ValueError(f"observed frame N outside Floodlight XY range for {segment}/{team_key}")
            ids = (["ball"] if team_key == "Ball" else
                   source.teamsheets[team_key].teamsheet.sort_values("xID")["pID"].astype(str).tolist())
            for index, player_id in enumerate(ids):
                for clock_index, xy_offset in enumerate(offsets):
                    x, y = xy.x[xy_offset, index], xy.y[xy_offset, index]
                    if pd.isna(x) or pd.isna(y):
                        continue
                    rows.append((source.match_id, period, int(clock.frame.iloc[clock_index]),
                                 float(times.iloc[clock_index]), team, player_id, x, y))
    return validate(pd.DataFrame(rows, columns=COLUMNS))


def load_idsse_game(root: str | Path, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one IDSSE match as the existing tracking/events tuple."""
    source = _read_idsse_source(root, match_id)
    return _tracking(source), select_idsse_events(source.events, match_id).events


def list_idsse_matches(root: str | Path) -> list[str]:
    """List IDs with exactly one file from each required XML family."""
    return sorted(match_id for match_id, files in _files(Path(root)).items()
                  if set(files) == {"info", "events", "positions"}
                  and all(len(paths) == 1 for paths in files.values()))
