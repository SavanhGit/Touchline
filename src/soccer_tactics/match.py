"""Match context and inspectable data quality, independent of dashboard state."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from soccer_tactics.schema import FPS, validate, validate_events


@dataclass
class MatchBundle:
    match_id: str
    tracking: pd.DataFrame
    events: pd.DataFrame
    goals: pd.DataFrame
    source: str
    fps: float
    score_known: bool = False
    team_names: dict = field(default_factory=lambda: {"home": "Home (anonymous)", "away": "Away (anonymous)"})
    quality: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    event_origins: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def score(self) -> dict | None:
        if not self.score_known:
            return None
        return {team: int((self.goals["team"] == team).sum()) for team in ("home", "away")}


def validate_bundle(bundle: MatchBundle) -> None:
    """Reject invalid source rows before analysis, regardless of cached quality claims.

    Nominal pitch bounds are diagnostic only. Shots may lack an endpoint, but
    every retained tracking position, event origin and pass endpoint is required.
    """
    validate(bundle.tracking)
    validate_events(bundle.events)
    if not bundle.event_origins.empty:
        origins = bundle.event_origins
        required_origin = {"match_id", "period", "team", "event", "player_id", "source_code",
                           "source_timestamp", "gameclock_s", "source_x_m", "source_y_m", "x_m", "y_m"}
        missing_origin = required_origin - set(origins)
        if missing_origin:
            raise ValueError(f"event origins missing columns: {sorted(missing_origin)}")
        if {"frame", "x1", "y1", "to_x", "to_y"} & set(origins):
            raise ValueError("event origins cannot claim tracking frames or destinations")
        if set(origins.match_id.astype(str)) != {str(bundle.match_id)}:
            raise ValueError("event origins contain a different match from the declared match")
        if not origins.team.isin(["home", "away"]).all():
            raise ValueError("event origins must have team home or away")
        if pd.to_datetime(origins.source_timestamp, utc=True, errors="coerce",
                          format="ISO8601").isna().any():
            raise ValueError("event origins must have valid source timestamps")
        if not np.isfinite(origins[["period", "gameclock_s", "source_x_m", "source_y_m", "x_m", "y_m"]].to_numpy(dtype=float)).all():
            raise ValueError("event origins must have finite periods, clocks, and coordinates")
    if bundle.tracking.empty:
        raise ValueError("empty tracking cannot form a match report")
    for label, rows in (("tracking", bundle.tracking), ("events", bundle.events)):
        if not rows.empty and set(rows.match_id.astype(str)) != {str(bundle.match_id)}:
            raise ValueError(f"bundle {label} contain a different match from the declared match")
    try:
        valid_fps = bool(np.isfinite(bundle.fps) and bundle.fps > 0)
    except TypeError:
        valid_fps = False
    if not valid_fps:
        raise ValueError("bundle fps must be positive and finite")
    keys = ["match_id", "period", "frame", "team", "player_id"]
    if bundle.tracking.duplicated(keys).any():
        raise ValueError("repair duplicate tracking object keys before analysis")
    clocks = bundle.tracking[["period", "frame", "time_s"]].drop_duplicates()
    if clocks.duplicated(["period", "frame"]).any():
        raise ValueError("repair conflicting tracking frame clocks before analysis")
    delta = clocks.sort_values(["period", "frame"]).groupby("period").time_s.diff()
    if (delta < 0).any():
        raise ValueError("source time must be nondecreasing as frames advance within each period")

    goals = bundle.goals
    required = ["team", "period", "frame", "time_s", "player_id", "penalty"]
    missing = [column for column in required if column not in goals]
    if missing:
        raise ValueError(f"goal records missing columns: {missing}")
    if "match_id" in goals and not goals.empty:
        if goals.match_id.isna().any() or set(goals.match_id.astype(str)) != {str(bundle.match_id)}:
            raise ValueError("goal records contain a different match from the declared match")
    if not goals.team.isin(["home", "away"]).all():
        raise ValueError("goal records must have team home or away")
    if goals.player_id.isna().any():
        raise ValueError("goal records cannot have missing player_id")
    if not goals.empty:
        try:
            finite_goals = np.isfinite(goals[["period", "frame", "time_s"]].to_numpy()).all()
        except TypeError:
            finite_goals = False
        if not finite_goals:
            raise ValueError("goal periods, frames and timestamps must be finite numeric values")
        if (goals[["period", "frame"]] % 1 != 0).any().any():
            raise ValueError("goal periods and frames must be integers")
    if bundle.quality.get("duplicate_objects", 0) or bundle.quality.get("conflicting_clocks", 0):
        raise ValueError("quality metadata flags duplicate objects or conflicting clocks; repair before analysis")


def infer_fps(tracking: pd.DataFrame) -> float:
    """Infer source frame rate from unique period/frame clocks; isolated rows use 25 Hz."""
    groups = ["match_id", "period"]
    clock = tracking[groups + ["frame", "time_s"]].drop_duplicates().sort_values(groups + ["frame"])
    delta = clock.groupby(groups)[["frame", "time_s"]].diff()
    rates = (delta["frame"] / delta["time_s"]).replace([np.inf, -np.inf], np.nan)
    rates = rates[rates > 0]
    return round(float(rates.median()), 8) if len(rates) else float(FPS)


def inspect_quality(tracking: pd.DataFrame, events: pd.DataFrame) -> dict:
    keys = ["match_id", "period", "frame", "team", "player_id"]
    players = tracking[tracking.team != "ball"]
    origins = events.merge(players[keys + ["x", "y"]], on=keys, how="left")
    distances = np.hypot(origins.x0 - origins.x, origins.y0 - origins.y)
    shots = events[events.event == "shot"]
    direction_ok = (shots.x0.where(shots.team == "home", -shots.x0) > 0)
    clocks = tracking[["match_id", "period", "frame", "time_s"]].drop_duplicates()
    groups = clocks.sort_values(["match_id", "period", "frame"]).groupby(["match_id", "period"])
    gaps = groups.frame.diff()
    return {
        "tracking_rows": len(tracking), "events": len(events),
        "periods": sorted(int(p) for p in tracking.period.unique()),
        "fps": round(infer_fps(tracking), 4),
        "duplicate_objects": int(tracking.duplicated(keys).sum()),
        "missing_positions": int(tracking[["x", "y"]].isna().any(axis=1).sum()),
        "nonfinite_positions": int((~np.isfinite(tracking[["x", "y"]])).any(axis=1).sum()),
        "out_of_bounds": int(((tracking.x.abs() > 52.5) | (tracking.y.abs() > 34)).sum()),
        "frame_gaps": int((gaps > 1).sum()),
        "conflicting_clocks": int(clocks.duplicated(["match_id", "period", "frame"]).sum()),
        "event_tracking_coverage": float(distances.notna().mean()) if len(events) else None,
        "alignment_median_m": float(distances.median()) if distances.notna().any() else None,
        "alignment_p90_m": float(distances.quantile(.9)) if distances.notna().any() else None,
        "shots_in_attacking_half": float(direction_ok.mean()) if len(shots) else None,
    }
