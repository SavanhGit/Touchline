"""Match context and inspectable data quality, independent of dashboard state."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from soccer_tactics.schema import FPS


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

    @property
    def score(self) -> dict | None:
        if not self.score_known:
            return None
        return {team: int((self.goals["team"] == team).sum()) for team in ("home", "away")}


def infer_fps(tracking: pd.DataFrame) -> float:
    """Infer source frame rate from unique period/frame clocks; isolated rows use 25 Hz."""
    clock = tracking[["period", "frame", "time_s"]].drop_duplicates().sort_values(["period", "frame"])
    delta = clock.groupby("period")[["frame", "time_s"]].diff()
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
    clocks = tracking[["period", "frame", "time_s"]].drop_duplicates()
    groups = clocks.sort_values(["period", "frame"]).groupby("period")
    gaps = groups.frame.diff()
    return {
        "tracking_rows": len(tracking), "events": len(events),
        "periods": sorted(int(p) for p in tracking.period.unique()),
        "fps": round(infer_fps(tracking), 4),
        "duplicate_objects": int(tracking.duplicated(keys).sum()),
        "missing_positions": int(tracking[["x", "y"]].isna().any(axis=1).sum()),
        "out_of_bounds": int(((tracking.x.abs() > 52.5) | (tracking.y.abs() > 34)).sum()),
        "frame_gaps": int((gaps > 1).sum()),
        "conflicting_clocks": int(clocks.duplicated(["period", "frame"]).sum()),
        "event_tracking_coverage": float(distances.notna().mean()) if len(events) else None,
        "alignment_median_m": float(distances.median()) if distances.notna().any() else None,
        "alignment_p90_m": float(distances.quantile(.9)) if distances.notna().any() else None,
        "shots_in_attacking_half": float(direction_ok.mean()) if len(shots) else None,
    }
