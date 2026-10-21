"""Boundary between video models and the canonical tracking schema."""
from dataclasses import dataclass
import math
from typing import Literal

import pandas as pd

from soccer_tactics.schema import validate


REQUIRED_DETECTION_COLUMNS = {"frame", "team", "player_id", "x", "y"}


@dataclass(frozen=True)
class TrackAssignment:
    """A team and optional roster label; the clip-local ID stays the identity."""

    team: Literal["home", "away"]
    roster_name: str | None = None

    def __post_init__(self):
        if self.team not in ("home", "away"):
            raise ValueError("track assignment team must be home or away")


def video_tracks_to_tracking(
    detections: pd.DataFrame,
    *,
    match_id: str,
    fps: float,
    assignments: dict[str, TrackAssignment],
    home_attacks_positive_x: bool,
    period: int = 1,
) -> pd.DataFrame:
    """Map assigned video tracks to canonical rows with home attacking +x."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    missing = {"frame", "track_id", "x", "y"} - set(detections.columns)
    if missing:
        raise ValueError(f"detections missing columns: {sorted(missing)}")
    if not isinstance(home_attacks_positive_x, bool):
        raise ValueError("home_attacks_positive_x must be a boolean")
    for assignment in assignments.values():
        if not isinstance(assignment, TrackAssignment):
            raise ValueError("assignments must contain TrackAssignment values")
    out = detections.loc[detections.track_id.isin(assignments), ["frame", "track_id", "x", "y"]].copy()
    out["team"] = out.track_id.map({track_id: value.team for track_id, value in assignments.items()})
    out["player_id"] = out.track_id
    if not home_attacks_positive_x:
        out[["x", "y"]] *= -1
    return detections_to_tracking(out, match_id=match_id, fps=fps, period=period)


def detections_to_tracking(
    detections: pd.DataFrame,
    *,
    match_id: str,
    fps: float = 25.0,
    period: int = 1,
) -> pd.DataFrame:
    """Convert pitch-mapped detector output to canonical tracking rows.

    Video systems may use any detector/tracker internally. They must emit
    pitch coordinates in metres and stable team/player labels at this boundary.
    """
    missing = REQUIRED_DETECTION_COLUMNS - set(detections.columns)
    if missing:
        raise ValueError(f"detections missing columns: {sorted(missing)}")
    out = detections.copy()
    out["match_id"] = match_id
    out["period"] = period
    out["time_s"] = out["frame"] / fps
    return validate(out)


class TrackingAdapter:
    """Small callable adapter for plugging in a video inference pipeline."""

    def __init__(self, infer):
        if not callable(infer):
            raise TypeError("infer must be callable")
        self.infer = infer

    def __call__(self, video) -> pd.DataFrame:
        detections = self.infer(video)
        if not isinstance(detections, pd.DataFrame):
            raise TypeError("video inference must return a pandas DataFrame")
        return detections_to_tracking(detections, match_id=str(video))
