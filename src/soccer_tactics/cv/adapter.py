"""Boundary between video models and the canonical tracking schema."""
import pandas as pd

from soccer_tactics.schema import validate


REQUIRED_DETECTION_COLUMNS = {"frame", "team", "player_id", "x", "y"}


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
