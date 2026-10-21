"""Typed contracts for local video tracking and calibration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitchLandmarks:
    """Corresponding image pixels, pitch coordinates in metres, and scores."""

    image_xy: np.ndarray
    pitch_xy_m: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Image-pixel to pitch-metre homography estimated for one video frame."""

    valid: bool
    image_to_pitch: np.ndarray | None
    inlier_count: int
    median_error_px: float | None
    reason: str | None


@dataclass(frozen=True)
class VideoTrackingConfig:
    match_id: str
    home_attacks_positive_x: bool
    min_player_confidence: float = 0.25
    min_pitch_keypoint_confidence: float = 0.25
    min_calibration_inliers: int = 4
    max_reprojection_error_px: float = 12.0


@dataclass(frozen=True)
class PlayerDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str


@dataclass(frozen=True)
class TrackedPlayer:
    frame: int
    track_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str


@dataclass(frozen=True)
class VideoTrackingResult:
    detections: pd.DataFrame
    frame_diagnostics: pd.DataFrame
    video_metadata: dict[str, object]
    overlay_path: Path
