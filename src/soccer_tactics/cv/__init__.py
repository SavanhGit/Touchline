"""Phase 2: video -> tracking.

Wrap roboflow/sports (pitch keypoints, player/ball detection, tracking, homography to
pitch meters) and emit frames matching `soccer_tactics.schema.COLUMNS`, so everything
downstream runs unchanged.
"""

from soccer_tactics.cv.adapter import TrackingAdapter, detections_to_tracking
from soccer_tactics.cv.pipeline import Tracker, VisionBackend, run_video_tracking
from soccer_tactics.cv.roboflow_backend import RoboflowVideoBackend, SupervisionByteTracker

__all__ = ["TrackingAdapter", "detections_to_tracking", "Tracker", "VisionBackend",
    "run_video_tracking", "RoboflowVideoBackend", "SupervisionByteTracker"]
