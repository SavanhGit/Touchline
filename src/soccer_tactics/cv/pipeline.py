"""Sequential local video inference with independent calibration per frame."""
import math
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd

from soccer_tactics.cv.calibration import estimate_calibration, project_points
from soccer_tactics.cv.run_store import source_sha256
from soccer_tactics.cv.video_types import (
    PitchLandmarks, PlayerDetection, TrackedPlayer, VideoTrackingConfig, VideoTrackingResult,
)


class VisionBackend(Protocol):
    def detect_players(self, frame: np.ndarray) -> list[PlayerDetection]: ...
    def detect_pitch_landmarks(self, frame: np.ndarray) -> PitchLandmarks: ...


class Tracker(Protocol):
    def update(self, frame_index: int, detections: list[PlayerDetection]) -> list[TrackedPlayer]: ...


def _validate_config(config: VideoTrackingConfig) -> None:
    if not isinstance(config.match_id, str) or not config.match_id.strip():
        raise ValueError('match_id must be a nonempty label')
    if not isinstance(config.home_attacks_positive_x, bool):
        raise ValueError('home_attacks_positive_x must be a boolean')
    for confidence in (config.min_player_confidence, config.min_pitch_keypoint_confidence):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence thresholds must be finite and between 0 and 1')
    if not isinstance(config.min_calibration_inliers, int) or config.min_calibration_inliers < 4:
        raise ValueError('min_calibration_inliers must be an integer of at least four')
    if not math.isfinite(config.max_reprojection_error_px) or config.max_reprojection_error_px <= 0:
        raise ValueError('max_reprojection_error_px must be finite and positive')


def run_video_tracking(video_path: Path, config: VideoTrackingConfig, *, backend: VisionBackend,
    tracker: Tracker, work_dir: Path, on_progress: Callable[[int, int], None] | None = None,
) -> VideoTrackingResult:
    """Create one review overlay, clip tracks and diagnostics without retaining source paths.

    Coordinates are centred pitch metres. Team direction is applied only at the
    assignment/persistence boundary. Any failed run removes its partial overlay.
    """
    _validate_config(config)
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError('Source video does not exist or is not a regular file.')
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('OpenCV is missing. Install soccer-tactics[cv] to process local video.') from exc
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    pending = None
    complete = False
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video. Choose a readable MP4 with a supported codec.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video FPS must be finite and positive. Re-encode the source video.')
        raw = [capture.get(key) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
        if any(not math.isfinite(value) or value <= 0 or value != int(value) for value in raw):
            raise ValueError('Video metadata must contain positive integer dimensions and frame count.')
        width, height, frame_count = map(int, raw)
        duration_s = frame_count / fps
        if not math.isfinite(duration_s):
            raise ValueError('Video metadata produces a nonfinite duration. Re-encode the source video.')
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError('Cannot decode the first video frame. Re-encode the source video.')
        metadata = {'match_id': config.match_id, 'home_attacks_positive_x': config.home_attacks_positive_x,
            'fps': float(fps), 'width': width, 'height': height, 'frame_count': frame_count,
            'duration_s': duration_s, 'source_sha256': source_sha256(video_path)}
        provenance = getattr(backend, 'provenance', None)
        if provenance is not None:
            from soccer_tactics.cv.run_store import _safe_provenance
            metadata['provenance'] = _safe_provenance(provenance)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        pending, destination = work_dir / f'.pending-{token}.mp4', work_dir / f'overlay-{token}.mp4'
        writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('Cannot create video overlay. Check the output directory and MP4 encoder.')
        rows, diagnostics = [], []
        index = 0
        while ok:
            if frame is None or frame.shape != (height, width, 3):
                raise ValueError('Decoded video frame dimensions disagree with video metadata.')
            detections = backend.detect_players(frame)
            low_confidence = 0
            accepted = []
            for detection in detections:
                if detection.class_name not in ('player', 'goalkeeper'):
                    continue
                if not math.isfinite(detection.confidence) or detection.confidence < config.min_player_confidence:
                    low_confidence += 1
                else:
                    accepted.append(detection)
            tracks = tracker.update(index, accepted)
            landmarks = backend.detect_pitch_landmarks(frame)
            confidence = np.asarray(landmarks.confidence)
            keep = np.isfinite(confidence) & (confidence >= config.min_pitch_keypoint_confidence)
            calibration = estimate_calibration(PitchLandmarks(
                np.asarray(landmarks.image_xy)[keep], np.asarray(landmarks.pitch_xy_m)[keep], confidence[keep]),
                min_inliers=config.min_calibration_inliers,
                max_reprojection_error_px=config.max_reprojection_error_px)
            outside = 0
            emitted = 0
            annotated = frame.copy()
            for track in tracks:
                box = np.asarray(track.bbox_xyxy, dtype=float)
                if box.shape != (4,) or not np.isfinite(box).all():
                    outside += 1
                    continue
                left, top, right, bottom = box
                cv2.rectangle(annotated, (int(left), int(top)), (int(right), int(bottom)), (0, 220, 255), 2)
                cv2.putText(annotated, str(track.track_id), (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
                if not calibration.valid:
                    continue
                try:
                    x, y = project_points(np.array([[(left + right) / 2, bottom]]), calibration)[0]
                except ValueError:
                    outside += 1
                    continue
                if not np.isfinite([x, y]).all() or abs(x) > 52.5 or abs(y) > 34:
                    outside += 1
                    continue
                rows.append({'frame': index, 'track_id': str(track.track_id), 'x': float(x), 'y': float(y),
                    'confidence': track.confidence, 'class_name': track.class_name})
                emitted += 1
            diagnostics.append({'frame': index, 'calibration_valid': calibration.valid,
                'calibration_failed': int(not calibration.valid), 'calibration_reason': calibration.reason,
                'inlier_count': calibration.inlier_count, 'median_error_px': calibration.median_error_px,
                'low_confidence_players': low_confidence, 'out_of_pitch_players': outside,
                'emitted_positions': emitted})
            cv2.putText(annotated, f'Frame {index} | calibration: {"valid" if calibration.valid else "FAILED"}',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            writer.write(annotated)
            index += 1
            if on_progress is not None:
                on_progress(index, frame_count)
            ok, frame = capture.read()
        if index != frame_count:
            raise ValueError(f'Video decode incomplete: decoded {index} of {frame_count} frames. Re-encode the source video.')
        writer.release()
        writer = None
        pending.rename(destination)
        complete = True
        return VideoTrackingResult(pd.DataFrame(rows, columns=['frame', 'track_id', 'x', 'y', 'confidence', 'class_name']),
            pd.DataFrame(diagnostics), metadata, destination)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if pending is not None and not complete:
            pending.unlink(missing_ok=True)
