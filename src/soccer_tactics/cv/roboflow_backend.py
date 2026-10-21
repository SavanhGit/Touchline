"""Local adapters for the two Roboflow soccer YOLO checkpoints.

Imports never load models, download weights, or require the optional CV stack.
"""
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from soccer_tactics.cv.run_store import RunProvenance, source_sha256
from soccer_tactics.cv.video_types import PitchLandmarks, PlayerDetection, TrackedPlayer


def _pitch_vertices() -> np.ndarray:
    """Roboflow's 32 keypoint order, normalized to our 105 x 68 m pitch.

    Geometry/order: github.com/roboflow/sports, sports/configs/soccer.py.
    The checkpoint's reference is 120 x 70 m, with centimetre units.
    """
    edge_y = [0, 14.5, 25.84, 44.16, 55.5, 70]
    vertices = [(0, y) for y in edge_y]
    vertices += [(5.5, 25.84), (5.5, 44.16), (11, 35)]
    vertices += [(20.15, y) for y in edge_y[1:5]]
    vertices += [(60, y) for y in [0, 25.85, 44.15, 70]]
    vertices += [(99.85, y) for y in edge_y[1:5]]
    vertices += [(109, 35), (114.5, 25.84), (114.5, 44.16)]
    vertices += [(120, y) for y in edge_y]
    vertices += [(50.85, 35), (69.15, 35)]
    return np.asarray(vertices) * [105 / 120, 68 / 70] - [52.5, 34]


@dataclass
class RoboflowVideoBackend:
    player_model: object
    pitch_model: object
    device: str = 'cpu'
    provenance: RunProvenance | None = None

    @classmethod
    def from_weight_paths(cls, player_weights: Path, pitch_weights: Path, *, device: str = 'cpu'):
        paths = [Path(player_weights), Path(pitch_weights)]
        if any(not path.is_file() for path in paths):
            raise ValueError('Missing local model weights. Run scripts/download_video_models.py explicitly first.')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Local CV dependencies are missing. Install soccer-tactics[cv]; see the Python 3.15 setup instructions.') from exc
        packages = {}
        for name in ('ultralytics', 'torch', 'torchvision', 'supervision', 'opencv-python'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                pass
        provenance = RunProvenance(models={
            role: {'id': identifier, 'version': source_sha256(path)}
            for role, identifier, path in zip(('player', 'pitch'),
                ('roboflow-football-player-detection', 'roboflow-football-pitch-detection'), paths)
        }, packages=packages)
        return cls(YOLO(str(paths[0])), YOLO(str(paths[1])), device, provenance)

    def detect_players(self, frame: np.ndarray) -> list[PlayerDetection]:
        result = self.player_model(frame, device=self.device, imgsz=1280, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes
        detections = []
        for xyxy, score, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            name = result.names[int(class_id)].lower()
            if name in ('player', 'goalkeeper'):
                detections.append(PlayerDetection(tuple(float(x) for x in xyxy), float(score), name))
        return detections

    def detect_pitch_landmarks(self, frame: np.ndarray) -> PitchLandmarks:
        result = self.pitch_model(frame, device=self.device, verbose=False)[0]
        keypoints = result.keypoints
        if keypoints is None or keypoints.conf is None or len(keypoints.xy) == 0:
            return PitchLandmarks(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
        xy = keypoints.xy.cpu().numpy()[0]
        confidence = keypoints.conf.cpu().numpy()[0]
        if xy.shape != (32, 2) or confidence.shape != (32,):
            raise ValueError('Pitch weights must use the Roboflow 32-landmark soccer layout.')
        return PitchLandmarks(xy, _pitch_vertices(), confidence)


class SupervisionByteTracker:
    """A fresh instance owns IDs for one clip; update once per decoded frame."""

    def __init__(self, *, frame_rate: int = 25):
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError('Supervision is missing. Install soccer-tactics[cv].') from exc
        self._sv = sv
        self._tracker = sv.ByteTrack(frame_rate=frame_rate)

    def update(self, frame_index: int, detections: list[PlayerDetection]) -> list[TrackedPlayer]:
        players = [d for d in detections if d.class_name in ('player', 'goalkeeper')]
        values = self._sv.Detections(
            xyxy=np.asarray([d.bbox_xyxy for d in players], dtype=float).reshape(-1, 4),
            confidence=np.asarray([d.confidence for d in players], dtype=float),
            class_id=np.asarray([1 if d.class_name == 'goalkeeper' else 2 for d in players], dtype=int),
        )
        tracked = self._tracker.update_with_detections(values)
        return [TrackedPlayer(frame_index, str(int(track_id)), tuple(float(x) for x in box),
            float(score), 'goalkeeper' if int(class_id) == 1 else 'player')
            for box, score, class_id, track_id in zip(tracked.xyxy, tracked.confidence,
                tracked.class_id, tracked.tracker_id)]
