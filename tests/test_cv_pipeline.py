"""Pipeline behavior at the model/codec boundaries, with no weights required."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from soccer_tactics.cv.video_types import (
    CalibrationResult, PitchLandmarks, PlayerDetection, TrackedPlayer, VideoTrackingConfig,
)


@pytest.fixture
def media(monkeypatch, tmp_path):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    state = SimpleNamespace(fps=25., width=64., height=48., count=5., opened=True, writes=0)
    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return state.opened
        def get(self, key): return getattr(state, key)
        def read(self):
            if self.index == len(frames): return False, None
            frame = frames[self.index]; self.index += 1
            return True, frame
        def release(self): pass
    class Writer:
        def __init__(self, path, *args): self.path = path
        def isOpened(self): return True
        def write(self, frame): state.writes += 1
        def release(self):
            from pathlib import Path
            Path(self.path).write_bytes(b'overlay')
    fake = SimpleNamespace(VideoCapture=Capture, VideoWriter=Writer, VideoWriter_fourcc=lambda *x: 0,
        CAP_PROP_FPS='fps', CAP_PROP_FRAME_WIDTH='width', CAP_PROP_FRAME_HEIGHT='height',
        CAP_PROP_FRAME_COUNT='count', FONT_HERSHEY_SIMPLEX=0,
        rectangle=lambda *a: None, putText=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'cv2', fake)
    source = tmp_path / 'input.mp4'; source.write_bytes(b'source')
    return state, source


def test_processes_every_frame_and_uses_only_current_calibration(media, monkeypatch, tmp_path):
    from soccer_tactics.cv import pipeline
    state, source = media
    class Backend:
        index = -1
        def detect_players(self, frame):
            self.index += 1
            if self.index == 2: return [PlayerDetection((0, 0, 2, 2), .1, 'player')]
            if self.index == 3: return [PlayerDetection((200, 0, 202, 2), .9, 'player')]
            return [PlayerDetection((0, 0, 2, 2), .9, 'player')]
        def detect_pitch_landmarks(self, frame):
            return PitchLandmarks(np.zeros((4, 2)), np.zeros((4, 2)), np.ones(4))
    backend = Backend()
    class Tracker:
        def update(self, frame_index, detections):
            return [TrackedPlayer(frame_index, '7', d.bbox_xyxy, d.confidence, d.class_name) for d in detections]
    def calibrate(*a, **kw):
        matrix = np.eye(3); matrix[0, 2] = backend.index
        return CalibrationResult(backend.index != 1, matrix, 4, 0., None)
    monkeypatch.setattr(pipeline, 'estimate_calibration', calibrate)
    progress = []
    result = pipeline.run_video_tracking(source, VideoTrackingConfig('clip', True),
        backend=backend, tracker=Tracker(), work_dir=tmp_path / 'work',
        on_progress=lambda n, total: progress.append((n, total)))
    assert result.detections[['frame', 'x', 'y']].values.tolist() == [[0, 1., 2.], [4, 5., 2.]]
    assert result.frame_diagnostics.calibration_failed.sum() == 1
    assert result.frame_diagnostics.low_confidence_players.sum() == 1
    assert result.frame_diagnostics.out_of_pitch_players.sum() == 1
    assert state.writes == 5
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert result.overlay_path.is_file()
    assert result.video_metadata['frame_count'] == 5
    assert result.video_metadata['match_id'] == 'clip'
    assert len(result.video_metadata['source_sha256']) == 64
    assert 'input.mp4' not in str(result.video_metadata)


@pytest.mark.parametrize('field,value', [('fps', 0), ('fps', float('nan')), ('fps', float('inf')), ('fps', 5e-324),
    ('width', 0), ('height', -1), ('count', float('nan')), ('count', 0), ('opened', False)])
def test_invalid_video_has_no_partial_artifact(media, tmp_path, field, value):
    from soccer_tactics.cv.pipeline import run_video_tracking
    state, source = media; setattr(state, field, value)
    with pytest.raises(ValueError, match='video|FPS|metadata'):
        run_video_tracking(source, VideoTrackingConfig('clip', True), backend=None,
            tracker=None, work_dir=tmp_path / 'work')
    assert not (tmp_path / 'work').exists()


def test_decode_failure_cleans_overlay(media, tmp_path):
    from soccer_tactics.cv.pipeline import run_video_tracking
    state, source = media; state.count = 6
    backend = SimpleNamespace(detect_players=lambda frame: [],
        detect_pitch_landmarks=lambda frame: PitchLandmarks(np.empty((0, 2)), np.empty((0, 2)), np.array([])))
    with pytest.raises(ValueError, match='decoded|decode'):
        run_video_tracking(source, VideoTrackingConfig('clip', True), backend=backend,
            tracker=SimpleNamespace(update=lambda *a: []), work_dir=tmp_path / 'work')
    assert not list((tmp_path / 'work').glob('*.mp4'))


def test_real_tiny_video_finishes_overlay_and_current_frame_projection(tmp_path):
    cv2 = pytest.importorskip('cv2')
    from soccer_tactics.cv.pipeline import run_video_tracking
    source = tmp_path / 'tiny.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10., (64, 48))
    if not writer.isOpened(): pytest.skip('MP4 encoder unavailable')
    for _ in range(3): writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    class Backend:
        index = -1
        def detect_players(self, frame):
            self.index += 1
            return [PlayerDetection((8, 4, 12, 10), .9, 'goalkeeper')]
        def detect_pitch_landmarks(self, frame):
            image = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], dtype=float)
            return PitchLandmarks(image, image + [self.index, 0], np.ones(4))
    tracker = SimpleNamespace(update=lambda index, detections: [
        TrackedPlayer(index, '3', d.bbox_xyxy, d.confidence, d.class_name) for d in detections])
    result = run_video_tracking(source, VideoTrackingConfig('tiny', False), backend=Backend(),
        tracker=tracker, work_dir=tmp_path / 'work')
    np.testing.assert_allclose(result.detections.x, [10, 11, 12], atol=.001)
    assert result.frame_diagnostics.calibration_valid.all()
    capture = cv2.VideoCapture(str(result.overlay_path))
    decoded = 0
    while capture.read()[0]: decoded += 1
    capture.release()
    assert decoded == 3


def test_callback_failure_removes_partial_overlay(media, tmp_path):
    from soccer_tactics.cv.pipeline import run_video_tracking
    _, source = media
    backend = SimpleNamespace(detect_players=lambda frame: [],
        detect_pitch_landmarks=lambda frame: PitchLandmarks(np.empty((0, 2)), np.empty((0, 2)), np.array([])))
    def callback(*args): raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        run_video_tracking(source, VideoTrackingConfig('clip', True), backend=backend,
            tracker=SimpleNamespace(update=lambda *a: []), work_dir=tmp_path / 'work', on_progress=callback)
    assert not list((tmp_path / 'work').iterdir())


@pytest.mark.parametrize('threshold,emitted,rejected', [(.05, 5, 0), (.25, 0, 5)])
def test_model_low_scores_reach_pipeline_confidence_gate(media, monkeypatch, tmp_path,
    threshold, emitted, rejected):
    from soccer_tactics.cv import pipeline
    from soccer_tactics.cv.roboflow_backend import RoboflowVideoBackend
    _, source = media
    class Tensor:
        def __init__(self, data): self.data = np.array(data)
        def cpu(self): return self
        def numpy(self): return self.data
    def model(frame, **kwargs):
        # Model double mirrors Ultralytics' threshold behavior, so its default
        # 0.25 would silently discard this player before pipeline diagnostics.
        keep = kwargs.get('conf', .25) <= .1
        return [SimpleNamespace(names={2: 'player'}, boxes=SimpleNamespace(
            xyxy=Tensor([[0, 0, 2, 4]] if keep else []),
            conf=Tensor([.1] if keep else []), cls=Tensor([2] if keep else [])))]
    backend = RoboflowVideoBackend(model, None)
    monkeypatch.setattr(backend, 'detect_pitch_landmarks', lambda frame:
        PitchLandmarks(np.zeros((4, 2)), np.zeros((4, 2)), np.ones(4)))
    monkeypatch.setattr(pipeline, 'estimate_calibration', lambda *a, **kw:
        CalibrationResult(True, np.eye(3), 4, 0., None))
    tracker = SimpleNamespace(update=lambda index, detections: [
        TrackedPlayer(index, '3', d.bbox_xyxy, d.confidence, d.class_name) for d in detections])
    result = pipeline.run_video_tracking(source, VideoTrackingConfig('clip', True,
        min_player_confidence=threshold), backend=backend, tracker=tracker, work_dir=tmp_path / 'work')
    assert len(result.detections) == emitted
    assert result.frame_diagnostics.low_confidence_players.sum() == rejected
