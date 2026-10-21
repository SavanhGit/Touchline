import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_missing_weights_rejected_before_model_import(tmp_path):
    from soccer_tactics.cv.roboflow_backend import RoboflowVideoBackend
    with pytest.raises(ValueError, match='weights|download_video_models'):
        RoboflowVideoBackend.from_weight_paths(tmp_path / 'a.pt', tmp_path / 'b.pt')


def test_local_backend_filters_nonplayers_and_maps_keypoints():
    from soccer_tactics.cv.roboflow_backend import RoboflowVideoBackend
    class Tensor:
        def __init__(self, data): self.data = np.array(data)
        def __len__(self): return len(self.data)
        def cpu(self): return self
        def numpy(self): return self.data
    boxes = SimpleNamespace(xyxy=Tensor([[0, 0, 2, 4]] * 4), conf=Tensor([.9] * 4), cls=Tensor([0, 1, 2, 3]))
    players = SimpleNamespace(boxes=boxes, names={0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'})
    pitch = SimpleNamespace(keypoints=SimpleNamespace(xy=Tensor([[[10, 20]] * 32]), conf=Tensor([[.8] * 32])))
    backend = RoboflowVideoBackend(lambda *a, **k: [players], lambda *a, **k: [pitch])
    assert [d.class_name for d in backend.detect_players(np.zeros((2, 2, 3)))] == ['goalkeeper', 'player']
    points = backend.detect_pitch_landmarks(np.zeros((2, 2, 3)))
    assert points.pitch_xy_m.shape == (32, 2)
    assert points.pitch_xy_m[0].tolist() == [-52.5, -34.]
    assert points.pitch_xy_m[29].tolist() == [52.5, 34.]


def test_tracker_preserves_clip_ids_and_filters_ball(monkeypatch):
    from soccer_tactics.cv.roboflow_backend import SupervisionByteTracker
    from soccer_tactics.cv.video_types import PlayerDetection
    class Detections:
        def __init__(self, **values): self.__dict__.update(values)
    class ByteTrack:
        def __init__(self, **kwargs): pass
        def update_with_detections(self, detections):
            detections.tracker_id = np.array([42] * len(detections.xyxy)); return detections
    monkeypatch.setitem(sys.modules, 'supervision', SimpleNamespace(Detections=Detections, ByteTrack=ByteTrack))
    tracker = SupervisionByteTracker(frame_rate=25)
    result = tracker.update(3, [PlayerDetection((1, 2, 3, 4), .8, 'player'), PlayerDetection((1, 2, 3, 4), .8, 'ball')])
    assert [(t.frame, t.track_id, t.class_name) for t in result] == [(3, '42', 'player')]


def test_explicit_downloader_fetches_two_weights_only(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('download_video_models', Path(__file__).parents[1] / 'scripts/download_video_models.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    def download(**kwargs):
        calls.append(kwargs['id']); Path(kwargs['output']).write_bytes(b'weights'); return kwargs['output']
    monkeypatch.setitem(sys.modules, 'gdown', SimpleNamespace(download=download))
    monkeypatch.setattr(sys, 'argv', ['download_video_models', '--output-dir', str(tmp_path)])
    module.main()
    assert len(calls) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ['football-pitch-detection.pt', 'football-player-detection.pt']
    module.main()
    assert len(calls) == 2
