"""Local evaluation fixtures use hand-derived values and no external assets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from soccer_tactics.cv.evaluation import (
    evaluate_calibration, read_soccernet_annotations, run_trackeval,
    write_trackeval_mot_files,
)


def sample():
    # Manual frame 0: image / 2. Frame 1: image / 4.
    references = pd.DataFrame([
        (frame, ix * scale, iy * scale, ix, iy)
        for frame, scale in [(0, 2), (1, 4)]
        for ix, iy in [(0, 0), (10, 0), (10, 10), (0, 10)]
    ], columns=['frame', 'image_x', 'image_y', 'pitch_x', 'pitch_y'])
    annotations = pd.DataFrame([
        (0, '7', 8, 2, 12, 10), (1, '7', 16, 4, 24, 20),
    ], columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
    predicted = pd.DataFrame([
        (0, '7', 12, 2, 16, 10), (1, '7', 32, 4, 40, 20),
    ], columns=annotations.columns)
    for index, value in enumerate(np.eye(3).reshape(-1)):
        predicted[f'h{index // 3}{index % 3}'] = value
    predicted['h00'] = [.5, .25]
    predicted['h11'] = [.5, .25]
    predicted['h02'] = [1, 3]
    return predicted, annotations, references


def cli_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_video_tracking.py'
    spec = importlib.util.spec_from_file_location('evaluate_video_tracking', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_frame_independent_calibration_and_annotation_footpoint_errors():
    predicted, annotations, references = sample()
    result = evaluate_calibration(predicted, annotations, references)
    # Four landmark errors of 1 m and four of 3 m; player errors 3 m, 7 m.
    assert result['calibration_point_count'] == 8
    assert result['calibration_median_error_m'] == pytest.approx(2)
    assert result['calibration_p90_error_m'] == pytest.approx(3)
    assert result['player_location_point_count'] == 2
    assert result['player_location_median_error_m'] == pytest.approx(5)
    assert result['player_location_p90_error_m'] == pytest.approx(6.6)


def test_actual_pipeline_locations_do_not_require_model_homography_or_boxes():
    _, annotations, references = sample()
    predicted = pd.DataFrame({'frame': [0, 1], 'track_id': ['7', '7'], 'x': [8, 12], 'y': [5, 5]})
    result = evaluate_calibration(predicted, annotations, references)
    assert result['player_location_median_error_m'] == pytest.approx(5)
    assert result['calibration_point_count'] == 0
    assert np.isnan(result['calibration_median_error_m'])


def test_only_exact_frame_and_annotation_id_matches_contribute_without_fill():
    predicted, annotations, references = sample()
    predicted.loc[1, 'track_id'] = 'different'
    result = evaluate_calibration(predicted, annotations, references)
    assert result['player_location_point_count'] == 1
    assert result['player_location_median_error_m'] == pytest.approx(3)


@pytest.mark.parametrize('mutation,match', [
    (lambda refs: refs.drop(columns='image_x'), 'missing.*image_x'),
    (lambda refs: refs.assign(image_x='bad'), 'image_x.*numeric'),
    (lambda refs: refs.assign(pitch_x=np.nan), 'pitch_x.*finite'),
    (lambda refs: refs.iloc[:3], 'four.*landmarks'),
    (lambda refs: refs.assign(image_y=0), 'non.collinear|degenerate'),
    (lambda refs: refs.assign(frame=.5), 'frame.*integer'),
])
def test_malformed_manual_reference_columns_fail_clearly(mutation, match):
    predicted, annotations, references = sample()
    with pytest.raises(ValueError, match=match):
        evaluate_calibration(predicted, annotations, mutation(references))


def test_inconsistent_or_partial_model_transforms_fail_clearly():
    predicted, annotations, references = sample()
    with pytest.raises(ValueError, match='missing.*h22'):
        evaluate_calibration(predicted.drop(columns='h22'), annotations, references)
    duplicate = predicted.iloc[[0]].copy()
    duplicate['track_id'] = '8'
    duplicate['h02'] = 99
    with pytest.raises(ValueError, match='frame 0.*inconsistent'):
        evaluate_calibration(pd.concat([predicted, duplicate]), annotations, references)


def test_mot_export_preserves_numeric_id_converts_xyxy_and_frames(tmp_path):
    predicted, annotations, _ = sample()
    root = write_trackeval_mot_files(predicted, annotations, tmp_path / 'mot')
    output = pd.read_csv(root / 'trackers/touchline/data/clip.txt', header=None)
    assert output.iloc[0].tolist() == [1, 7, 12, 2, 4, 8, 1, -1, -1, -1]
    assert output.iloc[1].tolist() == [2, 7, 32, 4, 8, 16, 1, -1, -1, -1]
    gt = pd.read_csv(root / 'gt/clip/gt/gt.txt', header=None)
    assert gt.iloc[0].tolist() == [1, 7, 8, 2, 4, 8, 1, 1, 1]
    assert 'seqLength = 2' in (root / 'gt/clip/seqinfo.ini').read_text()


@pytest.mark.parametrize('change,match', [
    (lambda data: data.assign(frame=-1), 'frame.*nonnegative'),
    (lambda data: data.assign(x2=data.x1), 'positive.*width'),
    (lambda data: pd.concat([data, data]), 'duplicate.*frame'),
    (lambda data: data.assign(track_id=None), 'track_id'),
])
def test_invalid_boxes_or_duplicate_id_frames_fail_before_export(tmp_path, change, match):
    predicted, annotations, _ = sample()
    with pytest.raises(ValueError, match=match):
        write_trackeval_mot_files(change(predicted), annotations, tmp_path / 'mot')
    assert not (tmp_path / 'mot').exists()


def test_native_soccernet_mot_annotations_are_converted_to_source_frame_xyxy(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,7,8,2,4,8,1,-1,-1,-1\n2,7,16,4,8,16,1,-1,-1,-1\n')
    actual = read_soccernet_annotations(path)
    _, expected, _ = sample()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_soccernet_malformed_column_count_has_actionable_error(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,7,8,2,4\n')
    with pytest.raises(ValueError, match='10 columns'):
        read_soccernet_annotations(path)


def test_soccernet_id_strings_are_not_silently_normalized(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('1,007,8,2,4,8,1,-1,-1,-1\n')
    assert read_soccernet_annotations(path).track_id.tolist() == ['007']


def test_partial_prediction_box_columns_fail_clearly_even_with_pitch_locations():
    _, annotations, references = sample()
    predicted = pd.DataFrame({'frame': [0], 'track_id': ['7'], 'x': [8], 'y': [5], 'x1': [1]})
    with pytest.raises(ValueError, match='missing.*x2'):
        evaluate_calibration(predicted, annotations, references)


def fake_trackeval_checkout(tmp_path):
    checkout = tmp_path / 'TrackEval'
    (checkout / 'scripts').mkdir(parents=True)
    (checkout / 'scripts/run_mot_challenge.py').write_text('# local fake entry point\n')
    return checkout


def fake_trackeval_success(command, **kwargs):
    assert command[0] == sys.executable
    assert command[command.index('--METRICS') + 1:command.index('--METRICS') + 4] == ['HOTA', 'CLEAR', 'Identity']
    assert command[command.index('--DO_PREPROC') + 1] == 'False'
    assert command[command.index('--SKIP_SPLIT_FOL') + 1] == 'True'
    assert kwargs['check'] and kwargs['capture_output'] and kwargs['text']
    root = Path(command[command.index('--OUTPUT_FOLDER') + 1])
    (root / 'touchline').mkdir(parents=True)
    (root / 'touchline/pedestrian_summary.txt').write_text(
        'HOTA DetA AssA CLR_Re CLR_FN IDSW IDF1\n60 50 72 80 2 1 75\n')
    return subprocess.CompletedProcess(command, 0, stdout='TrackEval complete', stderr='')


def test_official_trackeval_invocation_reads_metrics_and_cleans_temporary_data(tmp_path, monkeypatch):
    predicted, annotations, _ = sample()
    checkout = fake_trackeval_checkout(tmp_path)
    roots = []

    def fake(command, **kwargs):
        roots.append(Path(command[command.index('--GT_FOLDER') + 1]).parent)
        return fake_trackeval_success(command, **kwargs)

    monkeypatch.setattr(subprocess, 'run', fake)
    result = run_trackeval(predicted, annotations, checkout)
    assert result == {'HOTA': 60., 'DetA': 50., 'AssA': 72., 'detection_recall_percent': 80.,
                      'missed_detections': 2, 'identity_switches': 1, 'IDF1': 75.}
    assert len(roots) == 1 and not roots[0].exists()


def test_trackeval_subprocess_failure_is_actionable(tmp_path, monkeypatch):
    predicted, annotations, _ = sample()
    checkout = fake_trackeval_checkout(tmp_path)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr='missing scipy')
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(RuntimeError, match='TrackEval.*missing scipy'):
        run_trackeval(predicted, annotations, checkout)


def test_cli_evaluates_local_files_exports_labelled_json(tmp_path, monkeypatch, capsys):
    predicted, annotations, refs = sample()
    predicted.to_csv(tmp_path / 'predictions.csv', index=False)
    refs.to_csv(tmp_path / 'refs.csv', index=False)
    (tmp_path / 'gt.txt').write_text('1,7,8,2,4,8,1,-1,-1,-1\n2,7,16,4,8,16,1,-1,-1,-1\n')
    checkout = fake_trackeval_checkout(tmp_path)
    monkeypatch.setattr(subprocess, 'run', fake_trackeval_success)
    report_path = tmp_path / 'report.json'
    cli_module().main(['--predictions', str(tmp_path / 'predictions.csv'), '--annotations', str(tmp_path / 'gt.txt'),
                       '--pitch-references', str(tmp_path / 'refs.csv'), '--trackeval', str(checkout),
                       '--clip-id', 'fixture', '--split', 'train', '--output', str(report_path)])
    report = json.loads(report_path.read_text())
    assert report['clip_id'] == 'fixture' and report['split'] == 'train'
    assert report['tracking']['identity_switches'] == 1
    assert report['calibration_and_location']['player_location_p90_error_m'] == pytest.approx(6.6)
    assert 'annotation-derived' in report['location_reference'].lower()
    assert 'telemetry' in report['location_reference']
    assert '2 missed detections' in capsys.readouterr().out


def test_cli_rejects_remote_input_before_reading_or_invoking(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli_module().main(['--predictions', 'https://example.com/predictions.csv', '--annotations', 'gt.txt',
                           '--pitch-references', 'refs.csv', '--trackeval', 'TrackEval',
                           '--clip-id', 'fixture', '--split', 'train', '--output', str(tmp_path / 'report.json')])
    assert exc.value.code == 2


def test_cli_pipeline_locations_use_separate_boxes_and_report_unavailable_calibration(tmp_path, monkeypatch):
    boxes, _, refs = sample()
    pd.DataFrame({'frame': [0, 1], 'track_id': ['7', '7'], 'x': [8, 12], 'y': [5, 5]}).to_csv(tmp_path / 'locations.csv', index=False)
    boxes.to_csv(tmp_path / 'boxes.csv', index=False)
    refs.to_csv(tmp_path / 'refs.csv', index=False)
    (tmp_path / 'gt.txt').write_text('1,7,8,2,4,8,1,-1,-1,-1\n2,7,16,4,8,16,1,-1,-1,-1\n')
    checkout = fake_trackeval_checkout(tmp_path)
    monkeypatch.setattr(subprocess, 'run', fake_trackeval_success)
    report_path = tmp_path / 'report.json'
    cli_module().main(['--predictions', str(tmp_path / 'locations.csv'), '--boxes', str(tmp_path / 'boxes.csv'),
                       '--annotations', str(tmp_path / 'gt.txt'), '--pitch-references', str(tmp_path / 'refs.csv'),
                       '--trackeval', str(checkout), '--clip-id', 'fixture', '--split', 'train', '--output', str(report_path)])
    report = json.loads(report_path.read_text())['calibration_and_location']
    assert report['calibration_median_error_m'] is None
    assert report['calibration_point_count'] == 0
    assert report['player_location_median_error_m'] == pytest.approx(5)


def test_trackeval_missing_or_malformed_summary_fails_instead_of_fabricating_metrics(tmp_path, monkeypatch):
    predicted, annotations, _ = sample()
    checkout = fake_trackeval_checkout(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    with pytest.raises(RuntimeError, match='did not produce'):
        run_trackeval(predicted, annotations, checkout)
    def malformed(command, **kwargs):
        result = fake_trackeval_success(command, **kwargs)
        root = Path(command[command.index('--OUTPUT_FOLDER') + 1])
        (root / 'touchline/pedestrian_summary.txt').write_text('HOTA DetA AssA CLR_Re CLR_FN IDSW IDF1\n60 50 72 80 2.5 1 75\n')
        return result
    monkeypatch.setattr(subprocess, 'run', malformed)
    with pytest.raises(RuntimeError, match='integer count'):
        run_trackeval(predicted, annotations, checkout)


def test_cli_separate_model_calibrations_do_not_override_actual_pipeline_locations(tmp_path, monkeypatch):
    boxes, _, refs = sample()
    # Real pipeline locations are deliberately perfect; model landmark errors
    # stay 1 m / 3 m. Applying model transforms to the boxes would produce 3/7 m.
    pd.DataFrame({'frame': [0, 1], 'track_id': ['7', '7'], 'x': [5, 5], 'y': [5, 5]}).to_csv(tmp_path / 'locations.csv', index=False)
    boxes.to_csv(tmp_path / 'boxes.csv', index=False)
    boxes[['frame', *[f'h{i}{j}' for i in range(3) for j in range(3)]]].to_csv(tmp_path / 'calibrations.csv', index=False)
    refs.to_csv(tmp_path / 'refs.csv', index=False)
    (tmp_path / 'gt.txt').write_text('1,7,8,2,4,8,1,-1,-1,-1\n2,7,16,4,8,16,1,-1,-1,-1\n')
    checkout = fake_trackeval_checkout(tmp_path)
    monkeypatch.setattr(subprocess, 'run', fake_trackeval_success)
    report_path = tmp_path / 'report.json'
    cli_module().main(['--predictions', str(tmp_path / 'locations.csv'), '--boxes', str(tmp_path / 'boxes.csv'),
                       '--calibrations', str(tmp_path / 'calibrations.csv'), '--annotations', str(tmp_path / 'gt.txt'),
                       '--pitch-references', str(tmp_path / 'refs.csv'), '--trackeval', str(checkout),
                       '--clip-id', 'fixture', '--split', 'train', '--output', str(report_path)])
    report = json.loads(report_path.read_text())['calibration_and_location']
    assert report['calibration_median_error_m'] == pytest.approx(2)
    assert report['player_location_median_error_m'] == pytest.approx(0, abs=1e-10)
