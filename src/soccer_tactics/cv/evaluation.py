"""Local SoccerNet/MOT evaluation and independent manual pitch references.

Internal boxes use source frame zero and columns frame,track_id,x1,y1,x2,y2.
Pitch locations use the pipeline's frame,track_id,x,y columns in centred pitch
metres, before attack-direction rotation. Optional h00..h22 columns contain the
model's image-to-pitch homography (one consistent matrix per frame). Manual CSV
columns are frame,image_x,image_y,pitch_x,pitch_y, in the same pitch convention.
Location errors are annotation-derived estimates, not player telemetry truth.
"""
from configparser import ConfigParser
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


BOX_COLUMNS = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2']
H_COLUMNS = [f'h{i}{j}' for i in range(3) for j in range(3)]
REFERENCE_COLUMNS = ['frame', 'image_x', 'image_y', 'pitch_x', 'pitch_y']


def _require(data: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f'{label} must be a DataFrame')
    if not data.columns.is_unique:
        raise ValueError(f'{label} has duplicate column names')
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f'{label} missing columns: {sorted(missing)}')
    return data.copy()


def _numeric(data: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(data[column], errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label} {column} must be numeric') from exc
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'{label} {column} must be finite')
        data[column] = values


def _frames(data: pd.DataFrame, label: str) -> None:
    _numeric(data, ['frame'], label)
    if (data.frame < 0).any():
        raise ValueError(f'{label} frame must be nonnegative')
    if (data.frame != np.floor(data.frame)).any():
        raise ValueError(f'{label} frame must be an integer')
    data['frame'] = data.frame.astype('int64')


def _identities(data: pd.DataFrame, label: str) -> None:
    if data.track_id.isna().any() or data.track_id.astype(str).str.strip().eq('').any():
        raise ValueError(f'{label} track_id must be a nonempty identity')
    data['track_id'] = data.track_id.astype(str)
    if data.duplicated(['frame', 'track_id']).any():
        raise ValueError(f'{label} has duplicate frame/track_id observations')


def _boxes(data: pd.DataFrame, label: str) -> pd.DataFrame:
    data = _require(data, BOX_COLUMNS, label)
    _frames(data, label)
    _identities(data, label)
    _numeric(data, BOX_COLUMNS[2:], label)
    if ((data.x2 <= data.x1) | (data.y2 <= data.y1)).any():
        raise ValueError(f'{label} boxes must have positive width and height')
    return data


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    # Homogeneous coordinates are equivalent under any nonzero scalar multiple.
    scale = np.max(np.abs(projected), axis=1)
    if np.any(np.abs(projected[:, 2]) <= np.finfo(float).eps * scale):
        raise ValueError('homography maps a reference or footpoint to infinity')
    result = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError('homography projects nonfinite coordinates')
    return result


def _manual_homography(image: np.ndarray, pitch: np.ndarray, frame: int) -> np.ndarray:
    """Normalized DLT keeps evaluation independent of the model/OpenCV fitter."""
    if len(image) < 4:
        raise ValueError(f'frame {frame}: at least four manual pitch landmarks are required')
    normalized = []
    transforms = []
    for points in (image, pitch):
        centre = points.mean(axis=0)
        centred = points - centre
        if np.linalg.matrix_rank(centred) < 2:
            raise ValueError(f'frame {frame}: manual landmarks must be non-collinear')
        scale = np.sqrt(2) / np.linalg.norm(centred, axis=1).mean()
        matrix = np.array([[scale, 0, -centre[0] * scale],
                           [0, scale, -centre[1] * scale], [0, 0, 1]])
        normalized.append(_project(points, matrix))
        transforms.append(matrix)
    equations = []
    for (u, v), (x, y) in zip(*normalized):
        equations.extend([[-u, -v, -1, 0, 0, 0, x*u, x*v, x],
                          [0, 0, 0, -u, -v, -1, y*u, y*v, y]])
    equations = np.asarray(equations)
    if np.linalg.matrix_rank(equations) < 8:
        raise ValueError(f'frame {frame}: degenerate manual landmark correspondences')
    _, _, right = np.linalg.svd(equations)
    matrix = np.linalg.inv(transforms[1]) @ right[-1].reshape(3, 3) @ transforms[0]
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f'frame {frame}: degenerate manual homography')
    return matrix / np.linalg.norm(matrix)


def _summary(errors: list[float], prefix: str) -> dict[str, float | int]:
    return {f'{prefix}_point_count': len(errors),
            f'{prefix}_median_error_m': float(np.median(errors)) if errors else float('nan'),
            f'{prefix}_p90_error_m': float(np.percentile(errors, 90)) if errors else float('nan')}


def evaluate_calibration(predicted: pd.DataFrame, ground_truth_boxes: pd.DataFrame,
                         pitch_references: pd.DataFrame) -> dict[str, float | int]:
    """Measure estimate-vs-reference errors without filling missing frames/IDs.

    Actual pipeline x,y locations take precedence over box projection. Otherwise
    model homographies project prediction bottom-centres. Reference locations
    always come from annotated box bottom-centres under independent manual
    per-frame calibration. IDs must already correspond; this function never
    guesses identity associations. No model matrices means calibration count 0
    and NaN calibration errors, while location evaluation remains available.
    """
    predicted = _require(predicted, ['frame', 'track_id'], 'predictions')
    _frames(predicted, 'predictions')
    _identities(predicted, 'predictions')
    if set(BOX_COLUMNS[2:]) & set(predicted.columns):
        predicted = _boxes(predicted, 'predictions')
    annotations = _boxes(ground_truth_boxes, 'annotations')
    refs = _require(pitch_references, REFERENCE_COLUMNS, 'pitch references')
    _frames(refs, 'pitch references')
    _numeric(refs, REFERENCE_COLUMNS[1:], 'pitch references')
    if refs.empty:
        raise ValueError('pitch references require at least four manual landmarks per sampled frame')
    has_matrices = bool(set(H_COLUMNS) & set(predicted.columns))
    if has_matrices:
        predicted = _require(predicted, H_COLUMNS, 'predictions homography')
        _numeric(predicted, H_COLUMNS, 'predictions homography')
    has_locations = bool({'x', 'y'} & set(predicted.columns))
    if has_locations:
        predicted = _require(predicted, ['x', 'y'], 'predictions pitch locations')
        _numeric(predicted, ['x', 'y'], 'predictions pitch locations')
    else:
        predicted = _boxes(predicted, 'predictions')
        if not has_matrices:
            raise ValueError('predictions require pitch x,y or boxes plus h00..h22 model homography columns')
    calibration_errors, location_errors = [], []
    for frame, landmarks in refs.groupby('frame', sort=True):
        image = landmarks[['image_x', 'image_y']].to_numpy(dtype=float)
        pitch = landmarks[['pitch_x', 'pitch_y']].to_numpy(dtype=float)
        reference_matrix = _manual_homography(image, pitch, frame)
        frame_predictions = predicted.loc[predicted.frame == frame]
        if frame_predictions.empty:
            continue
        matrix = None
        if has_matrices:
            matrices = frame_predictions[H_COLUMNS].to_numpy(dtype=float)
            matrix = matrices[0].reshape(3, 3)
            if np.linalg.matrix_rank(matrix) < 3:
                raise ValueError(f'frame {frame}: model homography must be invertible')
            if not np.allclose(matrices, matrices[0]):
                raise ValueError(f'frame {frame}: inconsistent model homographies')
            calibration_errors.extend(np.linalg.norm(_project(image, matrix) - pitch, axis=1).tolist())
        matched = frame_predictions.merge(annotations.loc[annotations.frame == frame],
                                          on=['frame', 'track_id'], suffixes=('', '_reference'), validate='one_to_one')
        if matched.empty:
            continue
        suffix = '_reference' if 'x1' in predicted.columns else ''
        feet = np.column_stack(((matched[f'x1{suffix}'] + matched[f'x2{suffix}']) / 2,
                                matched[f'y2{suffix}']))
        reference_locations = _project(feet, reference_matrix)
        if has_locations:
            locations = matched[['x', 'y']].to_numpy(dtype=float)
        else:
            feet = np.column_stack(((matched.x1 + matched.x2) / 2, matched.y2))
            locations = _project(feet, matrix)
        location_errors.extend(np.linalg.norm(locations - reference_locations, axis=1).tolist())
    return {**_summary(calibration_errors, 'calibration'), **_summary(location_errors, 'player_location')}


def read_soccernet_annotations(path: Path) -> pd.DataFrame:
    """Read the native headerless 10-column, one-based MOT XYWH annotations."""
    try:
        source = pd.read_csv(Path(path), header=None, dtype={1: str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('SoccerNet annotations require a nonempty CSV with 10 columns') from exc
    if source.shape[1] != 10:
        raise ValueError('SoccerNet annotations must have exactly 10 columns (frame,id,x,y,width,height,confidence,-1,-1,-1)')
    source.columns = ['frame', 'track_id', 'x1', 'y1', 'width', 'height', 'confidence', 'unused1', 'unused2', 'unused3']
    _numeric(source, ['frame', 'x1', 'y1', 'width', 'height', 'confidence'], 'SoccerNet annotations')
    if (source.frame < 1).any():
        raise ValueError('SoccerNet annotation frame must be a one-based positive integer')
    if not source.confidence.eq(1).all():
        raise ValueError('SoccerNet ground-truth confidence must be 1; provide annotated ground truth, not detections')
    source['frame'] -= 1
    source['x2'] = source.x1 + source.width
    source['y2'] = source.y1 + source.height
    return _boxes(source[BOX_COLUMNS], 'SoccerNet annotations')


def _mot_ids(data: pd.DataFrame) -> pd.Series:
    """Preserve integer identities; consistently encode arbitrary clip-local IDs."""
    values = pd.to_numeric(data.track_id, errors='coerce')
    if values.notna().all() and (values >= 0).all() and (values == np.floor(values)).all():
        if values.nunique() == data.track_id.nunique():
            return values.astype('int64')
    mapping = {identity: index + 1 for index, identity in enumerate(sorted(data.track_id.unique()))}
    return data.track_id.map(mapping)


def write_trackeval_mot_files(predictions: pd.DataFrame, annotations: pd.DataFrame,
                             output_root: Path) -> Path:
    """Write one MOT sequence; caller owns this directory (use a temporary path).

    All supplied annotations are evaluated, with no automatic class filtering.
    SoccerNet's unused class columns are converted to MOT pedestrian class 1.
    Numeric IDs retain their value; arbitrary strings are encoded consistently
    within each independent identity space. Frame zero becomes MOT frame one.
    """
    predictions = _boxes(predictions, 'predictions')
    annotations = _boxes(annotations, 'annotations')
    if annotations.empty:
        raise ValueError('annotations must contain at least one ground-truth box')
    if 'confidence' in predictions:
        _numeric(predictions, ['confidence'], 'predictions')
        if not predictions.confidence.between(0, 1).all():
            raise ValueError('predictions confidence must be between 0 and 1')
    root = Path(output_root)
    gt_dir = root / 'gt/clip/gt'
    tracker_dir = root / 'trackers/touchline/data'
    gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    for data, destination, is_gt in [(annotations, gt_dir / 'gt.txt', True),
                                     (predictions, tracker_dir / 'clip.txt', False)]:
        output = pd.DataFrame({'frame': data.frame + 1, 'id': _mot_ids(data),
                               'x': data.x1, 'y': data.y1, 'width': data.x2 - data.x1,
                               'height': data.y2 - data.y1, 'confidence': 1 if is_gt else data.get('confidence', 1)})
        output['class' if is_gt else 'unused1'] = 1 if is_gt else -1
        output['visibility' if is_gt else 'unused2'] = 1 if is_gt else -1
        if not is_gt:
            output['unused3'] = -1
        output.sort_values(['frame', 'id']).to_csv(destination, index=False, header=False)
    max_frame = max(int(annotations.frame.max()), int(predictions.frame.max()) if not predictions.empty else 0)
    config = ConfigParser()
    config.optionxform = str
    config['Sequence'] = {'name': 'clip', 'seqLength': str(max_frame + 1)}
    with (root / 'gt/clip/seqinfo.ini').open('w', encoding='utf-8') as file:
        config.write(file)
    return root


def run_trackeval(predictions: pd.DataFrame, annotations: pd.DataFrame,
                  trackeval_checkout: Path) -> dict[str, float | int]:
    """Invoke the supplied official checkout, read its summary, remove all MOT data."""
    script = Path(trackeval_checkout).resolve() / 'scripts/run_mot_challenge.py'
    if not script.is_file():
        raise ValueError('Local TrackEval checkout must contain scripts/run_mot_challenge.py')
    with TemporaryDirectory(prefix='touchline-trackeval-') as temporary:
        root = write_trackeval_mot_files(predictions, annotations, Path(temporary))
        command = [sys.executable, str(script), '--GT_FOLDER', str(root / 'gt'),
                   '--TRACKERS_FOLDER', str(root / 'trackers'), '--OUTPUT_FOLDER', str(root / 'results'),
                   '--TRACKERS_TO_EVAL', 'touchline', '--SEQ_INFO', 'clip', '--SKIP_SPLIT_FOL', 'True',
                   '--DO_PREPROC', 'False', '--METRICS', 'HOTA', 'CLEAR', 'Identity',
                   '--USE_PARALLEL', 'False', '--OUTPUT_SUMMARY', 'True', '--PLOT_CURVES', 'False',
                   '--THRESHOLD', '0.5']
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=script.parent.parent, timeout=300)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'TrackEval failed: {exc.stderr or exc.stdout or exc}') from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'Cannot complete local TrackEval: {exc}') from exc
        path = root / 'results/touchline/pedestrian_summary.txt'
        fields = {'HOTA': 'HOTA', 'DetA': 'DetA', 'AssA': 'AssA', 'CLR_Re': 'detection_recall_percent',
                  'CLR_FN': 'missed_detections', 'IDSW': 'identity_switches', 'IDF1': 'IDF1'}
        if not path.is_file():
            raise RuntimeError('TrackEval did not produce pedestrian_summary.txt; inspect the local checkout/dependencies')
        try:
            summary = pd.read_csv(path, sep=r'\s+')
            if len(summary) != 1 or not set(fields).issubset(summary.columns):
                raise ValueError('missing summary metrics or unexpected row count')
            metrics = {}
            for source, target in fields.items():
                value = float(summary.iloc[0][source])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'invalid {source}')
                if source in ('CLR_FN', 'IDSW'):
                    if not value.is_integer():
                        raise ValueError(f'{source} must be an integer count')
                    metrics[target] = int(value)
                else:
                    metrics[target] = value
            return metrics
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise RuntimeError(f'Malformed TrackEval summary: {exc}') from exc
