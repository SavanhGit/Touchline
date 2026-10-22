"""Evaluate local tracking files; never retrieve videos, weights or annotations.

Predictions CSV: source-zero frame,track_id,x,y for pipeline pitch locations,
or frame,track_id,x1,y1,x2,y2 plus h00..h22 for projected box locations.
Optional --boxes provides image boxes for TrackEval when using pipeline x,y.
Optional --calibrations provides frame,h00..h22 (one row per frame).
Pitch reference CSV: frame,image_x,image_y,pitch_x,pitch_y, at least four
non-collinear known landmarks per sampled frame. Use centred pitch metres
before attack-direction rotation. Native SoccerNet annotations: headerless
10-column MOT, one-based frames and XYWH. For location comparisons, prediction
IDs must already match annotation IDs; no identity mapping is inferred.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from soccer_tactics.cv.evaluation import (
    H_COLUMNS, evaluate_calibration, read_soccernet_annotations, run_trackeval,
)


def _local_path(value: str) -> Path:
    if '://' in value or value.startswith(('\\\\', '//')):
        raise argparse.ArgumentTypeError('Use a local filesystem path; URLs and network shares are unsupported')
    return Path(value)


def _read_csv(path: Path) -> pd.DataFrame:
    # String identities preserve e.g. "007" instead of silently changing IDs.
    return pd.read_csv(path, dtype={'track_id': str})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ('predictions', 'annotations', 'pitch-references', 'trackeval', 'output'):
        parser.add_argument(f'--{option}', type=_local_path, required=True)
    parser.add_argument('--boxes', type=_local_path, help='Optional separate source-frame XYXY box CSV for TrackEval')
    parser.add_argument('--calibrations', type=_local_path, help='Optional frame,h00..h22 model homography CSV')
    parser.add_argument('--clip-id', required=True)
    parser.add_argument('--split', required=True)
    args = parser.parse_args(argv)
    try:
        if not args.clip_id.strip() or not args.split.strip():
            raise ValueError('clip-id and split must be nonempty labels')
        predictions = _read_csv(args.predictions)
        annotations = read_soccernet_annotations(args.annotations)
        references = _read_csv(args.pitch_references)
        if args.calibrations:
            calibrations = _read_csv(args.calibrations)
            if not {'frame', *H_COLUMNS}.issubset(calibrations.columns):
                raise ValueError('calibrations CSV requires frame,h00..h22 columns')
            if set(H_COLUMNS) & set(predictions.columns):
                raise ValueError('Supply model homographies in predictions or --calibrations, not both')
            predictions = predictions.merge(calibrations[['frame', *H_COLUMNS]], on='frame', how='left', validate='many_to_one')
        location = evaluate_calibration(predictions, annotations, references)
        tracking = run_trackeval(_read_csv(args.boxes) if args.boxes else predictions, annotations, args.trackeval)
        report = {'clip_id': args.clip_id, 'split': args.split, 'tracking': tracking,
                  'calibration_and_location': {key: None if isinstance(value, float) and not math.isfinite(value) else value
                                               for key, value in location.items()},
                  'location_reference': 'Annotation-derived reference from SoccerNet box footpoints and independent manual per-frame pitch calibration; not ground-truth player telemetry.',
                  'calibration_reference': 'Model transform estimate versus independent manually marked pitch landmarks; unavailable without model transforms.',
                  'evaluation_config': {'metrics': ['HOTA', 'CLEAR', 'Identity'], 'preprocessing': False,
                                        'clear_identity_iou_threshold': .5, 'metric_units': 'TrackEval summary percentages; counts for misses and switches',
                                        'annotation_scope': 'all supplied boxes; no automatic class filtering',
                                        'internal_frame_origin': 0, 'mot_frame_origin': 1,
                                        'pitch_convention': 'centred metres before attack-direction rotation'}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, RuntimeError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    print(f"{tracking['missed_detections']} missed detections; {tracking['identity_switches']} identity switches. Report: {args.output}")


if __name__ == '__main__':
    main()
