# V5 local player-tracking evaluation

**Real-footage acceptance gate: pending (2026-10-21).** The local MP4 workspace, manual mapping/export and evaluator are implemented. No real-model run, SoccerNet metrics, manual-pitch measurements or completed representative visual review are recorded here. Unit tests and a generated synthetic MP4 do not establish real-footage accuracy. No accuracy threshold is claimed before measuring the first baseline.

The [V5 design](../../superpowers/specs/2026-10-21-v5-local-player-tracking-design.md) defines the gate. See [setup, Windows Python 3.15 workaround and model terms](../../../README.md#optional-local-video-player-tracking-v5), the [roadmap](../../ROADMAP.md#v5--video-derived-player-tracking), and [release validation context](../../VALIDATION.md#v5-local-player-tracking-2026-10-21). URL import, ball tracking, event extraction, and live analysis are not implemented. There is no video-derived event table, score, tactical match report or assistant evidence.

## Local inputs and storage

Obtain an appropriately licensed short, continuous [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking) clip and its annotations locally under the dataset's current access/use terms. The intended initial target is approximately 30 seconds with a single attack direction. Record the clip and split labels, source hash, annotation provenance and any class filtering. Do not treat dataset access as permission to redistribute footage or annotations.

Use a local checkout of official [TrackEval](https://github.com/JonathonLuiten/TrackEval), containing `scripts/run_mot_challenge.py`. Install that checkout's required dependencies into the evaluation environment following its setup instructions. Touchline invokes it through the active Python interpreter. The evaluator never fetches videos, annotations, weights or TrackEval; URLs and network shares are rejected by its CLI.

Keep source videos, MOT annotations, manually marked pitch points and the external checkout under ignored `data/raw/` or external local storage. Weights belong under ignored `data/models/video/`. Predicted CSVs, evaluation scratch and generated runs belong under ignored `data/processed/`. A saved UI run lives at `data/processed/video_runs/<run_id>/` and contains `tracking.parquet`, `frame_diagnostics.parquet`, `metadata.json`, and `overlay.mp4`; the ZIP exports the same files. The overlay contains source imagery and must stay local too. Do not commit raw video/weights/annotation assets. Only a reviewed aggregate JSON report and written acceptance notes may later be added here, subject to source terms.

The upstream [Roboflow soccer example](https://github.com/roboflow/sports/blob/main/examples/soccer/README.md) attributes its YOLOv8 models to AGPL-3.0 and Supervision-based analytics code to MIT. Its [setup script](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh) downloads the player and pitch checkpoints from Google Drive; exact binary terms are not clearly published there and are not independently verified here. Dataset CC BY 4.0 labeling does not establish the terms for those checkpoints. No weights are redistributed; the main setup notes provide the separate code/package/model attribution.

## Prepare evaluation CSVs

All internal prediction and manual-reference frames are **zero-based decoded source indices**, with no resampling or filling of missing observations. Pitch coordinates use centered 105 × 68 m metres **before attack-direction rotation**: x spans −52.5 to +52.5 and y spans −34 to +34. Manual reference points must use that same orientation and scale. Image coordinates are source-frame pixels.

| Input | Shape and meaning |
|---|---|
| `--predictions` | Headered `frame,track_id,x,y`, the pipeline's projected pitch locations; optional model homography columns can be added. |
| `--boxes` | Headered `frame,track_id,x1,y1,x2,y2`, source-image XYXY boxes for TrackEval; optional `confidence` in [0,1]. Required when predictions supply pitch x/y without image boxes. |
| `--calibrations` | Headered `frame,h00,h01,h02,h10,h11,h12,h20,h21,h22`, one image-to-centered-pitch model homography per prediction frame, row-major. Optional; alternatively put all nine columns on each prediction row. Do not supply both forms. |
| `--pitch-references` | Headered `frame,image_x,image_y,pitch_x,pitch_y`; at least four independent, non-collinear known pitch landmarks for each sampled frame. |
| `--annotations` | Native SoccerNet headerless ten-column MOT: `frame,id,x,y,width,height,confidence,-1,-1,-1`. Frames are one-based; ground-truth confidence must be 1. Touchline converts frames to zero-based and XYWH boxes to XYXY. |

An alternative predictions CSV contains `frame,track_id,x1,y1,x2,y2` plus all nine `h00` through `h22` model homography columns (or a separate `--calibrations` file). In that form Touchline projects box bottom-centres to obtain pitch locations and uses the same boxes for TrackEval. When x/y are also present, those actual pipeline locations take precedence for location error.

Current saved pipeline runs **do not retain image boxes or per-frame transforms**. You must capture/provide those separate local CSVs for TrackEval and model-calibration metrics; the app export alone cannot reproduce every metric. Do not recover boxes or transforms by estimating them from the rendered overlay. `frame_diagnostics.parquet` contains model-fit quality summaries, not homography matrices. Without model matrices, calibration point count is zero and calibration errors are unavailable (`null` in CLI JSON), rather than zero error. The CLI still requires image boxes for its TrackEval step.

Record processing thresholds separately with the acceptance evidence: current saved metadata does not persist them. UI defaults are player/keypoint confidence 0.25, at least four calibration inliers and at most 12 px model-fit reprojection error. That pixel diagnostic is not the independent manual-pitch error in metres. Model/package provenance, when supplied by the backend, records checkpoint digests rather than local weight paths.

For location-only input preparation, use the pipeline's pre-rotation `result.detections` columns. If converting a saved `tracking.parquet`, rename `player_id` to `track_id`, retain `frame,x,y`, and undo the whole-clip rotation: when `metadata.json` has `home_attacks_positive_x=false`, multiply **both** x and y by −1. Export only one run/period at a time. Save the conversion under ignored processed storage. A saved export contains only manually included tracks, so record its exclusion policy when comparing detection coverage.

For location error, prediction track IDs must already correspond exactly to the annotation IDs (including string IDs such as `007`); the evaluator does not infer identity mapping. Document any manually established mapping. TrackEval evaluates association in its own predicted/ground-truth identity spaces; matching ID text is not required for those tracking metrics. All supplied annotation boxes are evaluated, with no automatic class filtering; explicitly record any player/goalkeeper-only filtering applied beforehand.

Mark pitch points independently from the model, spanning each sampled view. Each sampled frame gets its own reference homography; no stale transform is reused across camera movement. Ground-truth player reference positions are annotated box bottom-centres projected through that independent manual homography. Player-location errors are **annotation-derived estimates, not ground-truth player telemetry**. Footpoint ambiguity, occlusion, lens distortion, reference-point placement and the planar pitch assumption all affect them.

## Run the evaluator

After explicitly installing `.[dev,dashboard,cv]`, preparing the local files above and installing local TrackEval dependencies, run from the repository root (replace filenames and labels with the actual clip):

```powershell
python scripts/evaluate_video_tracking.py `
  --predictions data/processed/video_eval/predictions.csv `
  --boxes data/processed/video_eval/boxes.csv `
  --calibrations data/processed/video_eval/calibrations.csv `
  --annotations data/raw/soccernet/clip/gt.txt `
  --pitch-references data/raw/soccernet/clip/pitch-references.csv `
  --trackeval data/raw/TrackEval `
  --clip-id YOUR_CLIP_ID --split YOUR_SPLIT `
  --output data/processed/video_eval/evaluation.json
```

Use `& $cvPython` in place of `python` for the README's Windows `.venv-cv` environment. Omit `--calibrations` if transforms are absent, recording calibration as unavailable. Omit `--boxes` only when `--predictions` already contains image boxes. These are separate supplied artifacts; no command in this guide claims the app currently exports them.

Touchline writes temporary MOTChallenge inputs for the local official evaluator and removes them afterward. It runs HOTA, CLEAR and Identity metrics, disables preprocessing (`DO_PREPROC=False`), uses a 0.5 IoU threshold for CLEAR/Identity, and evaluates all supplied boxes as pedestrian class 1. This is a local SoccerNet-to-MOTChallenge comparison through TrackEval, not a submitted official SoccerNet leaderboard result. Record TrackEval's commit/version and the exact population/configuration before comparing scores elsewhere.

The JSON includes `clip_id`, `split`, `tracking`, `calibration_and_location`, reference descriptions and evaluation configuration. Tracking reports **HOTA, DetA, AssA, IDF1**, detection recall (summary percentages), **missed detections** and **identity switches** (counts). Calibration and player location each report point count, median error and p90 error in metres. Only exact frame/ID matches at manually sampled frames contribute to location error; missing predictions are reflected by coverage/tracking metrics, never replaced with invented positions. Keep denominators and sampled frame selection with the report. Do not interpret a small error on a few matched points as complete tracking coverage.

## Evidence required before closing the gate

| Acceptance evidence | Current status |
|---|---|
| Native Windows Python 3.15 CV install, `pip check` and real inference | Pending; only cross-target dependency resolution on bundled Python 3.12.14 has been demonstrated with the README's PyYAML/CPU-wheel workaround. |
| Real local clip identity, split, source hash, model digests/package versions and processing settings | Pending; no clip or model weights were available for a real acceptance run. |
| Manual team/name assignment and canonical saved export from that run | Pending; automated coverage exercises fake-model mapping/exclusion and persistence. |
| Local official TrackEval output, misses, ID switches and detection/association metrics | Pending; no local TrackEval checkout/annotations were available for measured evaluation. |
| Independently marked pitch points, calibration/location errors and observation counts | Pending; no real measured errors are recorded. |
| Representative frame review plus real browser overlay playback | Pending; AppTest verifies artifact controls, not browser video decoding. |

Visual review must inspect opening/middle/ending frames, camera movement or zoom where present, boxes, track IDs, pitch projection, missing/calibration-failed frames and selected attack orientation. Record concrete defects, browser playback behavior and reviewer/date, with only source-permitted artifacts. A rendered overlay alone does not pass this review.

The implementation estimates calibration independently per frame and preserves gaps for rejected confidence, insufficient landmarks, off-pitch projections, occlusions and off-camera players. It does not interpolate positions or repair identity switches. Track IDs stay clip-local; assigning the same roster name to multiple IDs does not merge their series. The Roboflow 120 × 70 m keypoint geometry normalized to Touchline's 105 × 68 m pitch is an assumption still needing validation. Full-match/camera-cut handling, automatic identity/team recognition and performance/latency claims are deferred.

Automated reproduction uses `python -m pytest tests/test_roadmap.py -q` for documentation consistency and `python -m pytest -q` for the suite. Record the actual results and environment separately from the acceptance gate. Missing source files, skipped optional dependencies or fake-model success must not be called passing real-video acceptance.

The 2026-10-21 documentation checkpoint used bundled Python 3.12.14 with the existing ignored Task 3/4 verification dependencies. Focused documentation tests passed **11 tests**. The complete suite reported **538 passed, 29 skipped, 1 failed**; the failure was `tests/test_release_validation.py::test_real_export_is_finite_and_has_other_match_training_only`, because `data/raw/metrica/data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv` is absent from this checkout. That is a missing existing-release input, not a passing release gate. No videos, annotations, checkpoints or TrackEval checkout were available for real V5 acceptance.
