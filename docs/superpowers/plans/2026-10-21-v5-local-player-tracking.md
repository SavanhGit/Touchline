# V5 Local Video Player Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a short continuous local soccer clip into reviewed, manually mapped player-tracking rows in Touchline's canonical format.

**Architecture:** Keep the inference pipeline independent from Streamlit and expose typed detector, tracker, calibration, and result boundaries in `soccer_tactics.cv`. The local Roboflow/Ultralytics/Supervision adapter emits per-frame player tracks and pitch projections; a mapping/export layer validates canonical rows and saves ignored run artifacts; Streamlit handles local file selection, configuration, progress, mapping, review, and export.

**Tech Stack:** Python 3.15 on Windows, pandas/NumPy, optional OpenCV/Ultralytics/Supervision, Streamlit AppTest, pytest, Parquet, TrackEval for the real SoccerNet run.

**Spec:** `docs/superpowers/specs/2026-10-21-v5-local-player-tracking-design.md`

## Global Constraints

- Process one local, continuous clip of approximately 30 seconds; do not add URL retrieval, camera-cut handling, ball tracking, event extraction, or live analysis.
- Run inference locally with no paid API; obtain model weights only through an explicit setup action, never during app startup or processing.
- Preserve the canonical `[match_id, period, frame, time_s, team, player_id, x, y]` schema; one clip is period 1 and home attacks toward +x after orientation.
- Map track IDs manually to `home` or `away`; roster names are optional sidecar labels, and raw tracker IDs remain in canonical `player_id` values.
- Estimate image-to-pitch calibration independently on each processed frame; omit and diagnose invalid positions rather than interpolate them or reuse a stale transform.
- Use a 105 m by 68 m pitch; keep source video, annotations, and model weights out of Git; keep generated runs ignored by Git; never persist source paths in saved metadata.
- Store processed runs under `data/processed/video_runs/<run_id>/`; keep video inference dependencies optional and keep the existing Metrica/IDSSE dashboard usable without them.
- Document Roboflow/Ultralytics and exact weight licensing separately; do not commit model weights.
- Verify the optional CV stack on Windows/Python 3.15 before changing the project's Python requirement.

## Review Focus

- A corrupt video, unsupported codec, zero/nonfinite frame rate, or unreadable frame gives an actionable UI error and creates no partial saved run. Test in Task 4.
- A camera move, zoom, missing pitch landmarks, or degenerate landmark geometry invalidates only the affected frame; no previous transform is reused. Test in Tasks 1 and 3.
- Low-confidence detections, failed calibration, and player footpoints outside the pitch are omitted with diagnostics; no interpolated row appears. Test in Tasks 1 and 3.
- Unassigned tracks are excluded from canonical export; separate IDs mapped to the same roster name remain separate tracks and are never silently merged. Test in Tasks 2 and 4.
- Saved metadata contains no source path/video/annotation, and the run output and model files remain ignored by Git. Test in Task 2.

---

### Task 1: Define video contracts and per-frame pitch calibration

**Files:**
- Create: `src/soccer_tactics/cv/video_types.py`
- Create: `src/soccer_tactics/cv/calibration.py`
- Test: `tests/test_cv_calibration.py`

**Interfaces:**
- Produces `PitchLandmarks(image_xy: np.ndarray, pitch_xy_m: np.ndarray, confidence: np.ndarray)`.
- Produces `CalibrationResult(valid: bool, image_to_pitch: np.ndarray | None, inlier_count: int, median_error_px: float | None, reason: str | None)`.
- Produces `VideoTrackingConfig(match_id: str, home_attacks_positive_x: bool, min_player_confidence: float = 0.25, min_pitch_keypoint_confidence: float = 0.25, min_calibration_inliers: int = 4, max_reprojection_error_px: float = 12.0)`.
- Produces `PlayerDetection(bbox_xyxy: tuple[float, float, float, float], confidence: float, class_name: str)` and `TrackedPlayer(frame: int, track_id: str, bbox_xyxy: tuple[float, float, float, float], confidence: float, class_name: str)`; tracker output retains the image-space box needed for pitch footpoints and overlays.
- Produces `VideoTrackingResult(detections: pd.DataFrame, frame_diagnostics: pd.DataFrame, video_metadata: dict[str, object], overlay_path: Path)`.
- Produces `estimate_calibration(landmarks: PitchLandmarks, *, min_inliers: int = 4, max_reprojection_error_px: float = 12.0) -> CalibrationResult` and `project_points(points_xy: np.ndarray, calibration: CalibrationResult) -> np.ndarray`.
- Calibration maps source image pixel coordinates to centered pitch metres, with pitch limits x=[-52.5, 52.5], y=[-34, 34]. The calibrated transform is frame-specific.

- [ ] **Step 1: Write failing geometry and validation tests**

In `tests/test_cv_calibration.py`, use `pytest.importorskip("cv2")` because the CV package is optional. Test a known projective transform, one outlier rejected by RANSAC, fewer than four correspondences, collinear landmarks, nonfinite coordinates, and a median reprojection error over the configured 12-pixel limit. Assert accepted test points project to the known metre coordinates and every rejected result has `valid is False` with a nonempty reason.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_cv_calibration.py -q`
Expected: FAIL because the video types and calibration functions do not exist.

- [ ] **Step 3: Implement typed video contracts and OpenCV homography estimation**

Add the dataclasses and a `cv2.findHomography(..., method=cv2.RANSAC, ransacReprojThreshold=max_reprojection_error_px)` implementation. Reject nonfinite/collinear inputs, fewer than four inliers, noninvertible transforms, and median inlier reprojection error above the supplied threshold. Keep OpenCV imports inside the calibration function so normal core imports do not require the optional CV extra.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `pytest tests/test_cv_calibration.py -q`
Expected: PASS for the known transform, outlier rejection, and all invalid-geometry cases.

- [ ] **Step 5: Commit the calibration boundary**

```powershell
git add src/soccer_tactics/cv/video_types.py src/soccer_tactics/cv/calibration.py tests/test_cv_calibration.py
git commit -m "feat: add video calibration contracts"
```

### Task 2: Map tracks into canonical rows and persist safe run artifacts

**Files:**
- Create: `src/soccer_tactics/cv/run_store.py`
- Modify: `src/soccer_tactics/cv/adapter.py`
- Modify: `.gitignore`
- Test: `tests/test_cv_adapter.py`
- Test: `tests/test_cv_run_store.py`

**Interfaces:**
- Produces `TrackAssignment(team: Literal["home", "away"], roster_name: str | None)`.
- Produces `video_tracks_to_tracking(detections: pd.DataFrame, *, match_id: str, fps: float, assignments: dict[str, TrackAssignment], home_attacks_positive_x: bool, period: int = 1) -> pd.DataFrame`; input columns are `frame`, `track_id`, `x`, and `y` in standard pitch metres.
- Produces `save_video_run(result: VideoTrackingResult, *, assignments: dict[str, TrackAssignment], output_root: Path) -> Path` and `source_sha256(video_path: Path) -> str`.
- Existing `detections_to_tracking` and `TrackingAdapter` call signatures and results remain compatible.

- [ ] **Step 1: Write failing conversion and persistence tests**

Assert mapped `track_7` becomes canonical `player_id="track_7"`; an unassigned track is absent; `fps=10` and frame 15 produce `time_s=1.5`; `home_attacks_positive_x=False` rotates both coordinates by 180 degrees; two track IDs assigned to one roster name remain two IDs; bad FPS and unknown team assignments fail clearly. For persistence, assert Parquet round-trips through `validate`, metadata includes the source SHA-256 and mapping but not a source path, and `overlay.mp4`, diagnostics, and metadata are written under the run directory.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_cv_adapter.py tests/test_cv_run_store.py -q`
Expected: FAIL because the track mapping and run store functions do not exist.

- [ ] **Step 3: Implement canonical mapping, metadata persistence, and ignore rules**

Use the existing `validate` and `detections_to_tracking` boundary without changing their public behavior. Convert assigned teams and clip-local IDs, apply the selected home orientation, omit unassigned rows, and write run files atomically beneath `data/processed/video_runs/<run_id>/`. Add `data/models/` to `.gitignore` (the existing `data/processed/` rule already covers run outputs); do not serialize source paths or copy the input video.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `pytest tests/test_cv_adapter.py tests/test_cv_run_store.py -q`
Expected: PASS; `git check-ignore data/models/probe.pt data/processed/video_runs/probe/tracking.parquet` reports both paths ignored.

- [ ] **Step 5: Commit canonical mapping and run storage**

```powershell
git add .gitignore src/soccer_tactics/cv/adapter.py src/soccer_tactics/cv/run_store.py tests/test_cv_adapter.py tests/test_cv_run_store.py
git commit -m "feat: export mapped video tracks"
```

### Task 3: Build the local Roboflow backend and video orchestrator

**Files:**
- Create: `src/soccer_tactics/cv/roboflow_backend.py`
- Create: `src/soccer_tactics/cv/pipeline.py`
- Create: `scripts/download_video_models.py`
- Modify: `src/soccer_tactics/cv/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cv_pipeline.py`
- Test: `tests/test_cv_backend.py`

**Interfaces:**
- `VisionBackend.detect_players(frame: np.ndarray) -> list[PlayerDetection]` and `VisionBackend.detect_pitch_landmarks(frame: np.ndarray) -> PitchLandmarks` are model-facing boundaries.
- `Tracker.update(frame_index: int, detections: list[PlayerDetection]) -> list[TrackedPlayer]` is the association boundary.
- Produces `run_video_tracking(video_path: Path, config: VideoTrackingConfig, *, backend: VisionBackend, tracker: Tracker, work_dir: Path, on_progress: Callable[[int, int], None] | None = None) -> VideoTrackingResult`.
- Produces `RoboflowVideoBackend.from_weight_paths(player_weights: Path, pitch_weights: Path, *, device: str = "cpu") -> RoboflowVideoBackend`.

- [ ] **Step 1: Write fake-backend orchestration tests**

Use a tiny generated video and fakes for detector, tracker, and progress callback. Assert every decoded frame is processed; one good frame emits a mapped track; a failed calibration, low-confidence player, and out-of-pitch point emit no position and increment the corresponding diagnostic; each emitted coordinate uses that frame's transform; the overlay and progress callback finish at the source frame count. Assert zero/nonfinite FPS and unreadable video raise actionable `ValueError`s.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_cv_pipeline.py tests/test_cv_backend.py -q`
Expected: FAIL because the video pipeline and backend do not exist.

- [ ] **Step 3: Implement lazy local model loading and frame processing**

Before implementation, probe Python 3.15 Windows binary-wheel availability for the required CV stack. Then declare compatible bounded optional CV dependencies in `pyproject.toml`; if a required dependency has no compatible wheel, stop this task and report the constraint without changing `requires-python`. Wrap the Roboflow soccer player and pitch YOLO weights with Ultralytics, and use Supervision ByteTrack for player/goalkeeper classes only. Process source frames sequentially, calibrate each frame independently, project each detection's bottom-center point, and produce an annotated review overlay and frame diagnostics. Keep torch/OpenCV/Supervision imports lazy and give a clear missing-dependency or missing-weight message. `download_video_models.py` downloads only the two required weights when explicitly run; it must not execute during imports, app startup, or processing.

- [ ] **Step 4: Run focused tests and confirm they pass without real model weights**

Run: `pytest tests/test_cv_pipeline.py tests/test_cv_backend.py -q`
Expected: PASS with fakes; real weights are not needed. Also run `py -3.15 -m pip install --dry-run --only-binary=:all: -e ".[cv]"` in a disposable environment. If Python 3.15-compatible binary dependencies cannot be resolved, stop before changing `requires-python` and report the supported local alternatives.

- [ ] **Step 5: Commit the local inference pipeline**

```powershell
git add src/soccer_tactics/cv scripts/download_video_models.py pyproject.toml tests/test_cv_pipeline.py tests/test_cv_backend.py
git commit -m "feat: process local soccer video"
```

### Task 4: Add the independent Streamlit video workflow

**Files:**
- Create: `src/soccer_tactics/dashboard/video_tracking.py`
- Modify: `app.py`
- Test: `tests/test_dashboard_video.py`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Produces `render_video_tracking(st, *, processor=run_video_tracking, model_paths=...) -> None`.
- The app's sidebar workspace selector uses exactly `Match analysis` and `Video tracking`; selecting video tracking calls the new page before requiring Metrica/IDSSE files.

- [ ] **Step 1: Write AppTest workflow tests with fake processing**

Assert the video tracking workspace renders even when no Metrica or IDSSE data exists; the normal match-analysis workspace still renders with the existing synthetic fixtures; invalid input shows an actionable error; successful fake processing displays the overlay, track mapping table, attack-direction choice, and export control; missing optional CV dependencies/weights preserve the rest of the app.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_dashboard_video.py tests/test_dashboard.py -q`
Expected: FAIL because there is no independent video workspace.

- [ ] **Step 3: Implement file selection, processing, mapping, review, and export UI**

Add a local video uploader and metadata preview, a one-time attack-direction radio, progress reporting, the output overlay/pitch preview, a track-to-team/name editor, and a save/download action. Store results only after valid mappings and use the Task 2 run store. Branch before the current match catalogue loading so this workspace is independent; leave existing report tabs and assistant logic unchanged.

- [ ] **Step 4: Run dashboard tests and the complete synthetic suite**

Run: `pytest tests/test_dashboard_video.py tests/test_dashboard.py -q`
Expected: PASS for both workspaces and error states.

Run: `pytest -q`
Expected: all available tests pass; optional CV-specific tests may skip only when their optional dependencies are absent.

- [ ] **Step 5: Commit the video workspace**

```powershell
git add app.py src/soccer_tactics/dashboard/video_tracking.py tests/test_dashboard_video.py tests/test_dashboard.py
git commit -m "feat: add local video tracking workspace"
```

### Task 5: Add SoccerNet and pitch-reference evaluation tooling

**Files:**
- Create: `src/soccer_tactics/cv/evaluation.py`
- Create: `scripts/evaluate_video_tracking.py`
- Test: `tests/test_cv_evaluation.py`

**Interfaces:**
- Produces `evaluate_calibration(predicted: pd.DataFrame, ground_truth_boxes: pd.DataFrame, pitch_references: pd.DataFrame) -> dict[str, float | int]` with point count, median/p90 calibration metre error, and median/p90 player-location metre error. Use SoccerNet box footpoints projected through the independent manually marked per-frame pitch calibration as the annotation-derived reference; compare predicted track footpoints against those references on matched frames/IDs.
- Produces `write_trackeval_mot_files(predictions: pd.DataFrame, annotations: pd.DataFrame, output_root: Path) -> Path`; source frame 0 is written as MOT frame 1 and XYXY boxes are converted to MOT XYWH.
- The CLI accepts local paths for predictions, SoccerNet annotations, manually marked pitch-reference CSV, and a local TrackEval checkout; it never downloads data or annotations.

- [ ] **Step 1: Write evaluator conversion and hand-calculated calibration tests**

Use a two-frame hand-computed example to assert the exact median/p90 calibration error and player-location error; assert the MOT file uses one-based frame numbers, correct ID, and `x,y,width,height` conversion. Assert missing or malformed reference columns fail clearly. The manual reference CSV must identify frame, image landmark coordinates, pitch landmark coordinates, and enough non-collinear known pitch landmarks per sampled frame; ground-truth boxes provide player IDs and image-space footpoints.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_cv_evaluation.py -q`
Expected: FAIL because the evaluation functions do not exist.

- [ ] **Step 3: Implement export and official metric invocation**

Write SoccerNet predictions and ground truth to a temporary MOTChallenge-compatible layout, invoke the supplied official TrackEval checkout for HOTA/CLEAR/Identity metrics with preprocessing disabled, and export HOTA, DetA, AssA, detection recall/misses, and ID switches alongside manually referenced median/p90 metre errors. Keep all temporary annotations and videos outside Git. Do not label annotation-derived footpoint comparisons as telemetry ground truth.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `pytest tests/test_cv_evaluation.py -q`
Expected: PASS without SoccerNet files or TrackEval installed; integration invocation is covered with a fake subprocess result.

- [ ] **Step 5: Commit evaluation tooling**

```powershell
git add src/soccer_tactics/cv/evaluation.py scripts/evaluate_video_tracking.py tests/test_cv_evaluation.py
git commit -m "feat: evaluate video tracking outputs"
```

### Task 6: Document setup, licensing, limitations, and release evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/VALIDATION.md`
- Create: `docs/validation/v5/README.md`
- Test: `tests/test_roadmap.py`

**Interfaces:**
- Documents `python -m pip install -e ".[dev,dashboard,cv]"`, the explicit model-weight setup command, local file processing, export location, and the evaluation command.
- Documents Roboflow/Ultralytics package/model terms and attribution separately; raw video, weights, and annotations stay untracked.

- [ ] **Step 1: Add documentation consistency tests**

Assert the README roadmap says player tracking/local files first; the V5 section links the spec and evaluation notes; setup commands use the project's optional CV extra and explicit weight command; no document suggests URL import, ball/event extraction, or live analysis is already implemented.

- [ ] **Step 2: Run the focused documentation test and confirm it fails**

Run: `pytest tests/test_roadmap.py -q`
Expected: FAIL until the README and V5 validation links exist.

- [ ] **Step 3: Document the supported local workflow and current limitations**

Add install/model-license details, local processing and ignored output paths, the unavailability of video-derived events, and how to reproduce SoccerNet/TrackEval plus manual pitch-point evaluation. Record measurements only after the real local run; do not invent scores or claim the gate passed without annotated footage and visual review.

- [ ] **Step 4: Run documentation and complete test checks**

Run: `pytest tests/test_roadmap.py -q`
Expected: PASS.

Run: `pytest -q`
Expected: all available tests pass.

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 5: Commit the V5 setup and validation notes**

```powershell
git add README.md docs/ROADMAP.md docs/VALIDATION.md docs/validation/v5/README.md tests/test_roadmap.py
git commit -m "docs: add V5 tracking setup and validation"
```

### Task 7: Run the real local acceptance gate

**Files:**
- Modify: `docs/validation/v5/README.md`
- Modify: `docs/VALIDATION.md`

**Interfaces:**
- Uses Task 3's pipeline, Task 4's review workspace, and Task 5's evaluator.
- Writes measured results and limitations only; raw inputs and generated artifacts remain in ignored directories.

- [ ] **Step 1: Run one continuous SoccerNet clip locally**

Use a locally obtained, appropriately licensed 30-second clip with its tracking annotations. Run the explicit model setup, process the clip at its source frame rate, assign team/roster mappings, and save the run under `data/processed/video_runs/`.

- [ ] **Step 2: Evaluate tracking and calibration**

Run the evaluator with the clip predictions, SoccerNet annotations, manual pitch-reference CSV, and local TrackEval checkout. Save the JSON report beneath `docs/validation/v5/` with HOTA/DetA/AssA, missed detections, identity switches, median/p90 calibration and player-location errors, source clip identifier, and software/model versions.

- [ ] **Step 3: Manually inspect and record overlays**

Inspect the beginning, middle, end, and any camera movement/zoom. Record missing-calibration behavior, track identity issues, coordinate orientation, and the overlay review in `docs/validation/v5/README.md`. Do not report success if the clip/annotations are unavailable or the overlay is not checked.

- [ ] **Step 4: Verify privacy and branch state**

Run: `git check-ignore data/models/football-player-detection.pt data/processed/video_runs/probe/tracking.parquet`.
Expected: both paths are ignored; `git status --short` contains no raw video, weights, annotations, or generated run data.

- [ ] **Step 5: Commit acceptance evidence and push the testing branch**

```powershell
git add docs/validation/v5/README.md docs/VALIDATION.md
git commit -m "docs: record V5 tracking evaluation"
git push origin Touchline-Testing-Branch
```
