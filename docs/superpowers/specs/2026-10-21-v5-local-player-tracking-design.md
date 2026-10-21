# V5 Local Video Player Tracking Design

## Goal

Let a Touchline user turn a short, continuous local soccer clip into reviewable player-tracking data in Touchline's canonical format. The first V5 milestone prioritizes trustworthy position and track data over broad video support: it does not derive passes, shots, or tactical conclusions from video.

The intended first use is a local portfolio workflow with no paid API and no upload to a remote inference service. The evaluated target is one continuous clip of approximately 30 seconds, aligned with the SoccerNet Tracking clip format.

## User workflow

1. Open a dedicated video-tracking view in the existing local Streamlit app and select a short video file.
2. Touchline validates that the file can be decoded and reports its duration, frame rate, and dimensions. It gives an actionable error for unsupported or corrupt input.
3. The user identifies the home and away sides, chooses the home team's attacking direction for this clip, and may provide roster names.
4. Local inference detects players and estimates player tracks and pitch landmarks frame by frame. Touchline projects player footpoints to pitch coordinates, preserving missing observations when detection or calibration is not trustworthy.
5. The review view shows the source frame with detections and track IDs alongside the pitch reconstruction. The user assigns each track ID to home/away and optionally to a roster name.
6. The user saves and exports the processed tracking data and run metadata. The original video and annotation files remain outside Git.

Manual team and roster assignment is required for exported tracks. Names are optional. A track ID is a tracker identity, not proof of a real player's identity. If an identity switch creates multiple IDs, the user can associate those IDs with the same roster name in metadata, but this milestone does not merge their coordinate series or automatically repair the switch.

## Processing architecture

The pipeline is UI-independent and sits behind the existing `soccer_tactics.cv` adapter. Streamlit is responsible for selecting a local clip, collecting the attack-direction and mapping inputs, displaying progress and review frames, and saving the result. Tests and later callers can run the pipeline without Streamlit.

The pipeline consists of replaceable local components:

- A player detector emits per-frame player bounding boxes and confidence values.
- A multi-object tracker associates detections across adjacent frames and emits clip-local track IDs. Team and roster identity are not inferred by the tracker.
- A pitch landmark detector estimates visible, known pitch landmarks for each frame.
- A per-frame homography maps image coordinates to the standard 105 m by 68 m pitch. Robust fitting rejects inconsistent landmark correspondences. The calibration result includes a quality diagnostic.
- The bottom-center of each player detection is projected onto the pitch. The selected home attack direction is applied so exported coordinates follow Touchline's convention that home attacks toward +x.
- The existing canonical schema validates the final rows. Rows with unassigned teams, invalid player detections, failed calibration, or out-of-pitch mapped locations are omitted and counted in diagnostics rather than filled or guessed.

The initial local baseline uses Roboflow Sports soccer components, with its player detector, tracker integration, and pitch-keypoint approach behind these replaceable boundaries. No Roboflow or model weights are committed to the repository. Third-party source, package, and weight licenses must be checked and documented separately; the Roboflow soccer example identifies the YOLOv8 component as AGPL-3.0, while the `roboflow/sports` repository is MIT-licensed. Inference runs locally. Weights are obtained through an explicit documented setup step; Touchline does not download them during app startup or video processing.

The optional computer-vision environment must be compatible with the project's supported Windows and Python 3.15 setup. If a required package does not support that environment, the implementation must report the incompatibility and evaluate alternatives before changing the project's Python requirement.

## Calibration and missing data

The camera may move or zoom within the evaluated segment, so calibration is estimated independently on every processed frame. A calibration failure on a frame means no player positions are emitted for that frame. The implementation must not reuse a stale homography, interpolate an off-camera player, or fabricate a pitch location. Any confidence threshold or geometric quality filter must be explicit, configurable for tests, and recorded in run metadata.

The pitch transform is a planar approximation. Player footpoints, partial pitch visibility, perspective, lens distortion, and imperfect landmarks introduce location error. Review output must make failed or low-quality calibration visible instead of presenting every frame as equally reliable.

## Data contracts and saved runs

The pipeline exposes a UI-independent entry point equivalent to:

```python
run_video_tracking(video_path, config) -> VideoTrackingResult
```

`VideoTrackingResult` contains canonical tracking rows, track summaries, per-frame detection/calibration diagnostics, source video metadata, and the selected orientation. The exact Python types are fixed in the implementation plan, but the boundary between inference, review UI, and persistence remains stable.

Canonical rows use the existing columns `[match_id, period, frame, time_s, team, player_id, x, y]`. `player_id` retains the clip-local track ID; `team` is `home` or `away` after manual mapping. `frame` is the decoded source frame index, `time_s` is computed from the source frame rate, and one clip is represented as period 1. Roster names and user-facing labels are stored in a sidecar mapping, not substituted for the tracker ID in canonical rows.

Saved runs live under the Git-ignored `data/processed/video_runs/<run_id>/` directory and include:

- canonical tracking data in Parquet;
- `metadata.json` with source-content hash, clip properties, run ID, model/package identifiers and versions, calibration settings, attack direction, and track-to-team/name mapping;
- review overlay video and machine-readable per-frame quality diagnostics.

The source path is not persisted. The original video is read from a temporary local copy when needed and is not copied into the saved run. Raw videos, generated run data, weights, and annotations stay outside Git.

The output is ready for existing tracking-based analytics and later video-derived event work. This milestone does not create an event table, so event-dependent match reports and the question-answering assistant must not imply that video-only data supports those answers.

## Evaluation and acceptance

### Tracking evaluation

Use an appropriately licensed, locally obtained SoccerNet Tracking clip with ground-truth bounding boxes and track IDs. Compare predicted tracks with ground truth using the official SoccerNet/TrackEval workflow and record HOTA, DetA, and AssA. Also report missed detections and identity switches in plain language. Record the clip/split identifier and evaluation configuration without committing the video or annotation files.

### Calibration and location evaluation

Manually mark pitch reference landmarks in sampled frames from the evaluation clip, recording their known pitch coordinates. Use those independent points to report pixel-to-pitch reprojection error in metres. Estimate player-location error by comparing projected prediction footpoints against reference locations obtained from the annotated ground-truth boxes and manual pitch calibration. Clearly label this as an annotation-derived estimate, not ground-truth player telemetry.

### Visual review

Inspect representative frames across the clip, including a camera movement or zoom when present, and record whether boxes, track IDs, pitch projections, missing frames, and attack orientation match the source. Do not claim success based only on unit tests or a rendered overlay.

### Automated tests

- Use fake detector, tracker, and calibrator implementations; CI must not need model weights, network access, or video annotations.
- Verify video metadata/time conversion, coordinate mapping, attack-direction rotation, team/name sidecar mapping, and canonical schema output.
- Verify that low-confidence/failed calibration, missing detections, and off-pitch projections omit rows and produce diagnostics without position interpolation.
- Verify stable run persistence, source hashing, and that raw video, annotations, generated runs, and model weights remain excluded by Git ignore rules.
- Verify the Streamlit view handles invalid files, missing optional CV dependencies/weights, progress and review/export state, and does not prevent existing Metrica/IDSSE dashboard use.
- Run the suite under the supported Windows/Python 3.15 environment when its dependencies are available.

The milestone is complete when a local short-clip run produces valid canonical tracking, users can assign teams and optional names, the output can be reviewed and exported, automated tests cover missing-data behavior, and the SoccerNet/manual-pitch evaluation and visual-review results are recorded. No fixed accuracy threshold is claimed before the first baseline is measured; measured results and limitations must be documented.

## Out of scope

- URL or broadcast-site retrieval; this follows after local-file processing.
- Ball tracking, event extraction, or video-derived tactical/score analysis.
- Jersey OCR, automatic team classification, or automatic player naming.
- Full-match processing, camera-cut handling, or live/incremental analysis.
- Paid APIs, remote inference, public hosting, or committing video/model/annotation assets.

## References

- Roboflow Sports repository and MIT license: <https://github.com/roboflow/sports>
- Roboflow Sports soccer example and its YOLOv8 license note: <https://github.com/roboflow/sports/blob/main/examples/soccer/README.md>
- SoccerNet Tracking dataset, annotations, and evaluation: <https://github.com/SoccerNet/sn-tracking>
- GNU AGPL-3.0 overview: <https://www.gnu.org/licenses/why-affero-gpl.html>
