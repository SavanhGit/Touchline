"""Independent local-video workspace. Optional inference dependencies load on demand."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import pandas as pd

from soccer_tactics.cv.pipeline import run_video_tracking
from soccer_tactics.cv.video_types import VideoTrackingConfig


MODEL_ROOT = Path(__file__).resolve().parents[3] / "data" / "models" / "video"
MODEL_PATHS = (MODEL_ROOT / "football-player-detection.pt", MODEL_ROOT / "football-pitch-detection.pt")
RUN_ROOT = MODEL_ROOT.parents[1] / "processed" / "video_runs"


def _process_upload(payload, config, processor, model_paths, st):
    """Keep source and intermediate overlays inside one cleaned processing scope."""
    with TemporaryDirectory(prefix="touchline-video-") as temporary:
        work = Path(temporary)
        source = work / "input.mp4"
        source.write_bytes(payload)
        import cv2
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ok, frame = capture.read()
            if (not capture.isOpened() or not ok or frame is None
                    or any(not math.isfinite(v) or v <= 0 for v in (fps, count, width, height))):
                raise ValueError("Unreadable video")
        finally:
            capture.release()
        st.json({"fps": fps, "frame_count": int(count), "width": int(width),
                 "height": int(height), "duration_s": count / fps})
        kwargs = {}
        if processor is run_video_tracking:
            from soccer_tactics.cv.roboflow_backend import RoboflowVideoBackend, SupervisionByteTracker
            kwargs = {"backend": RoboflowVideoBackend.from_weight_paths(*model_paths),
                      "tracker": SupervisionByteTracker(frame_rate=max(1, round(fps)))}
        progress = st.progress(0, text="Processing frames…")
        result = processor(source, config, work_dir=work,
                           on_progress=lambda done, total: progress.progress(min(done / max(total, 1), 1.)),
                           **kwargs)
        overlay = result.overlay_path.read_bytes()
        return replace(result, overlay_path=Path("overlay.mp4")), overlay


def render_video_tracking(st, *, processor=run_video_tracking, model_paths=MODEL_PATHS,
                          output_root=RUN_ROOT) -> None:
    """Upload, process, review, map and save a clip without a match catalogue."""
    st.title("Local video tracking")
    st.caption("Clip-local player IDs require manual team mapping. Pitch positions depend on visible field landmarks; failed calibration frames have no positions.")
    upload = st.file_uploader("Choose a local MP4", type=["mp4"], key="video_upload")
    match_id = st.text_input("Clip label", value="local-clip", key="video_match_id")
    direction = st.radio("Home attack direction for this entire clip", ["Home attacks +x", "Home attacks −x"],
                         index=None, key="video_direction")
    st.caption("Use one clip with a consistent attack direction. Team/name labels are manual; no events, ball tracking or match report are inferred.")
    payload = upload.getvalue() if upload is not None else None
    identity = (sha256(payload).hexdigest() if payload else None, match_id, direction)
    if st.session_state.get("_video_identity") != identity:
        for key in ("_video_review", "_video_archive", "video_mapping"):
            st.session_state.pop(key, None)
        st.session_state["_video_identity"] = identity
    if st.button("Process video", key="process_video", disabled=upload is None):
        st.session_state.pop("_video_review", None)
        st.session_state.pop("_video_archive", None)
        st.session_state.pop("video_mapping", None)
        if not payload or not match_id.strip() or direction is None:
            st.error("Choose a nonempty MP4, enter a clip label, and select the home attack direction.")
        else:
            config = VideoTrackingConfig(match_id.strip(), direction == "Home attacks +x")
            try:
                st.session_state["_video_review"] = _process_upload(payload, config, processor, model_paths, st)
            except (ImportError, RuntimeError, ValueError, OSError):
                st.error("Video processing unavailable. Choose a readable MP4 with a supported codec. Install soccer-tactics[cv] and run scripts/download_video_models.py explicitly to provide local weights, then retry.")
    review = st.session_state.get("_video_review")
    if review is None:
        return
    result, overlay = review
    st.subheader("Review overlay")
    st.video(overlay)
    st.download_button("Download review overlay", overlay, file_name="review-overlay.mp4", mime="video/mp4")
    st.json(result.video_metadata)
    st.dataframe(result.frame_diagnostics, hide_index=True)
    st.subheader("Pitch preview")
    if result.detections.empty:
        st.warning("No calibrated player positions. Review the overlay and retry footage with more pitch landmarks visible.")
        return
    frames = sorted(result.detections.frame.unique())
    selected = st.select_slider("Preview frame", options=frames, key="video_preview_frame") if len(frames) > 1 else frames[0]
    import plotly.express as px
    preview = result.detections[result.detections.frame == selected]
    figure = px.scatter(preview, x="x", y="y", text="track_id", range_x=[-52.5, 52.5], range_y=[-34, 34])
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(figure, width="stretch")
    st.subheader("Track mapping")
    st.caption("Assign every track to home/away or explicitly exclude it. Names are optional; IDs remain clip-local.")
    mapping = pd.DataFrame({"track_id": sorted(result.detections.track_id.unique()),
                            "team": "unassigned", "roster_name": ""})
    edited = st.data_editor(mapping, key="video_mapping", hide_index=True, disabled=["track_id"],
        column_config={"team": st.column_config.SelectboxColumn("Team", options=["unassigned", "home", "away", "exclude"], required=True),
                       "roster_name": st.column_config.TextColumn("Name (optional)")})
    mapping_identity = edited.to_json(orient="records")
    if st.session_state.get("_video_mapping_identity") != mapping_identity:
        st.session_state.pop("_video_archive", None)
        st.session_state["_video_mapping_identity"] = mapping_identity
    valid = edited.team.isin(["home", "away", "exclude"]).all() and edited.team.isin(["home", "away"]).any()
    if not valid:
        st.info("Map all tracks and include at least one player before saving.")
    if st.button("Save run", key="save_video_run", disabled=not valid) and valid:
        from soccer_tactics.cv.adapter import TrackAssignment
        from soccer_tactics.cv.run_store import save_video_run
        assignments = {str(row.track_id): TrackAssignment(row.team, str(row.roster_name).strip() or None)
                       for row in edited.itertuples() if row.team != "exclude"}
        try:
            with TemporaryDirectory(prefix="touchline-save-") as temporary:
                overlay_path = Path(temporary) / "overlay.mp4"
                overlay_path.write_bytes(overlay)
                destination = save_video_run(replace(result, overlay_path=overlay_path),
                                             assignments=assignments, output_root=output_root)
            archive = BytesIO()
            with ZipFile(archive, "w") as zipped:
                for path in sorted(destination.iterdir()):
                    zipped.writestr(path.name, path.read_bytes())
            st.session_state["_video_archive"] = archive.getvalue()
            st.success("Run saved with tracking, diagnostics, metadata and review overlay.")
        except (ValueError, OSError, ImportError):
            st.error("Could not save the run. Install pyarrow and check that the run storage is writable, then retry.")
    archive = st.session_state.get("_video_archive")
    if archive:
        st.download_button("Download run", archive, file_name="video-tracking-run.zip", mime="application/zip")
