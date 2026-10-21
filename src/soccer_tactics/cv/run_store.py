"""Persist local video runs without retaining input video or weight paths."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

from soccer_tactics.cv.adapter import TrackAssignment, video_tracks_to_tracking
from soccer_tactics.cv.video_types import VideoTrackingResult


def source_sha256(video_path: Path) -> str:
    """Hash the source incrementally without loading a full video into memory."""
    digest = hashlib.sha256()
    with Path(video_path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_video_run(
    result: VideoTrackingResult,
    *,
    assignments: dict[str, TrackAssignment],
    output_root: Path,
) -> Path:
    """Publish a complete run directory atomically beneath ``output_root``.

    The caller provides match_id, fps, home_attacks_positive_x and the source
    SHA-256 in video_metadata. Only explicitly allowed clip properties persist.
    """
    source = result.video_metadata
    source_hash = source.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
        raise ValueError("video metadata requires a source_sha256 digest")
    tracking = video_tracks_to_tracking(
        result.detections, match_id=source["match_id"], fps=source["fps"],
        assignments=assignments, home_attacks_positive_x=source["home_attacks_positive_x"],
        period=source.get("period", 1),
    )
    run_id = uuid4().hex
    metadata = {
        "run_id": run_id, "source_sha256": source_hash.lower(),
        "match_id": source["match_id"], "fps": source["fps"],
        "home_attacks_positive_x": source["home_attacks_positive_x"],
        "period": source.get("period", 1),
        "assignments": {track_id: asdict(value) for track_id, value in assignments.items()},
    }
    for key in ("width", "height", "frame_count", "duration_s"):
        value = source.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            metadata[key] = value
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=output_root))
    destination = output_root / run_id
    try:
        tracking.to_parquet(temporary / "tracking.parquet", index=False)
        result.frame_diagnostics.to_parquet(temporary / "frame_diagnostics.parquet", index=False)
        shutil.copyfile(result.overlay_path, temporary / "overlay.mp4")
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8",
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination
