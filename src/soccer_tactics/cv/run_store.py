"""Persist local video runs without retaining input video or weight paths."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import TypedDict
from uuid import uuid4

from soccer_tactics.cv.adapter import TrackAssignment, video_tracks_to_tracking
from soccer_tactics.cv.video_types import VideoTrackingResult


class ModelProvenance(TypedDict):
    """Non-path model identifier and release/checkpoint version."""

    id: str
    version: str


class RunProvenance(TypedDict):
    """Model roles and package releases used to produce a video run."""

    models: dict[str, ModelProvenance]
    packages: dict[str, str]


def _provenance_identifier(value: object) -> str:
    # Dots are excluded so a weight/package filename cannot pass as an ID.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_@-]{0,127}", value):
        raise ValueError("provenance names and IDs must be identifiers without dots or path separators")
    return value


def _provenance_version(value: object) -> str:
    # Version tokens retain dots and local/build suffixes used by releases.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+@-]{0,127}", value):
        raise ValueError("provenance versions must be non-path version tokens")
    return value


def _safe_provenance(value: object) -> RunProvenance:
    if not isinstance(value, dict) or set(value) != {"models", "packages"}:
        raise ValueError("provenance requires only models and packages")
    models, packages = value["models"], value["packages"]
    if not isinstance(models, dict) or not isinstance(packages, dict):
        raise ValueError("provenance models and packages must be mappings")
    safe_models = {}
    for role, model in models.items():
        if not isinstance(model, dict) or set(model) != {"id", "version"}:
            raise ValueError("provenance models require only id and version")
        safe_models[_provenance_identifier(role)] = ModelProvenance(
            id=_provenance_identifier(model["id"]), version=_provenance_version(model["version"]),
        )
    return RunProvenance(
        models=safe_models,
        packages={_provenance_identifier(name): _provenance_version(version)
                  for name, version in packages.items()},
    )


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
    SHA-256 in video_metadata. Optional ``provenance`` follows RunProvenance;
    names and IDs must be 1-128 character tokens starting with an alphanumeric
    character, followed by alphanumerics or ``_@-`` (no filename dots). Version
    tokens additionally allow ``.+`` for ordinary dotted/build releases.
    Only explicitly allowed clip properties and validated provenance persist.
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
    if "provenance" in source:
        metadata["provenance"] = _safe_provenance(source["provenance"])
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
