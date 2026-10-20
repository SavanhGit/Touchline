"""Reproducible IDSSE source audit for origin-only match views.

Tracking is inspected for source structure only. Event timestamps are never
joined to tracking frames by this validator.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import math
from numbers import Integral, Real
import re
import tracemalloc

import numpy as np
import pandas as pd

from soccer_tactics.loaders.idsse_bundle import load_idsse_bundle
from soccer_tactics.loaders.idsse import _event_kind


SCHEMA_VERSION = "idsse-source-quality-v1"
SOURCE_POSITION_SLOT_SEMANTICS = (
    "Observed-frame XY slots across every source object column, including roster slots never observed; not active-player detections.")
_FAMILIES = {"matchinformation": "info", "events_raw": "events",
             "positions_raw_observed": "positions"}
_REQUIRED = frozenset(_FAMILIES.values())
_MATCH_ID = re.compile(r"DFL-MAT-[A-Z0-9]+")
_QUALITY_KEYS = ("tracking_rows", "periods", "fps", "source_fps_by_period",
                 "frame_gaps", "source_missing_positions", "missing_positions",
                 "nonfinite_positions", "duplicate_objects", "conflicting_clocks",
                 "out_of_bounds", "source_events_total", "event_origin_count")
_COUNT_QUALITY_KEYS = ("frame_gaps", "source_missing_positions", "missing_positions",
                       "nonfinite_positions", "duplicate_objects", "conflicting_clocks",
                       "out_of_bounds")
_TRACKING_QUALITY_KEYS = _QUALITY_KEYS[:11]


def _fingerprint(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {"relative_name": path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _counts(values) -> dict:
    return {str(key): int(value) for key, value in sorted(dict(values).items(),
                                                          key=lambda item: str(item[0]))}


def _score(value) -> dict:
    return {team: int(value[team]) for team in ("home", "away")}


def _nonnegative_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _positive_finite(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_quality_diagnostics(quality: dict, reasons: list[str]) -> None:
    for key in _TRACKING_QUALITY_KEYS:
        if key not in quality:
            reasons.append(f"missing_quality_diagnostic:{key}")
    if "tracking_rows" in quality and (not _nonnegative_int(quality["tracking_rows"]) or
                                       quality["tracking_rows"] == 0):
        reasons.append("invalid_quality_diagnostic:tracking_rows")
    for key in _COUNT_QUALITY_KEYS:
        if key in quality and not _nonnegative_int(quality[key]):
            reasons.append(f"invalid_quality_diagnostic:{key}")
    periods = quality.get("periods")
    if "periods" in quality and (not isinstance(periods, (list, tuple)) or
                                 not periods or any(type(period) is not int or period not in (1, 2)
                                                    for period in periods) or
                                 len(set(periods)) != len(periods)):
        reasons.append("invalid_quality_diagnostic:periods")
    if "fps" in quality and not _positive_finite(quality["fps"]):
        reasons.append("invalid_quality_diagnostic:fps")
    source_fps = quality.get("source_fps_by_period")
    if "source_fps_by_period" in quality:
        expected = {str(period) for period in periods} if isinstance(periods, (list, tuple)) else set()
        if (not isinstance(source_fps, Mapping) or
                len(source_fps) != len(expected) or
                {str(key) for key in source_fps} != expected or
                any(not _positive_finite(value) for value in source_fps.values())):
            reasons.append("invalid_quality_diagnostic:source_fps_by_period")


def _validate_origin_rows(bundle, reasons: list[str]) -> None:
    rows = bundle.event_origins
    if rows.empty:
        reasons.append("no_usable_event_origins")
        return
    required = {"match_id", "period", "team", "event", "source_code",
                "source_timestamp", "gameclock_s", "source_x_m", "source_y_m",
                "x_m", "y_m"}
    if required - set(rows):
        reasons.append("event_origin_schema_missing")
        return
    if {"frame", "x1", "y1", "to_x", "to_y"} & set(rows):
        reasons.append("event_origin_claims_frame_or_endpoint")
    if (set(rows.match_id.astype(str)) != {str(bundle.match_id)} or
            not rows.team.isin(("home", "away")).all() or
            not rows.event.isin(("pass", "shot")).all() or
            not rows.period.isin((1, 2)).all()):
        reasons.append("invalid_event_origin_schema")
    for code, event in zip(rows.source_code, rows.event):
        if not isinstance(code, str) or not code.strip():
            reasons.append("missing_event_origin_source_code")
            continue
        kind, _, modeling_penalty = _event_kind(code)
        if kind is None or modeling_penalty:
            reasons.append("unsupported_event_origin_source_code")
        elif kind != event:
            reasons.append("event_origin_source_code_mismatch")
    timestamps = pd.to_datetime(rows.source_timestamp, utc=True, errors="coerce",
                                format="ISO8601")
    clock = pd.to_numeric(rows.gameclock_s, errors="coerce").to_numpy(dtype=float)
    if timestamps.isna().any() or not np.isfinite(clock).all() or (clock < 0).any():
        reasons.append("invalid_event_origin_time")
    pitch = bundle.metadata["source_pitch_dimensions_m"]
    length, width = float(pitch["length"]), float(pitch["width"])
    if not math.isfinite(length) or not math.isfinite(width) or min(length, width) <= 0:
        reasons.append("invalid_source_pitch_dimensions")
        return
    x = pd.to_numeric(rows.source_x_m, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows.source_y_m, errors="coerce").to_numpy(dtype=float)
    cx = pd.to_numeric(rows.x_m, errors="coerce").to_numpy(dtype=float)
    cy = pd.to_numeric(rows.y_m, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite([x, y, cx, cy]).all() or (x < 0).any() or
            (x > length).any() or (y < 0).any() or (y > width).any()):
        reasons.append("event_origin_out_of_bounds")
    elif not (np.allclose(cx, x - length / 2, atol=1e-6) and
              np.allclose(cy, y - width / 2, atol=1e-6)):
        reasons.append("event_origin_transform_mismatch")


def _audit_bundle(bundle, row: dict) -> None:
    reasons = row["reasons"]
    quality = bundle.quality
    _validate_quality_diagnostics(quality, reasons)
    row["quality"] = {key: quality[key] for key in _QUALITY_KEYS if key in quality}
    if (_nonnegative_int(quality.get("tracking_rows")) and
            _nonnegative_int(quality.get("source_missing_positions"))):
        row["quality"]["source_position_slots_total"] = (
            int(quality["tracking_rows"]) + int(quality["source_missing_positions"]))
    row["source_event_codes"] = _counts(quality.get("source_event_codes", {}))
    row["event_exclusion_reasons"] = _counts(quality.get("event_exclusion_reasons", {}))
    row["source_score"] = _score(bundle.metadata["source_score"])
    row["scored_goals"] = _counts(bundle.goals.team.value_counts())
    row["scored_goals"] = {team: row["scored_goals"].get(team, 0)
                           for team in ("home", "away")}
    row["supported_event_origins"] = _counts(bundle.event_origins.event.value_counts()) if "event" in bundle.event_origins else {}
    row["penalty_goal_count"] = int(bundle.goals.penalty.sum())
    row["team_names"] = {team: str(bundle.team_names[team]) for team in ("home", "away")} if hasattr(bundle, "team_names") else {}
    if row["source_score"] != row["scored_goals"] or _score(bundle.score) != row["source_score"]:
        reasons.append("source_score_goal_mismatch")
    if quality.get("scored_goals", len(bundle.goals)) != len(bundle.goals):
        reasons.append("goal_record_count_mismatch")
    total = quality.get("source_events_total")
    accepted = len(bundle.event_origins)
    if (total is None or total != sum(row["source_event_codes"].values()) or
            total != accepted + sum(row["event_exclusion_reasons"].values()) or
            quality.get("event_origin_count") != accepted):
        reasons.append("unexplained_event_count")
    capabilities = bundle.metadata.get("event_capabilities", {})
    if capabilities != {"origin_coordinates": True, "pass_endpoints": False,
                        "attack_direction": False, "tracking_sync": False}:
        reasons.append("unsafe_event_capabilities")
    for key in ("duplicate_objects", "conflicting_clocks", "nonfinite_positions"):
        if quality.get(key, 0):
            reasons.append(f"tracking_{key}")
    _validate_origin_rows(bundle, reasons)


def validate_idsse_dataset(root: str | Path) -> dict:
    """Audit every discoverable XML candidate, loading one full bundle at a time."""
    root = Path(root)
    result = {
        "schema_version": SCHEMA_VERSION,
        "source": {"provider": "IDSSE/DFL", "credit": "Deutsche Fußball Liga (DFL)",
                   "dataset_url": "https://github.com/spoho-datascience/idsse-data",
                   "license": "CC BY 4.0",
                   "license_url": "https://creativecommons.org/licenses/by/4.0/",
                   "paper": "Bassek et al., An integrated dataset of spatiotemporal and event data in elite soccer, Scientific Data 12, 195 (2025)",
                   "paper_url": "https://doi.org/10.1038/s41597-025-04505-y"},
        "readiness_scope": "source_event_origins_only",
        "source_position_slot_semantics": SOURCE_POSITION_SLOT_SEMANTICS,
        "limitations": {"tracking_sync": "not_validated",
                        "attack_direction": "not_established",
                        "pass_endpoints": "unavailable"},
        "errors": [], "candidate_count": 0, "candidates": [],
        "source_fingerprints": [],
        "memory": {"metric": "peak traced Python allocations during one bundle load; rounded up to 64 MiB, excludes native NumPy allocations",
                   "peak_python_mib_upper_bound": 0},
    }
    if not root.is_dir():
        result["errors"].append("data_root_missing")
        return result
    candidates: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(root.glob("*.xml")):
        result["source_fingerprints"].append(_fingerprint(path, root))
        family = next((key for name, key in _FAMILIES.items() if name in path.name.lower()), None)
        match = _MATCH_ID.search(path.name)
        if family is None or match is None:
            result["errors"].append(f"unrecognized_xml_file:{path.name}")
            continue
        candidates.setdefault(match.group(), {}).setdefault(family, []).append(path)
    for match_id, families in sorted(candidates.items()):
        row = {"match_id": match_id, "status": "rejected", "reasons": [],
               "detail": "", "xml_families": {family: len(families.get(family, []))
                                              for family in sorted(_REQUIRED)}}
        row["reasons"].extend(f"missing_xml_family:{family}" for family in sorted(_REQUIRED - families.keys()))
        row["reasons"].extend(f"duplicate_xml_family:{family}" for family in sorted(
            family for family, paths in families.items() if len(paths) != 1))
        if not row["reasons"]:
            bundle = None
            tracemalloc.start()
            try:
                bundle = load_idsse_bundle(root, match_id)
                _audit_bundle(bundle, row)
                if not row["reasons"]:
                    row["status"] = "accepted"
            except ImportError as exc:
                row["reasons"].append("missing_idsse_dependency")
                row["detail"] = str(exc)
            except Exception as exc:
                row["reasons"].append("bundle_validation_failed")
                row["detail"] = f"{type(exc).__name__}: {exc}"
            finally:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # Quantization makes repeated manifests stable while reporting an
                # honest upper bound for traced Python allocations.
                ceiling = math.ceil(peak / (64 * 1024 * 1024)) * 64
                result["memory"]["peak_python_mib_upper_bound"] = max(
                    result["memory"]["peak_python_mib_upper_bound"], ceiling)
                bundle = None
        row["reasons"] = sorted(set(row["reasons"]))
        result["candidates"].append(row)
    result["candidate_count"] = len(result["candidates"])
    if result["candidate_count"] != 7:
        result["errors"].append(f"expected_seven_candidates_found:{result['candidate_count']}")
    return result
