"""Readiness-gated metadata catalogue for local Metrica and IDSSE matches."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

from soccer_tactics.loaders import idsse


@dataclass(frozen=True)
class CatalogEntry:
    provider: str
    match_id: str
    source_path: str
    display_name: str
    team_names: dict[str, str]
    ready: bool
    reason: str | None = None


def _metrically_available(root: Path, game: int) -> bool:
    folder = root / "data" / f"Sample_Game_{game}"
    paths = [folder / f"Sample_Game_{game}_Raw{suffix}.csv" for suffix in
             ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team")]
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _fingerprints(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    return [{"relative_name": path.name, "size_bytes": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(root.glob("*.xml"))]


def _idsse_available() -> tuple[bool, str]:
    try:
        available = importlib.util.find_spec("floodlight") is not None
    except (ImportError, ValueError):
        available = False
    return available, "" if available else "IDSSE needs Floodlight. Install it with: pip install -e '.[idsse]'"


def list_match_catalogue(metrica_root: str | Path, idsse_root: str | Path,
                         validation_manifest: str | Path) -> list[CatalogEntry]:
    """List usable Metrica sources and manifest candidates with explicit readiness."""
    metrica_root, idsse_root = Path(metrica_root), Path(idsse_root)
    entries = []
    for game in (1, 2):
        if _metrically_available(metrica_root, game):
            entries.append(CatalogEntry("Metrica", f"metrica_{game}", str(metrica_root),
                                        f"Metrica · Sample game {game}",
                                        {"home": "Home (anonymous)", "away": "Away (anonymous)"}, True))

    manifest = None
    manifest_error = None
    try:
        with Path(validation_manifest).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("manifest root must be an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest_error = f"IDSSE validation manifest is unavailable or invalid: {exc}"

    if manifest is not None:
        fingerprints = manifest.get("source_fingerprints")
        candidates_data = manifest.get("candidates")
        valid_fingerprints = (isinstance(fingerprints, list) and all(
            isinstance(row, dict) and isinstance(row.get("relative_name"), str)
            and row.get("relative_name") and isinstance(row.get("size_bytes"), int)
            and row.get("size_bytes") >= 0 and isinstance(row.get("mtime_ns"), int)
            and row.get("mtime_ns") >= 0
            for row in fingerprints))
        valid_candidates = isinstance(candidates_data, list)
        seen_ids = set()
        if valid_candidates:
            for row in candidates_data:
                if not isinstance(row, dict):
                    valid_candidates = False
                    break
                match_id = row.get("match_id")
                names = row.get("team_names")
                status = row.get("status")
                if (not isinstance(match_id, str) or not match_id
                        or match_id in seen_ids
                        or not isinstance(status, str)
                        or status not in {"accepted", "rejected"}
                        or not isinstance(row.get("detail"), str)
                        or not isinstance(row.get("reasons"), list)
                        or not all(isinstance(reason, str) for reason in row.get("reasons", []))
                        or not isinstance(names, dict)
                        or set(names) != {"home", "away"}
                        or not all(isinstance(names.get(side), str) and names[side]
                                   for side in ("home", "away"))):
                    valid_candidates = False
                    break
                seen_ids.add(match_id)
        valid_errors = isinstance(manifest.get("errors"), list) and not manifest["errors"]
        valid_count = (isinstance(manifest.get("candidate_count"), int)
                       and not isinstance(manifest.get("candidate_count"), bool)
                       and manifest["candidate_count"] == 7
                       and len(candidates_data or []) == 7)
        valid_schema = manifest.get("schema_version") == "idsse-source-quality-v1"
        if not all((valid_fingerprints, valid_candidates, valid_errors, valid_count, valid_schema)):
            manifest_error = "IDSSE validation manifest is malformed or incomplete"

    found = idsse._files(idsse_root) if idsse_root.is_dir() else {}
    complete_ids = set(idsse.list_idsse_matches(idsse_root)) if idsse_root.is_dir() else set()
    candidate_rows = (manifest or {}).get("candidates", [])
    if not isinstance(candidate_rows, list):
        candidate_rows = []
    candidates = {row.get("match_id"): row for row in candidate_rows
                  if isinstance(row, dict) and isinstance(row.get("match_id"), str)}
    if manifest is not None and set(candidates) != set(found):
        manifest_error = "IDSSE validation manifest is incomplete: candidate IDs do not match discovered XML IDs"
    all_ids = sorted(set(candidates) | set(found))
    if not all_ids:
        return entries

    stale = manifest_error
    if stale is None:
        if manifest.get("readiness_scope") != "source_event_origins_only":
            stale = "IDSSE manifest has an unsupported readiness scope"
        elif sorted(manifest["source_fingerprints"], key=lambda row: row["relative_name"]) != _fingerprints(idsse_root):
            stale = "IDSSE validation manifest is stale: XML file fingerprints do not match"
    installed, install_message = _idsse_available()
    for match_id in all_ids:
        row = candidates.get(match_id, {})
        ready = not stale and match_id in complete_ids and row.get("status") == "accepted" and installed
        reason = stale
        if reason is None and match_id not in complete_ids:
            reason = "Complete IDSSE XML triplet is unavailable"
        if reason is None and row.get("status") != "accepted":
            reason = row.get("detail") or "IDSSE candidate was not accepted by validation"
        if reason is None and not installed:
            reason = install_message
        labels = row.get("team_names") or {"home": "Home", "away": "Away"}
        entries.append(CatalogEntry("IDSSE", f"idsse_{match_id}", str(idsse_root),
                                    f"IDSSE · {match_id}", labels, bool(ready), reason))
    return entries
