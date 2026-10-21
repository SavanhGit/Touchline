import json
from pathlib import Path

import pytest

from soccer_tactics.loaders import catalog


MATCH_IDS = [f"DFL-MAT-A{i}" for i in range(1, 8)]
FAMILIES = ("matchinformation", "events_raw", "positions_raw_observed")


def _triplets(root):
    paths = []
    for match_id in MATCH_IDS:
        for family in FAMILIES:
            path = root / f"DFL_{family}_{match_id}.xml"
            path.write_text("xml")
            paths.append(path)
    return paths


def _manifest(path, fingerprints, *, statuses=None):
    status_overrides = statuses or {}
    statuses = {match_id: "accepted" for match_id in MATCH_IDS}
    statuses.update(status_overrides)
    path.write_text(json.dumps({
        "readiness_scope": "source_event_origins_only",
        "schema_version": "idsse-source-quality-v1",
        "candidate_count": len(statuses),
        "source_fingerprints": fingerprints,
        "errors": [],
        "candidates": [
            {"match_id": match_id, "status": status,
             "team_names": {"home": "Home Name", "away": "Away Name"},
             "detail": "quality rejected" if status != "accepted" else "",
             "reasons": []}
            for match_id, status in statuses.items()
        ],
    }))
    return path


def _fingerprint(path):
    stat = path.stat()
    return {"relative_name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def test_metrica_games_remain_available_without_idsse_data(tmp_path):
    root = tmp_path / "metrica"
    for game in (1, 2):
        folder = root / "data" / f"Sample_Game_{game}"
        folder.mkdir(parents=True)
        for suffix in ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team"):
            (folder / f"Sample_Game_{game}_Raw{suffix}.csv").write_text("ok")
    manifest = tmp_path / "missing.json"
    entries = catalog.list_match_catalogue(root, tmp_path / "idsse", manifest)
    assert [(e.provider, e.match_id, e.ready) for e in entries] == [
        ("Metrica", "metrica_1", True), ("Metrica", "metrica_2", True)]


def test_accepted_idsse_requires_current_fingerprints_and_optional_dependency(tmp_path, monkeypatch):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    manifest = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources])
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (False, "Install it: pip install -e '.[idsse]'"))
    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
    entry = next(entry for entry in entries if entry.match_id == "idsse_DFL-MAT-A1")
    assert entry.provider == "IDSSE" and not entry.ready
    assert "pip install -e '.[idsse]'" in entry.reason


def test_stale_or_malformed_manifest_fails_closed(tmp_path):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    fingerprints = [_fingerprint(source) for source in sources]
    fingerprints[0] = {**fingerprints[0], "mtime_ns": 1}
    manifest = _manifest(tmp_path / "manifest.json", fingerprints)
    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
    assert all(not entry.ready and "stale" in entry.reason.lower() for entry in entries)
    manifest.write_text("{")
    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
    assert all(not entry.ready and "manifest" in entry.reason.lower() for entry in entries)
    manifest.write_text(json.dumps({"readiness_scope": "source_event_origins_only", "candidates": [], "source_fingerprints": [None]}))
    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
    assert all(not entry.ready and "manifest" in entry.reason.lower() for entry in entries)


def test_rejected_idsse_candidate_is_present_but_not_ready(tmp_path, monkeypatch):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    manifest = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources], statuses={"DFL-MAT-A1": "rejected"})
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (True, ""))
    entry = next(entry for entry in catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
                 if entry.match_id == "idsse_DFL-MAT-A1")
    assert not entry.ready and entry.reason == "quality rejected"


def test_idsse_ids_are_provider_qualified_and_labels_are_from_manifest(tmp_path, monkeypatch):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    manifest = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources])
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (True, ""))
    entry = next(entry for entry in catalog.list_match_catalogue(tmp_path / "none", idsse, manifest)
                 if entry.match_id == "idsse_DFL-MAT-A1")
    assert entry.ready
    assert entry.match_id.startswith("idsse_")
    assert entry.team_names == {"home": "Home Name", "away": "Away Name"}
    assert entry.display_name.startswith("IDSSE ·")


@pytest.mark.parametrize("damage", ["incomplete", "duplicate", "count", "errors", "status_dict", "status_list"])
def test_malformed_candidate_manifest_never_enables_idsse(tmp_path, monkeypatch, damage):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    manifest_path = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources])
    document = json.loads(manifest_path.read_text())
    if damage == "incomplete":
        del document["candidates"][0]["team_names"]
    elif damage == "duplicate":
        document["candidates"].append({**document["candidates"][0], "status": "rejected"})
        document["candidate_count"] = 8
    elif damage == "count":
        document["candidate_count"] = 6
    elif damage == "status_dict":
        document["candidates"][0]["status"] = {}
    elif damage == "status_list":
        document["candidates"][0]["status"] = []
    else:
        document["errors"] = ["audit error"]
    manifest_path.write_text(json.dumps(document))
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (True, ""))

    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, manifest_path)
    assert entries and all(not entry.ready for entry in entries)
    assert all("manifest" in entry.reason.lower() for entry in entries)


def test_manifest_must_account_for_all_seven_discovered_triplets(tmp_path, monkeypatch):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    document_path = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources])
    document = json.loads(document_path.read_text())
    document["candidates"] = document["candidates"][:-1]
    document["candidate_count"] = 6
    document_path.write_text(json.dumps(document))
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (True, ""))

    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, document_path)
    assert len(entries) == 7
    assert all(not entry.ready and "manifest" in entry.reason.lower() for entry in entries)


def test_manifest_candidate_ids_must_equal_discovered_xml_ids(tmp_path, monkeypatch):
    idsse = tmp_path / "idsse"
    idsse.mkdir()
    sources = _triplets(idsse)
    document_path = _manifest(tmp_path / "manifest.json", [_fingerprint(source) for source in sources])
    document = json.loads(document_path.read_text())
    document["candidates"][-1]["match_id"] = "DFL-MAT-SYNTHETIC"
    document_path.write_text(json.dumps(document))
    monkeypatch.setattr(catalog, "_idsse_available", lambda: (True, ""))

    entries = catalog.list_match_catalogue(tmp_path / "none", idsse, document_path)
    assert len(entries) == 8
    assert all(not entry.ready and "manifest" in entry.reason.lower() for entry in entries)
