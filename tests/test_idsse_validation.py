"""Source-quality manifest behavior without loading large Floodlight XML fixtures."""
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from soccer_tactics.analytics import idsse_validation
from scripts import validate_idsse as cli


FAMILIES = ("matchinformation", "events_raw", "positions_raw_observed")


def files(root, match_id="DFL-MAT-ONE", families=FAMILIES):
    for family in families:
        (root / f"DFL_{family}_{match_id}.xml").write_bytes(family.encode())


def bundle(match_id):
    origins = pd.DataFrame([{
        "match_id": match_id, "period": 1, "team": "home", "event": "pass",
        "source_code": "Play_Pass", "source_timestamp": "2025-01-01T12:00:00Z",
        "gameclock_s": 1.0, "source_x_m": 80.0, "source_y_m": 20.0,
        "x_m": 25.0, "y_m": -15.0, "player_id": "secret-player",
    }])
    goals = pd.DataFrame([{"team": "home", "penalty": True}])
    return SimpleNamespace(
        match_id=match_id, score={"home": 1, "away": 0}, goals=goals,
        event_origins=origins, events=pd.DataFrame(), tracking=pd.DataFrame(),
        metadata={"source_score": {"home": 1, "away": 0},
                  "source_pitch_dimensions_m": {"length": 110.0, "width": 70.0},
                  "event_capabilities": {"origin_coordinates": True,
                                         "tracking_sync": False,
                                         "attack_direction": False,
                                         "pass_endpoints": False}},
        quality={"source_events_total": 2, "source_event_codes": {"Play_Pass": 1, "Mystery": 1},
                 "event_exclusion_reasons": {"unsupported_event_code": 1},
                 "event_origin_count": 1, "source_fps_by_period": {1: 25.0, 2: 25.0},
                 "source_missing_positions": 3, "tracking_rows": 50,
                 "periods": [1, 2], "fps": 25.0, "duplicate_objects": 0,
                 "conflicting_clocks": 0, "nonfinite_positions": 0,
                 "frame_gaps": 2, "missing_positions": 0, "out_of_bounds": 0},
    )


def test_every_candidate_has_status_and_incomplete_triplet_fails_closed(tmp_path, monkeypatch):
    ids = [f"DFL-MAT-{i}" for i in range(7)]
    for match_id in ids[:6]:
        files(tmp_path, match_id)
    files(tmp_path, ids[6], FAMILIES[:2])
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: bundle(mid))
    result = idsse_validation.validate_idsse_dataset(tmp_path)
    assert result["candidate_count"] == 7
    assert [row["match_id"] for row in result["candidates"]] == sorted(ids)
    assert sum(row["status"] == "accepted" for row in result["candidates"]) == 6
    assert result["candidates"][-1]["reasons"] == ["missing_xml_family:positions"]
    assert len(result["source_fingerprints"]) == 20


def test_missing_root_and_missing_dependency_are_reported(tmp_path, monkeypatch):
    absent = idsse_validation.validate_idsse_dataset(tmp_path / "absent")
    assert absent["candidate_count"] == 0
    assert absent["errors"] == ["data_root_missing"]
    files(tmp_path)
    def unavailable(root, match_id):
        raise ImportError("pip install -e '.[idsse]'")
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", unavailable)
    rejected = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert rejected["status"] == "rejected"
    assert "missing_idsse_dependency" in rejected["reasons"]
    assert "pip install -e '.[idsse]'" in rejected["detail"]


def test_existing_empty_root_reports_incomplete_seven_match_dataset(tmp_path):
    report = idsse_validation.validate_idsse_dataset(tmp_path)
    assert report["candidate_count"] == 0
    assert report["errors"] == ["expected_seven_candidates_found:0"]


def test_score_mismatch_and_unsupported_counts_are_visible(tmp_path, monkeypatch):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.metadata["source_score"] = {"home": 2, "away": 0}
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert "source_score_goal_mismatch" in row["reasons"]
    assert row["source_score"] == {"home": 2, "away": 0}
    assert row["scored_goals"] == {"home": 1, "away": 0}
    assert row["event_exclusion_reasons"] == {"unsupported_event_code": 1}
    assert row["source_event_codes"] == {"Mystery": 1, "Play_Pass": 1}


def test_invalid_origin_time_and_coordinates_reject(tmp_path, monkeypatch):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.event_origins.loc[0, "source_timestamp"] = "bad"
    source.event_origins.loc[0, "source_x_m"] = 111.0
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert "invalid_event_origin_time" in row["reasons"]
    assert "event_origin_out_of_bounds" in row["reasons"]


def test_fingerprint_changes_with_source_mtime_and_json_excludes_observations(tmp_path, monkeypatch):
    files(tmp_path)
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: bundle(mid))
    first = idsse_validation.validate_idsse_dataset(tmp_path)
    second = idsse_validation.validate_idsse_dataset(tmp_path)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    encoded = json.dumps(first)
    assert "secret-player" not in encoded
    assert "source_timestamp" not in encoded
    assert first["readiness_scope"] == "source_event_origins_only"
    assert first["limitations"] == {
        "tracking_sync": "not_validated", "attack_direction": "not_established",
        "pass_endpoints": "unavailable"}
    assert first["candidates"][0]["status"] == "accepted"
    fingerprint = first["source_fingerprints"][0]
    assert set(fingerprint) == {"relative_name", "size_bytes", "mtime_ns"}
    changed = tmp_path / fingerprint["relative_name"]
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    refreshed = idsse_validation.validate_idsse_dataset(tmp_path)
    assert refreshed["source_fingerprints"][0]["mtime_ns"] != fingerprint["mtime_ns"]


def test_duplicate_family_and_zero_usable_origins_both_reject(tmp_path, monkeypatch):
    files(tmp_path, "DFL-MAT-ONE")
    (tmp_path / "alternate_events_raw_DFL-MAT-ONE.xml").write_bytes(b"duplicate")
    files(tmp_path, "DFL-MAT-TWO")
    empty = bundle("DFL-MAT-TWO")
    empty.event_origins = empty.event_origins.iloc[:0]
    empty.quality.update(source_events_total=1, source_event_codes={"Mystery": 1},
                         event_exclusion_reasons={"unsupported_event_code": 1},
                         event_origin_count=0)
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: empty)
    rows = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"]
    assert rows[0]["reasons"] == ["duplicate_xml_family:events"]
    assert "no_usable_event_origins" in rows[1]["reasons"]
    assert all(row["status"] == "rejected" for row in rows)


def test_inconsistent_event_totals_reject_even_with_unsupported_reason(tmp_path, monkeypatch):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.quality["source_events_total"] = 3
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert "unexplained_event_count" in row["reasons"]


def test_missing_source_position_count_has_slot_denominator_and_meaning(tmp_path, monkeypatch):
    files(tmp_path)
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: bundle(mid))
    report = idsse_validation.validate_idsse_dataset(tmp_path)
    assert report["source_position_slot_semantics"] == (
        "Observed-frame XY slots across every source object column, including roster slots never observed; not active-player detections.")
    assert report["candidates"][0]["quality"]["source_position_slots_total"] == 53


def test_parser_failure_rejects_one_candidate_and_continues(tmp_path, monkeypatch):
    files(tmp_path, "DFL-MAT-ONE")
    files(tmp_path, "DFL-MAT-TWO")
    def parse(root, match_id):
        if match_id == "DFL-MAT-ONE":
            raise RuntimeError("malformed XML")
        return bundle(match_id)
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", parse)
    rows = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"]
    assert rows[0]["status"] == "rejected"
    assert rows[0]["reasons"] == ["bundle_validation_failed"]
    assert "malformed XML" in rows[0]["detail"]
    assert rows[1]["status"] == "accepted"


def test_cli_returns_failure_when_every_candidate_is_rejected(tmp_path, monkeypatch):
    output = tmp_path / "manifest.json"
    report = {"candidate_count": 7, "candidates": [
        {"status": "rejected", "reasons": ["missing_idsse_dependency"]}
        for _ in range(7)], "errors": []}
    monkeypatch.setattr(cli, "validate_idsse_dataset", lambda root: report)
    monkeypatch.setattr(sys, "argv", ["validate_idsse.py", "--data-root", str(tmp_path),
                                      "--output", str(output)])
    assert cli.main() == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("key", ["source_fps_by_period", "frame_gaps",
                                 "source_missing_positions", "tracking_rows"])
def test_missing_source_quality_diagnostic_rejects(tmp_path, monkeypatch, key):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    del source.quality[key]
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert f"missing_quality_diagnostic:{key}" in row["reasons"]


@pytest.mark.parametrize("key,value", [
    ("source_fps_by_period", {1: 0.0, 2: 25.0}),
    ("source_fps_by_period", {1: 25.0}),
    ("source_fps_by_period", None),
    ("frame_gaps", -1),
    ("frame_gaps", 1.5),
    ("source_missing_positions", -1),
    ("tracking_rows", 0),
    ("periods", None),
    ("fps", float("nan")),
])
def test_invalid_source_quality_diagnostic_rejects(tmp_path, monkeypatch, key, value):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.quality[key] = value
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert f"invalid_quality_diagnostic:{key}" in row["reasons"]


@pytest.mark.parametrize("code,event,reason", [
    ("UnknownKick", "pass", "unsupported_event_origin_source_code"),
    ("ShotAtGoal_SavedShot", "pass", "event_origin_source_code_mismatch"),
    ("", "pass", "missing_event_origin_source_code"),
    ("Penalty_ShotAtGoal_SuccessfulShot", "shot", "unsupported_event_origin_source_code"),
])
def test_unsupported_blank_or_mismatched_retained_source_code_rejects(
        tmp_path, monkeypatch, code, event, reason):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.event_origins.loc[0, "source_code"] = code
    source.event_origins.loc[0, "event"] = event
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert reason in row["reasons"]


@pytest.mark.parametrize("code,event", [
    ("Cross", "pass"), ("ShotAtGoal_SavedShot", "shot")])
def test_documented_retained_source_code_mapping_accepts(tmp_path, monkeypatch, code, event):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    source.event_origins.loc[0, "source_code"] = code
    source.event_origins.loc[0, "event"] = event
    source.quality["source_event_codes"] = {code: 1, "Mystery": 1}
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "accepted"


def test_source_code_mapping_is_checked_for_every_retained_origin(tmp_path, monkeypatch):
    files(tmp_path)
    source = bundle("DFL-MAT-ONE")
    second = source.event_origins.iloc[[0]].copy()
    second.loc[:, "source_code"] = "UnknownKick"
    source.event_origins = pd.concat((source.event_origins, second), ignore_index=True)
    source.quality.update(source_events_total=3, event_origin_count=2,
                          source_event_codes={"Play_Pass": 1, "UnknownKick": 1,
                                              "Mystery": 1})
    monkeypatch.setattr(idsse_validation, "load_idsse_bundle", lambda root, mid: source)
    row = idsse_validation.validate_idsse_dataset(tmp_path)["candidates"][0]
    assert row["status"] == "rejected"
    assert "unsupported_event_origin_source_code" in row["reasons"]
