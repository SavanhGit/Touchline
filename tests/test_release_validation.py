import json

import pandas as pd
import pytest

from soccer_tactics.analytics.report import Evidence
from soccer_tactics.match import MatchBundle
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS
from scripts.validate_release import (
    _featured_markdown,
    _json_safe,
    _verify_modeling_shots,
    run,
    verify_evidence,
)


def _bundle_and_source(event="pass", success=True):
    tracking = pd.DataFrame(
        [("test", 1, 10, 1.0, "home", "h1", 0.0, 0.0),
         ("test", 1, 10, 1.0, "away", "a1", 2.0, 0.0)], columns=COLUMNS)
    row = ("test", 1, "home", event, success, "h1", None, 10, 1.0,
           10.5, 6.8, 0.0, -6.8)
    bundle = MatchBundle("test", tracking, pd.DataFrame([row], columns=EVENT_COLUMNS),
                         pd.DataFrame(columns=["team", "period", "frame", "time_s", "player_id", "penalty"]),
                         "test", 10.0)
    evidence = Evidence("test:1:10:0", "test", 1, 10, 1.0, "home", "h1", event,
                        10.5, 6.8, 0.0, -6.8)
    raw = pd.DataFrame([{
        "Team": "Home", "Type": "PASS" if success else "BALL LOST", "Subtype": None,
        "Period": 1, "Start Frame": 10, "Start Time [s]": 1.0,
        "From": "h1", "Start X": .6, "Start Y": .4, "End X": .5, "End Y": .6,
    }, {
        "Team": "Home", "Type": "SHOT", "Subtype": None,
        "Period": 1, "Start Frame": 20, "Start Time [s]": 2.0,
        "From": "h1", "Start X": .9, "Start Y": .4, "End X": .9, "End Y": .4,
    }, {
        "Team": "Away", "Type": "SHOT", "Subtype": None,
        "Period": 1, "Start Frame": 30, "Start Time [s]": 3.0,
        "From": "a1", "Start X": .1, "Start Y": .4, "End X": .1, "End Y": .4,
    }])
    return bundle, evidence, raw


def test_source_reference_with_converted_coordinates_succeeds():
    bundle, evidence, raw = _bundle_and_source()
    verify_evidence(bundle, evidence, raw)


@pytest.mark.parametrize("field,value", [
    ("player_id", "h2"), ("team", "away"), ("period", 2),
    ("time_s", 9.0), ("event", "shot"), ("x0", 12.0),
    ("x1", -1.0), ("y0", 7.0), ("y1", -8.0),
    ("event_id", "test:1:10:1"),
])
def test_evidence_mismatches_fail(field, value):
    bundle, evidence, raw = _bundle_and_source()
    setattr(evidence, field, value)
    with pytest.raises(ValueError):
        verify_evidence(bundle, evidence, raw)


@pytest.mark.parametrize("field,value", [
    ("Team", "Away"), ("Period", 2), ("Start Time [s]", 9.0),
    ("From", "h2"), ("Type", "OTHER"), ("Start X", .61), ("End Y", .61),
])
def test_original_source_row_mismatches_fail(field, value):
    bundle, evidence, raw = _bundle_and_source()
    raw.loc[0, field] = value
    with pytest.raises(ValueError):
        verify_evidence(bundle, evidence, raw)


def test_failed_pass_must_match_ball_lost_source_semantics():
    bundle, evidence, raw = _bundle_and_source(success=False)
    verify_evidence(bundle, evidence, raw)
    raw.loc[0, "Type"] = "PASS"
    with pytest.raises(ValueError):
        verify_evidence(bundle, evidence, raw)


def test_evidence_period_without_shots_fails_orientation_audit():
    bundle, evidence, raw = _bundle_and_source()
    bundle.events.loc[0, "period"] = 2
    evidence.period = 2
    evidence.event_id = "test:2:10:0"
    raw.loc[0, "Period"] = 2
    with pytest.raises(ValueError, match="orientation"):
        verify_evidence(bundle, evidence, raw)


def test_modeling_shots_equal_source_shots_after_penalty_exclusion():
    normal_shot = ("test", 1, "home", "shot", False, "h1", None, 20, 2.0,
                   10.5, 6.8, 20.0, 0.0)
    penalty_shot = ("test", 1, "home", "shot", True, "h1", None, 40, 4.0,
                    20.0, 0.0, 52.5, 0.0)
    tracking = pd.DataFrame(columns=COLUMNS)
    events = pd.DataFrame([normal_shot], columns=EVENT_COLUMNS)
    bundle = MatchBundle("test", tracking, events, pd.DataFrame(), "test", 25.0)
    raw = pd.DataFrame([
        {"Team": "Home", "Type": "SHOT", "Subtype": None, "Period": 1,
         "Start Frame": 20, "Start Time [s]": 2.0, "From": "h1"},
        {"Team": "Home", "Type": "SET PIECE", "Subtype": "PENALTY", "Period": 1,
         "Start Frame": 39, "Start Time [s]": 3.96, "From": "h1"},
        {"Team": "Home", "Type": "SHOT", "Subtype": "GOAL", "Period": 1,
         "Start Frame": 40, "Start Time [s]": 4.0, "From": "h1"},
    ])
    _verify_modeling_shots(bundle, raw)
    bundle.events = pd.DataFrame([normal_shot, penalty_shot], columns=EVENT_COLUMNS)
    with pytest.raises(ValueError, match="penalty-excluded"):
        _verify_modeling_shots(bundle, raw)


def test_featured_prose_uses_measured_quality_values():
    featured = {"match_id": "metrica_1", "team": "away", "score": {"home": 3, "away": 0},
                "quality": {"event_tracking_coverage": .8, "alignment_median_m": .4,
                            "alignment_p90_m": .8}, "findings": []}
    prose = _featured_markdown(featured)
    assert "80.0%" in prose
    assert "100% event-to-player tracking coverage" not in prose


@pytest.mark.parametrize("score,outcome,phrase", [
    ({"home": 0, "away": 3}, "win", "The away team won."),
    ({"home": 1, "away": 1}, "draw", "The match was a draw."),
])
def test_featured_outcome_prose_uses_measured_result(score, outcome, phrase):
    featured = {"score": score, "outcome": outcome, "quality": {}, "findings": []}
    assert phrase in _featured_markdown(featured)


def test_unavailable_json_numbers_become_null():
    payload = _json_safe({"metrics": [float("nan"), float("inf"), float("-inf")],
                          "nested": {"coverage": float("nan")}})
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == {"metrics": [None, None, None], "nested": {"coverage": None}}


def test_real_export_is_finite_and_has_other_match_training_only(tmp_path):
    run(__import__("pathlib").Path("data/raw/metrica"), tmp_path)
    payload = json.loads((tmp_path / "release-validation.json").read_text(encoding="utf-8"))
    assert len(payload) == 4
    assert all(row["training_match_ids"] == ["metrica_2" if row["match_id"] == "metrica_1" else "metrica_1"]
               for row in payload)
    assert "NaN" not in (tmp_path / "release-validation.json").read_text(encoding="utf-8")
    assert "Infinity" not in (tmp_path / "release-validation.json").read_text(encoding="utf-8")
