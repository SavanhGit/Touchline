import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]


def test_idsse_setup_and_release_docs_are_reproducible_and_attributed():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs/VALIDATION.md").read_text(encoding="utf-8")
    assert 'pip install -e ".[idsse]"' in readme
    assert "data/raw/idsse" in readme
    assert "scripts/validate_idsse.py --data-root data/raw/idsse --output docs/validation/idsse-v1.json" in readme
    assert "all seven" in readme.lower() and "fingerprint" in readme.lower()
    assert "raw XML" in readme and "Git" in readme
    assert "Deutsche Fußball Liga (DFL)" in readme
    assert "CC BY 4.0" in readme
    assert "Bassek et al." in readme and "Scientific Data" in readme
    assert "7/7" in validation and "21/21" in validation
    assert "1600 MiB" in validation and "traced Python allocation" in validation
    for deferred in ("tracking synchronization", "attack direction", "pass endpoints",
                     "IDSSE xT", "progression"):
        assert deferred in validation
    assert "held-out IDSSE evaluation" in validation


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
    assert len(payload) == 6
    assert all(set(row["training_match_ids"]) == {"metrica_1", "metrica_2", "metrica_3"} - {row["match_id"]}
               for row in payload)
    assert "NaN" not in (tmp_path / "release-validation.json").read_text(encoding="utf-8")
    assert "Infinity" not in (tmp_path / "release-validation.json").read_text(encoding="utf-8")


@pytest.mark.parametrize('event,success,source_type,subtype', [
    ('tackle', True, 'CHALLENGE', 'TACKLE-WON'),
    ('tackle', False, 'CHALLENGE', 'TACKLE-LOST'),
    ('interception', True, 'RECOVERY', 'INTERCEPTION'),
    ('recovery', True, 'RECOVERY', 'THEFT'),
])
def test_defensive_source_audit_accepts_exact_mapping_and_missing_endpoints(event, success, source_type, subtype):
    b, e, raw = _bundle_and_source(event, success)
    b.events.loc[0, ['x1', 'y1']] = float('nan')
    e.x1 = e.y1 = float('nan')
    raw.loc[0, ['Type', 'Subtype', 'End X', 'End Y']] = [source_type, subtype, float('nan'), float('nan')]
    verify_evidence(b, e, raw)
    raw.loc[0, 'Subtype'] = 'GROUND'
    with pytest.raises(ValueError, match='original'):
        verify_evidence(b, e, raw)


@pytest.mark.parametrize('event', ['pass', 'shot'])
def test_pass_and_shot_audit_still_rejects_missing_endpoints(event):
    b, e, raw = _bundle_and_source(event)
    b.events.loc[0, ['x1', 'y1']] = float('nan')
    e.x1 = e.y1 = float('nan')
    raw.loc[0, ['Type', 'End X', 'End Y']] = ['SHOT' if event == 'shot' else 'PASS', float('nan'), float('nan')]
    with pytest.raises(ValueError):
        verify_evidence(b, e, raw)


def test_defensive_audit_rejects_changed_source_origin_or_present_endpoint():
    b, e, raw = _bundle_and_source('tackle')
    raw.loc[0, ['Type', 'Subtype']] = ['CHALLENGE', 'TACKLE-WON']
    verify_evidence(b, e, raw)
    for column in ['Start X', 'Start Y', 'End X', 'End Y']:
        changed = raw.copy()
        changed.loc[0, column] += .01
        with pytest.raises(ValueError, match='coordinates'):
            verify_evidence(b, e, changed)


def test_v4_fold_uses_sequence_rates_and_pooled_other_team_exposure():
    from scripts.validate_cross_match import summarize_v4_fold
    from types import SimpleNamespace
    report = SimpleNamespace(match_id='held', team='home',
        possessions=pd.DataFrame({'team':['home']*4+['away'], 'shots':[0,1,2,0,1]}),
        transitions=pd.DataFrame({'team':['home']*2, 'shots':[0,2]}),
        defensive_actions=pd.DataFrame([dict(team='home', action_type='tackle', total_actions=2,
            observed_tracking_minutes=45, actions_per_90=4)]),
        shot_summaries=pd.DataFrame([dict(team='home',channel='left',shots=3,goals=1),
                                    dict(team='home',channel='centre',shots=1,goals=0)]))
    other = pd.DataFrame([dict(team='home',action_type='tackle',total_actions=2,observed_tracking_minutes=30),
                          dict(team='away',action_type='tackle',total_actions=8,observed_tracking_minutes=90)])
    result = summarize_v4_fold(report, [('other',other)])
    assert result['transition_shot_rate'] == {'numerator':1,'denominator':2,'rate':.5}
    assert result['possession_shot_rate'] == {'numerator':2,'denominator':4,'rate':.5}
    assert result['defense'][0]['dataset_baseline_per_90'] == 7.5
    assert result['defense'][0]['baseline_actions'] == 10
    assert result['defense'][0]['baseline_observed_team_minutes'] == 120
    assert result['shots_by_channel'][0]['team_shots'] == 4
    assert result['shots_by_channel'][0]['team_goals'] == 1
    assert 'dataset' in result['baseline_label'].lower()
    with pytest.raises(ValueError, match='held-out'):
        summarize_v4_fold(report, [('held',other)])


def test_suggestion_audit_rejects_wrong_support_count_and_unrelated_citations():
    from scripts.validate_release import audit_suggestions
    from soccer_tactics.analytics.report import analyze_match
    from factories import make_bundle
    report = analyze_match(make_bundle(), 'home', [make_bundle('other')])
    audit = audit_suggestions(report)
    assert all(row['descriptive_accuracy'] == 'pass' for row in audit)
    finding = report.findings[0]
    finding.metrics['numerator'] += 1
    with pytest.raises(ValueError, match='count'):
        audit_suggestions(report)
    finding.metrics['numerator'] -= 1
    finding.evidence = [next(e for e in report.event_evidence if e.team != 'home')]
    with pytest.raises(ValueError, match='support'):
        audit_suggestions(report)


def test_suggestion_audit_requires_visible_tradeoff():
    from scripts.validate_release import audit_suggestions
    from soccer_tactics.analytics.report import analyze_match
    from factories import make_bundle
    report = analyze_match(make_bundle(), 'home', [make_bundle('other')])
    report.findings[0].tradeoff = ' '
    with pytest.raises(ValueError, match='tradeoff'):
        audit_suggestions(report)


def test_sequence_audit_rejects_cross_period_and_overlong_transition():
    from scripts.validate_release import verify_sequences
    from soccer_tactics.analytics.report import analyze_match
    from factories import make_bundle
    selected = make_bundle()
    selected.events = selected.events.iloc[[0, 10]].copy()
    selected.events.loc[selected.events.index[-1], ['time_s', 'frame']] = [1., 1]
    report = analyze_match(selected, 'home', [make_bundle('other')])
    assert len(report.transitions) == 1
    verify_sequences(report)
    old_end = report.transitions.at[0, 'end_time_s']
    report.transitions.at[0, 'end_time_s'] = report.transitions.at[0, 'start_time_s'] + 10.01
    with pytest.raises(ValueError, match='10-second'):
        verify_sequences(report)
    report.transitions.at[0, 'end_time_s'] = old_end
    report.possessions.at[0, 'period'] = 2
    with pytest.raises(ValueError, match='period'):
        verify_sequences(report)


def test_suggestion_audit_checks_cited_fields_and_handles_no_findings():
    from scripts.validate_release import audit_suggestions
    from soccer_tactics.analytics.report import analyze_match
    from factories import make_bundle
    report = analyze_match(make_bundle(), 'home', [make_bundle('other')])
    report.findings[0].evidence[0].x0 += 1
    with pytest.raises(ValueError, match='fields'):
        audit_suggestions(report)
    report.findings = []
    report.event_evidence = []
    assert audit_suggestions(report) == []
