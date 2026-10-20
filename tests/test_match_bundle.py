from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soccer_tactics import match
from soccer_tactics.analytics.report import analyze_match
from soccer_tactics.loaders import load_canonical, load_metrica_bundle
from soccer_tactics.schema import validate, validate_events
from conftest import GAMES, ROOT
from factories import make_bundle


def test_unknown_score_and_known_goalless_match_are_distinct():
    assert make_bundle().score is None
    assert replace(make_bundle(), score_known=True).score == {"home": 0, "away": 0}


def test_invalid_bundle_without_quality_metadata():
    b = make_bundle()
    duplicate = replace(b, tracking=pd.concat([b.tracking, b.tracking.iloc[:1]]), quality={})
    with pytest.raises(ValueError, match="duplicate"):
        match.validate_bundle(duplicate)


@pytest.mark.parametrize("column", ["x", "y", "time_s"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_tracking_is_rejected(column, value):
    b = make_bundle()
    b.tracking.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        match.validate_bundle(b)


def test_conflicting_clock_is_rejected_without_metadata():
    b = make_bundle()
    b.tracking.loc[0, "time_s"] = .1
    with pytest.raises(ValueError, match="conflicting"):
        match.validate_bundle(b)


def test_decreasing_source_time_is_rejected_but_period_reset_is_allowed():
    b = make_bundle()
    b.tracking.loc[b.tracking.frame == 2, "time_s"] = .5
    with pytest.raises(ValueError, match="nondecreasing"):
        match.validate_bundle(b)
    b = make_bundle()
    second_period = b.tracking.assign(period=2)
    match.validate_bundle(replace(b, tracking=pd.concat([b.tracking, second_period])))


@pytest.mark.parametrize("table", ["tracking", "events", "goals"])
def test_required_columns_are_checked(table):
    b = make_bundle()
    with pytest.raises(ValueError, match="missing columns"):
        match.validate_bundle(replace(b, **{table: getattr(b, table).drop(columns="time_s")}))


@pytest.mark.parametrize("table", ["tracking", "events"])
def test_rows_belong_to_declared_match(table):
    b = make_bundle()
    with pytest.raises(ValueError, match="match"):
        match.validate_bundle(replace(b, **{table: getattr(b, table).assign(match_id="foreign")}))


@pytest.mark.parametrize("fps", [0., -1., np.nan, np.inf])
def test_invalid_sampling_rates_are_rejected(fps):
    with pytest.raises(ValueError, match="fps"):
        match.validate_bundle(replace(make_bundle(), fps=fps))


@pytest.mark.parametrize("column", ["x0", "y0", "x1", "y1", "time_s"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_pass_coordinates_and_event_time_are_rejected(column, value):
    b = make_bundle()
    b.events.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        match.validate_bundle(b)


@pytest.mark.parametrize("column", ["x0", "y0"])
def test_shot_origins_are_required(column):
    b = make_bundle()
    b.events.loc[b.events.event == "shot", column] = np.nan
    with pytest.raises(ValueError, match="finite"):
        match.validate_bundle(b)


def test_missing_shot_endpoint_preserves_origin_and_omits_arrow():
    pytest.importorskip("plotly")
    from soccer_tactics.dashboard.charts import evidence_pitch

    b = make_bundle()
    b.events.loc[b.events.event == "shot", ["x1", "y1"]] = np.nan
    match.validate_bundle(b)
    report = analyze_match(b, "home", [make_bundle("training")])
    moment = next(f for f in report.findings if f.category == "danger").evidence[0]
    assert (moment.x0, moment.y0) == (-40., -20.)
    assert np.isnan(moment.x1) and np.isnan(moment.y1)
    assert not evidence_pitch(b.tracking.iloc[:3], moment).layout.annotations


@pytest.mark.parametrize("column,value", [("match_id", "foreign"), ("team", "ball"),
                                         ("time_s", np.nan), ("time_s", np.inf)])
def test_goal_records_belong_to_the_bundle(column, value):
    b = make_bundle()
    goals = pd.DataFrame([dict(match_id="test", team="home", period=1, frame=3,
                               time_s=3., player_id="h", penalty=False)])
    goals.loc[0, column] = value
    with pytest.raises(ValueError, match="goal"):
        match.validate_bundle(replace(b, goals=goals))


@pytest.mark.parametrize("table,column,value", [("tracking", "team", "ref"),
                                               ("tracking", "player_id", None),
                                               ("tracking", "frame", np.nan),
                                               ("events", "team", "ball"),
                                               ("events", "event", "mystery"),
                                               ("events", "period", np.nan)])
def test_invalid_identity_and_labels_are_rejected(table, column, value):
    b = make_bundle()
    getattr(b, table).loc[0, column] = value
    with pytest.raises(ValueError):
        match.validate_bundle(b)


def test_empty_tracking_is_rejected_and_empty_events_are_allowed():
    b = make_bundle()
    with pytest.raises(ValueError, match="empty tracking"):
        match.validate_bundle(replace(b, tracking=b.tracking.iloc[:0]))
    match.validate_bundle(replace(b, events=b.events.iloc[:0]))


def test_empty_events_with_schema_only_are_allowed():
    b = make_bundle()
    events = pd.DataFrame(columns=b.events.columns)
    match.validate_bundle(replace(b, events=events))


@pytest.mark.parametrize("bad_training", [False, True])
def test_analysis_validates_selected_and_training_data(bad_training):
    selected, training = make_bundle(), make_bundle("training")
    invalid = training if bad_training else selected
    invalid.tracking = pd.concat([invalid.tracking, invalid.tracking.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_match(selected, "home", [training])


def test_nominal_bound_violations_are_reported_without_dropping_positions():
    b = make_bundle()
    b.tracking.loc[0, "x"] = 60.
    match.validate_bundle(b)
    assert match.inspect_quality(b.tracking, b.events)["out_of_bounds"] == 1
    assert b.tracking.loc[0, "x"] == 60.


def test_quality_reports_retained_nonfinite_positions():
    b = make_bundle()
    b.tracking.loc[0, "x"] = np.inf
    b.tracking.loc[1, "y"] = np.nan
    quality = match.inspect_quality(b.tracking, b.events)
    assert quality["nonfinite_positions"] == 2
    assert quality["missing_positions"] == 1


def test_canonical_validation_rejects_nonfinite_coordinates(tmp_path):
    b = make_bundle()
    b.tracking.loc[0, "x"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate(b.tracking)
    b.tracking.to_csv(tmp_path / "tracking.csv", index=False)
    with pytest.raises(ValueError, match="finite"):
        load_canonical(tmp_path / "tracking.csv")
    b.events.loc[0, "x1"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_events(b.events)


def test_factory_adjusts_frames_and_keeps_fifteen_observed_minutes():
    b = make_bundle(fps=5.)
    match.validate_bundle(b)
    assert match.infer_fps(b.tracking) == 5.
    assert b.tracking.query("player_id == 'h'").shape[0] / b.fps == 900
    assert b.events.iloc[1].frame == 5 and b.events.iloc[1].time_s == 1.


@pytest.mark.skipif(len(GAMES) < 2, reason="requires both real Metrica games")
def test_real_bundles_keep_complete_scores_and_penalty_context():
    first, second = [load_metrica_bundle(ROOT, game) for game in (1, 2)]
    assert first.score == {"home": 3, "away": 0}
    assert second.score == {"home": 3, "away": 2}
    assert second.goals.penalty.sum() == 1
    assert len(second.goals) == 5
    assert second.events.query("event == 'shot'").success.sum() == 4
    match.validate_bundle(first)
    match.validate_bundle(second)
