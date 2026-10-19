from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.impact import add_pressure, pass_values, player_impact
from soccer_tactics.analytics.report import analyze_match, channel
from soccer_tactics.loaders import load_canonical, load_metrica_bundle
from soccer_tactics.match import infer_fps, inspect_quality
from factories import make_bundle as bundle
from conftest import GAMES


def test_report_hand_computed_counts_channels_and_evidence():
    report = analyze_match(bundle(), "home", [bundle("training")])
    own = report.comparisons.set_index("team").loc["home"]
    assert own.buildup_losses == 6 and own.buildup_passes == 10
    assert own.final_third_entries == 2 and own.completed == 4
    assert np.isclose(own.buildup_loss_rate, .6)
    findings = {f.category: f for f in report.findings}
    assert set(findings) == {"buildup", "progression", "danger"}
    assert findings["buildup"].metrics["numerator"] == 6
    assert findings["progression"].metrics["numerator"] == 8
    assert findings["danger"].metrics["numerator"] == 3
    assert report.score is None and report.outcome == "unknown"
    assert report.provenance["training_match_ids"] == ["training"]
    assert len(report.findings) <= 3
    assert len({f.category for f in report.findings}) == len(report.findings)
    for finding in report.findings:
        assert 1 <= len(finding.evidence) <= 3
        assert finding.suggestion and finding.tradeoff
        for moment in finding.evidence:
            assert moment.match_id == "test"
            assert moment.event_id.startswith(f"test:{moment.period}:{moment.frame}:")
            assert ((bundle().events.frame == moment.frame) & (bundle().events.period == moment.period)).any()


def test_analyze_match_accepts_a_training_iterator():
    report = analyze_match(bundle(), "home", iter([bundle("training")]))
    assert report.provenance["training_match_ids"] == ["training"]


@pytest.mark.parametrize("passes,losses,qualifies", [(5, 3, True), (4, 3, False), (5, 2, False)])
def test_exact_buildup_eligibility_thresholds(passes, losses, qualifies):
    b = bundle()
    events = b.events.iloc[:passes].copy()
    events["success"] = np.arange(passes) >= losses
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    findings = [f for f in report.findings if f.category == "buildup"]
    assert bool(findings) is qualifies
    if qualifies:
        assert findings[0].metrics["numerator"] == 3
        assert findings[0].metrics["denominator"] == 5
        assert findings[0].metrics["rate"] == .6


@pytest.mark.parametrize("passes,entries,qualifies", [(5, 2, True), (4, 1, False), (5, 3, False)])
def test_progression_supporters_are_nonentries(passes, entries, qualifies):
    b = bundle()
    events = b.events.iloc[:passes].assign(success=True, x1=-5.)
    events.loc[events.index[:entries], "x1"] = 17.5
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    findings = [f for f in report.findings if f.category == "progression"]
    assert bool(findings) is qualifies
    if qualifies:
        assert findings[0].metrics == {"numerator": 3, "denominator": 5, "entries": 2, "entry_rate": .4}
        assert all(moment.x1 == -5. for moment in findings[0].evidence)


def test_entry_and_buildup_boundaries_count_only_successful_crossings():
    b = bundle()
    events = b.events.iloc[:5].assign(success=True, x0=[-0.01, 0., 17.5, 17.49, -20.],
                                     x1=[17.5, 17.49, 30., 17.5, 30.])
    events.loc[4, "success"] = False
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    own = report.comparisons.set_index("team").loc["home"]
    assert own.buildup_passes == 2
    assert own.buildup_losses == 1
    assert own.final_third_entries == 2
    assert report.passes.entry.tolist() == [True, False, False, True, False]
    right = report.channels.query("team == 'home' and channel == 'right'").iloc[0]
    assert right.outside_third_passes == 4
    assert right.entry_rate == .5


def test_all_success_passes_have_zero_loss_and_no_buildup_finding():
    b = bundle()
    report = analyze_match(replace(b, events=b.events.iloc[:10].assign(success=True)),
                           "home", [bundle("training")])
    own = report.comparisons.set_index("team").loc["home"]
    assert own.completed == 10 and own.completion_rate == 1.
    assert own.buildup_losses == 0 and own.buildup_loss_rate == 0.
    assert own.buildup_cost == 0.
    assert not any(f.category == "buildup" for f in report.findings)


@pytest.mark.parametrize("events", ["empty", "shots_only"])
def test_zero_pass_denominators_are_unavailable(events):
    b = bundle()
    rows = b.events.iloc[:0] if events == "empty" else b.events.query("event == 'shot'")
    report = analyze_match(replace(b, events=rows), "home", [bundle("training")])
    assert report.passes.empty
    assert report.comparisons.passes.eq(0).all()
    assert report.comparisons.completion_rate.isna().all()
    assert report.comparisons.buildup_loss_rate.isna().all()
    assert report.channels.entry_rate.isna().all()
    assert report.channels.buildup_loss_rate.isna().all()
    assert report.comparisons.buildup_cost.eq(0).all()
    assert report.score is None and report.outcome == "unknown"
    assert [f.category for f in report.findings] == ([] if events == "empty" else ["danger"])


def test_training_fit_receives_only_other_match_events(monkeypatch):
    import soccer_tactics.analytics.report as reports

    training = [bundle("first"), bundle("second")]
    observed = []
    real_fit = reports.fit_xt

    def record_fit(events):
        observed.append(events.copy(deep=True))
        return real_fit(events)

    monkeypatch.setattr(reports, "fit_xt", record_fit)
    report = analyze_match(bundle("selected"), "home", iter(training))
    assert len(observed) == 1
    pd.testing.assert_frame_equal(observed[0], pd.concat([b.events for b in training], ignore_index=True))
    assert set(observed[0].match_id) == {"first", "second"}
    assert report.provenance["training_match_ids"] == ["first", "second"]


def test_shot_free_training_has_clear_calibration_error():
    training = bundle("training")
    training = replace(training, events=training.events.query("event == 'pass'"))
    with pytest.raises(ValueError, match="shot.*calibrate"):
        analyze_match(bundle(), "home", [training])


def test_evidence_fields_and_ids_are_reproducible_without_input_mutation():
    b, training = bundle(), bundle("training")
    # Distinct source rows can share a frame and source clock.
    b.events.loc[:2, ["frame", "time_s"]] = [0, 0.]
    b.events.loc[:2, "x1"] = [-40., -41., -42.]
    b.events.index = np.arange(100, 100 + len(b.events))
    snapshots = [(part.copy(deep=True), part) for match in (b, training)
                 for part in (match.events, match.tracking, match.goals)]
    first = analyze_match(b, "home", [training])
    second = analyze_match(b, "home", [training])
    for finding, again in zip(first.findings, second.findings):
        assert [m.event_id.encode() for m in finding.evidence] == [m.event_id.encode() for m in again.evidence]
        for moment in finding.evidence:
            ordinal = int(moment.event_id.rsplit(":", 1)[1])
            source = b.events.iloc[ordinal]
            assert moment.event_id == f"test:{source.period}:{source.frame}:{ordinal}"
            for field in ("match_id", "period", "frame", "time_s", "team", "player_id", "event", "x0", "y0", "x1", "y1"):
                assert getattr(moment, field) == source[field]
    same_frame = first.passes.query("frame == 0").event_id.tolist()
    assert same_frame == ["test:1:0:0", "test:1:0:1", "test:1:0:2"]
    assert len(same_frame) == len(set(same_frame))
    assert [m.event_id for m in first.findings[0].evidence] == same_frame
    for before, after in snapshots:
        pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("ys,winner", [((20., 0., -20.), "right"), ((20., 0.), "centre")])
def test_final_channel_ties_use_right_centre_left_order(ys, winner):
    b = bundle()
    passes = b.events.iloc[:5].assign(success=False)
    shots = b.events.query("event == 'shot'").iloc[:3].assign(success=False)
    # Deliberately present channels in the reverse of the required final tie order.
    events = pd.concat([passes.assign(y0=y, y1=y) for y in ys] +
                       [shots.assign(y0=-y) for y in ys], ignore_index=True)
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    assert {f.category for f in report.findings} == {"buildup", "progression", "danger"}
    assert all(f.title.endswith(winner) for f in report.findings)
    assert all(f.metrics["numerator"] == 5 for f in report.findings if f.category != "danger")
    assert next(f for f in report.findings if f.category == "danger").metrics["numerator"] == 3


def test_progression_rate_ties_choose_largest_denominator():
    b = bundle()
    events = pd.concat([b.events.iloc[:5].assign(success=False),
                        b.events.iloc[:6].assign(success=False, y0=0.)], ignore_index=True)
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    finding = next(f for f in report.findings if f.category == "progression")
    assert finding.title.endswith("centre")
    assert finding.metrics["denominator"] == 6


def test_buildup_rate_ties_choose_highest_summed_turnover_cost():
    b = bundle()
    events = pd.concat([b.events.iloc[:5].assign(success=False, x0=-1., x1=40.),
                        b.events.iloc[:5].assign(success=False, x0=-40., x1=-40., y0=0., y1=0.)], ignore_index=True)
    report = analyze_match(replace(b, events=events), "home", [bundle("training")])
    costs = report.channels.query("team == 'home'").set_index("channel").turnover_cost
    assert costs.centre > costs.right
    finding = next(f for f in report.findings if f.category == "buildup")
    assert finding.title.endswith("centre")
    assert finding.metrics["cost"] == costs.centre


def test_evidence_selects_costly_losses_and_chronological_nonentries():
    b = bundle()
    b.events.loc[:5, "x1"] = [40., -40., 40., -40., 40., -40.]
    report = analyze_match(b, "home", [bundle("training")])
    findings = {f.category: f for f in report.findings}
    assert [m.frame for m in findings["buildup"].evidence] == [1, 3, 5]
    assert [m.frame for m in findings["progression"].evidence] == [1, 4, 6]


def test_danger_shots_take_priority_and_goals_are_first():
    b = bundle()
    b.events.loc[10:14, "success"] = [False, False, True, False, False]
    entries = b.events.iloc[:8].assign(team="away", player_id="a", success=True,
                                      x0=20., x1=-30., y0=20., y1=20.)
    b = replace(b, events=pd.concat([b.events, entries], ignore_index=True))
    danger = next(f for f in analyze_match(b, "home", [bundle("training")]).findings if f.category == "danger")
    assert danger.metrics == {"numerator": 3, "denominator": 5, "event_type": "shots"}
    assert [m.frame for m in danger.evidence] == [22, 20, 21]


def test_danger_uses_entries_when_no_shot_channel_qualifies():
    b = bundle()
    shots = b.events.iloc[10:15].assign(y0=[-20., -20., 0., 0., 20.])
    entries = b.events.iloc[:5].assign(team="away", player_id="a", success=True,
                                     x0=20., x1=-30., y0=20., y1=20.)
    report = analyze_match(replace(b, events=pd.concat([shots, entries], ignore_index=True)),
                           "home", [bundle("training")])
    danger = next(f for f in report.findings if f.category == "danger")
    assert danger.title.endswith("right")
    assert danger.metrics == {"numerator": 5, "denominator": 5, "event_type": "completed final-third entries"}
    assert [m.frame for m in danger.evidence] == [0, 2, 4]


@pytest.mark.parametrize("count,qualifies", [(4, False), (5, True)])
def test_danger_requires_five_events_and_three_in_a_channel(count, qualifies):
    b = bundle()
    shots = b.events.iloc[10:10 + count]
    report = analyze_match(replace(b, events=shots), "home", [bundle("training")])
    assert any(f.category == "danger" for f in report.findings) is qualifies


def test_away_channel_boundaries_apply_after_rotation():
    b = bundle()
    events = b.events.iloc[:5].assign(team="away", player_id="a", y0=[34., 68 / 6, 0., -68 / 6, -34.])
    report = analyze_match(replace(b, events=events), "away", [bundle("training")])
    assert report.passes.channel.tolist() == ["right", "centre", "centre", "left", "left"]


def test_finding_wording_keeps_hypotheses_and_loss_locations_distinct():
    report = analyze_match(bundle(), "home", [bundle("training")])
    for finding in report.findings:
        assert finding.suggestion.startswith(("Investigate", "Review"))
        assert finding.tradeoff
        text = " ".join((finding.observation, finding.suggestion, finding.tradeoff)).lower()
        assert not any(claim in text for claim in ("caused", "guarantee", "will win", "intended target", "shot probability", "pass-success probability"))
    progression = next(f for f in report.findings if f.category == "progression")
    assert "useful circulation" in progression.observation


def test_away_attacking_channels_are_rotated():
    report = analyze_match(bundle(), "home", [bundle("other")])
    assert "their left" in next(f for f in report.findings if f.category == "danger").title
    assert list(channel([-34, -68/6, 0, 68/6, 34])) == ["right", "centre", "centre", "left", "left"]


def test_insufficient_and_empty_events_have_no_findings():
    b = bundle()
    for rows in (b.events.iloc[:2], b.events.iloc[:0]):
        report = analyze_match(replace(b, events=rows), "home", [bundle("other")])
        assert report.findings == []


def test_training_leakage_and_invalid_team_are_rejected():
    with pytest.raises(ValueError, match="held-out"):
        analyze_match(bundle(), "home", [bundle()])
    with pytest.raises(ValueError, match="other training"):
        analyze_match(bundle(), "home", [])
    with pytest.raises(ValueError, match="team must"):
        analyze_match(bundle(), "ball", [bundle("other")])
    with pytest.raises(ValueError, match="different match"):
        analyze_match(bundle(), "home", [replace(bundle("other"), events=bundle().events)])


def test_canonical_events_and_tracking_flip_together(tmp_path):
    b = bundle()
    t = b.tracking.copy()
    t[["x", "y"]] *= -1
    e = b.events.copy()
    e[["x0", "y0", "x1", "y1"]] *= -1
    t.to_csv(tmp_path / "tracking.csv", index=False)
    e.to_csv(tmp_path / "events.csv", index=False)
    tr, ev = load_canonical(tmp_path / "tracking.csv", tmp_path / "events.csv")
    assert tr.iloc[0].x == -20 and ev.iloc[0].x0 == -20


def test_orientation_is_independent_for_multiple_matches(tmp_path):
    first, second = bundle(), bundle("flipped")
    second.tracking[["x", "y"]] *= -1
    second.events[["x0", "y0", "x1", "y1"]] *= -1
    pd.concat([first.tracking, second.tracking]).to_csv(tmp_path / "t.csv", index=False)
    pd.concat([first.events, second.events]).to_csv(tmp_path / "e.csv", index=False)
    tracking, events = load_canonical(tmp_path / "t.csv", tmp_path / "e.csv")
    assert tracking.query("team == 'home'").groupby("match_id").x.mean().eq(-32.5).all()
    assert events.query("team == 'home'").groupby("match_id").x0.mean().eq(-20).all()


def test_pressure_does_not_mix_periods_with_repeated_frames():
    b = bundle()
    t = b.tracking.iloc[:3].copy()
    t2 = t.assign(period=2, x=0.)
    p = b.events.iloc[:1].assign(y0=0.)
    result = add_pressure(p, pd.concat([t, t2]))
    assert result.pressure_m.iloc[0] == 40.


def test_kinematics_separate_matches_and_source_rates():
    b = bundle()
    t = b.tracking.query("team == 'home' and player_id == 'h'").copy()
    t["x"] = t.time_s * 2
    other = t.assign(match_id="other", x=t.x + 1000)
    out = add_kinematics(pd.concat([t, other]))
    assert out.groupby("match_id").speed.median().between(1.99, 2.01).all()
    assert infer_fps(t) == 1
    players = player_impact(pass_values(b.events, np.ones((2, 1)) * .1), b.tracking)
    assert np.allclose(players.minutes, 15.)


def test_missing_tracking_and_period_aware_snapshot():
    pytest.importorskip("plotly")
    from soccer_tactics.dashboard.charts import snapshot, tracking_window

    b = bundle()
    moment = analyze_match(b, "home", [bundle("train")]).findings[0].evidence[0]
    missing = replace(b, tracking=b.tracking.assign(period=2))
    assert tracking_window(missing, moment).empty
    window = tracking_window(b, moment)
    assert not snapshot(window, moment.time_s, b.fps).empty
    assert snapshot(window, 5000, b.fps).empty


def test_quality_reports_duplicates_and_missing_alignment():
    b = bundle()
    t = pd.concat([b.tracking, b.tracking.iloc[:1]])
    quality = inspect_quality(t, b.events)
    assert quality["duplicate_objects"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        analyze_match(replace(b, quality=quality), "home", [bundle("other")])
    assert inspect_quality(b.tracking.assign(period=2), b.events)["event_tracking_coverage"] == 0


@pytest.mark.skipif(len(GAMES) < 2, reason="requires both real Metrica games")
def test_real_scores_include_penalty_and_quality_is_usable():
    bundles = [load_metrica_bundle("data/raw/metrica", game) for game in (1, 2)]
    assert sum(bundles[0].score.values()) == 3
    assert sum(bundles[1].score.values()) == 5
    assert bundles[1].goals.penalty.sum() == 1
    assert bundles[1].events.query("event == 'shot'").success.sum() == 4
    for b in bundles:
        assert b.quality["duplicate_objects"] == 0
        assert b.quality["conflicting_clocks"] == 0
        assert b.quality["alignment_median_m"] < 1
        assert b.quality["event_tracking_coverage"] > .98
