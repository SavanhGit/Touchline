from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.impact import add_pressure, pass_values, player_impact
from soccer_tactics.analytics.report import analyze_match, channel
from soccer_tactics.loaders import load_canonical, load_metrica_bundle
from soccer_tactics.match import MatchBundle, infer_fps, inspect_quality
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS
from conftest import GAMES


def bundle(mid="test"):
    tracking = pd.DataFrame(
        [(mid, 1, f, float(f), side, pid, x, 0.) for f in range(900)
         for side, pid, x in [("home", "h", -20.), ("home", "g", -45.), ("away", "a", 20.)]],
        columns=COLUMNS,
    )
    rows = [(mid, 1, "home", "pass", i >= 6, "h", "r", i, float(i), -20., -20., 30. if i >= 8 else -5., -20.) for i in range(10)]
    rows += [(mid, 1, "away", "shot", i == 0, "a", None, 20+i, float(20+i), -40., -20. if i < 3 else 0., -52., 0.) for i in range(5)]
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return MatchBundle(mid, tracking, events, pd.DataFrame(columns=["team", "period", "frame", "time_s", "player_id", "penalty"]), "synthetic", 1.)


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
    for finding in report.findings:
        assert 1 <= len(finding.evidence) <= 3
        assert finding.suggestion and finding.tradeoff
        for moment in finding.evidence:
            assert moment.match_id == "test"
            assert moment.event_id.startswith(f"test:{moment.period}:{moment.frame}:")
            assert ((bundle().events.frame == moment.frame) & (bundle().events.period == moment.period)).any()


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
