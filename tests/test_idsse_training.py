"""Event-only training uses canonical evidence without keeping tracking in memory."""
from dataclasses import replace

import pytest

from soccer_tactics.analytics.report import TrainingMatch, analyze_match
from factories import make_bundle


def training(match_id="idsse-training", source="idsse"):
    rows = make_bundle(match_id).events.copy()
    return TrainingMatch(match_id, source, rows)


def selected():
    return replace(make_bundle("idsse-selected"), source="idsse")


def test_event_only_training_produces_report_and_exact_provenance():
    report = analyze_match(selected(), "home", iter([training(), training("idsse-second")]))
    assert report.provenance["source"] == "idsse"
    assert report.provenance["training_match_ids"] == ["idsse-training", "idsse-second"]
    assert set(report.passes.match_id) == {"idsse-selected"}


def test_event_only_training_rejects_selected_match():
    with pytest.raises(ValueError, match="held-out"):
        analyze_match(selected(), "home", [training("idsse-selected")])


def test_event_only_training_rejects_other_source():
    with pytest.raises(ValueError, match="source"):
        analyze_match(selected(), "home", [training(source="metrica")])


def test_event_only_training_rejects_repeated_training_match_ids():
    with pytest.raises(ValueError, match="distinct"):
        analyze_match(selected(), "home", [training(), training()])


@pytest.mark.parametrize("events", [
    lambda rows: rows.iloc[:0],
    lambda rows: rows.drop(columns="x0"),
    lambda rows: rows.assign(match_id="wrong-match"),
    lambda rows: rows.assign(x0=float("nan")),
])
def test_event_only_training_requires_nonempty_canonical_events(events):
    other = training()
    with pytest.raises(ValueError):
        analyze_match(selected(), "home", [replace(other, events=events(other.events))])


def test_selected_bundle_still_requires_tracking_with_event_only_training():
    match = selected()
    with pytest.raises(ValueError, match="empty tracking"):
        analyze_match(replace(match, tracking=match.tracking.iloc[:0]), "home", [training()])
