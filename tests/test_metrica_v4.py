import numpy as np
import pandas as pd
import pytest
from conftest import ROOT
from soccer_tactics.loaders.metrica import _load_events_raw
from soccer_tactics.loaders.bundle import load_metrica_bundle
from soccer_tactics.loaders import catalog
from soccer_tactics.schema import validate, validate_events


def source_events(tmp_path, pairs):
    rows = [{"Team": "Home", "Type": kind, "Subtype": subtype, "Period": 1,
             "Start Frame": i + 1, "Start Time [s]": (i + 1) / 25, "From": "Player 1",
             "To": None, "Start X": .4, "Start Y": .3, "End X": np.nan, "End Y": np.nan}
            for i, (kind, subtype) in enumerate(pairs)]
    pd.DataFrame(rows).to_csv(tmp_path / "Sample_Game_1_RawEventsData.csv", index=False)
    return _load_events_raw(tmp_path, 1, 105, 68)


def test_defensive_event_subtype_mapping(tmp_path):
    events = source_events(tmp_path, [("CHALLENGE", "TACKLE-WON"), ("CHALLENGE", "TACKLE-LOST"),
                                      ("RECOVERY", "INTERCEPTION"), ("RECOVERY", "THEFT")])
    assert list(zip(events.event, events.success)) == [("tackle", True), ("tackle", False),
                                                     ("interception", True), ("recovery", True)]
    assert events.receiver_id.isna().all()
    assert events.x1.isna().all()


def test_ambiguous_defensive_events_are_excluded_and_counted(tmp_path):
    pairs = [("CHALLENGE", s) for s in ("TACKLE-FAULT-WON", "TACKLE-ADVANTAGE-LOST", "AERIAL-WON", "GROUND-LOST", "DRIBBLE-WON", "")]
    pairs += [("RECOVERY", s) for s in ("SAVED", "BLOCKED", "", "GOAL")]
    events = source_events(tmp_path, pairs)
    assert events.empty
    assert events.attrs["defensive_exclusions"] == {f"{kind}/{sub or '<blank>'}": 1 for kind, sub in pairs}


@pytest.fixture(scope="module")
def game3():
    pytest.importorskip("kloppy")
    if not (ROOT / "data/Sample_Game_3/Sample_Game_3_tracking.txt").is_file():
        pytest.skip("Local Game 3 source unavailable")
    return load_metrica_bundle(ROOT, 3)


def test_game3_epts_to_canonical_bundle(game3):
    assert game3.score == {"home": 0, "away": 2}
    validate(game3.tracking)
    validate_events(game3.events)
    assert game3.fps == 25
    assert {"tackle", "interception", "recovery"} <= set(game3.events.event)
    assert game3.quality["defensive_exclusions"]["RECOVERY/<blank>"] == 131
    assert game3.metadata["source_score"] == {"home": 0, "away": 2}


def test_game3_event_clocks_align_to_tracking(game3):
    events, tracking = game3.events, game3.tracking
    clocks = tracking[["period", "frame", "time_s"]].drop_duplicates()
    joined = events.merge(clocks, on=["period", "frame"], suffixes=("", "_tracking"))
    assert len(joined) == len(events)
    assert np.allclose(joined.time_s, joined.time_s_tracking, atol=1e-6)
    players = tracking[tracking.team != "ball"]
    passes = events[events.event == "pass"].merge(players, on=["period", "frame", "team", "player_id"])
    assert len(passes) / (events.event == "pass").sum() > .99
    assert np.hypot(passes.x0 - passes.x, passes.y0 - passes.y).quantile(.9) < 2
    shots = events[events.event == "shot"]
    assert (shots.x0.where(shots.team == "home", -shots.x0) > 0).mean() > .9


def test_game3_catalogue_includes_only_complete_sources(tmp_path, monkeypatch):
    folder = tmp_path / "data/Sample_Game_3"
    folder.mkdir(parents=True)
    paths = [folder / f"Sample_Game_3_{suffix}" for suffix in ("metadata.xml", "tracking.txt", "events.json")]
    for path in paths[:-1]:
        path.write_text("source")
    assert catalog.list_match_catalogue(tmp_path, tmp_path / "none", tmp_path / "manifest") == []
    paths[-1].write_text("source")
    monkeypatch.setattr(catalog, "_metrica_epts_available", lambda: True)
    entries = catalog.list_match_catalogue(tmp_path, tmp_path / "none", tmp_path / "manifest")
    assert [(e.match_id, e.ready) for e in entries] == [("metrica_3", True)]
    monkeypatch.setattr(catalog, "_metrica_epts_available", lambda: False)
    entry = catalog.list_match_catalogue(tmp_path, tmp_path / "none", tmp_path / "manifest")[0]
    assert not entry.ready and "metrica-epts" in entry.reason
    paths[1].write_text("")
    assert catalog.list_match_catalogue(tmp_path, tmp_path / "none", tmp_path / "manifest") == []



def test_game3_missing_dependency_is_actionable_without_fetching(tmp_path, monkeypatch):
    import sys
    from soccer_tactics.loaders.metrica import load_metrica_game
    monkeypatch.setitem(sys.modules, "kloppy", None)
    with pytest.raises(ImportError, match="metrica-epts"):
        load_metrica_game(tmp_path, 3)
    assert list(tmp_path.iterdir()) == []
