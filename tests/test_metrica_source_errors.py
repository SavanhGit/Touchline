"""Malformed local source files should give recovery instructions before modeling."""
import pandas as pd
import pytest

from soccer_tactics.loaders.metrica import _load_events_raw, _read_team


@pytest.mark.parametrize("source", [
    ",,,Home,,\n",
    ",,,Home,,\n,,,1,,\n",
    ",,,Home,,\n,,,1,,\nPeriod,Frame,Time [s]\n1,1,0.04\n",
    ",,,Home,,\n,,,1,,\nPeriod,Frame,Time [s],Player1,,Ball\n",
])
def test_incomplete_tracking_header_names_file_and_recovery(tmp_path, source):
    path = tmp_path / "Sample_Game_1_RawTrackingData_Home_Team.csv"
    path.write_text(source)
    with pytest.raises(ValueError) as error:
        _read_team(path, "home")
    message = str(error.value)
    assert str(path) in message
    assert "header" in message.lower()
    assert "restore" in message.lower() or "download" in message.lower()


@pytest.mark.parametrize("missing", ["Team", "Subtype", "End Y", "Start Frame"])
def test_missing_event_columns_name_file_column_and_recovery(tmp_path, missing):
    path = tmp_path / "Sample_Game_1_RawEventsData.csv"
    columns = ["Team", "Type", "Subtype", "Period", "Start Frame", "Start Time [s]",
               "From", "To", "Start X", "Start Y", "End X", "End Y"]
    pd.DataFrame(columns=[column for column in columns if column != missing]).to_csv(path, index=False)
    with pytest.raises(ValueError) as error:
        _load_events_raw(tmp_path, 1, 105., 68.)
    message = str(error.value)
    assert str(path) in message
    assert missing in message
    assert "restore" in message.lower() or "download" in message.lower()


def test_valid_small_tracking_file_still_loads(tmp_path):
    path = tmp_path / "tracking.csv"
    path.write_text(",,,Home,,,\n,,,1,,,\nPeriod,Frame,Time [s],Player1,,Ball,\n"
                    "1,1,0.04,0.2,0.3,0.4,0.5\n")
    tracking = _read_team(path, "home")
    assert tracking.player_id.tolist() == ["Player1", "ball"]
    assert tracking.x.tolist() == [0.2, 0.4]
    assert tracking.y.tolist() == [0.3, 0.5]

@pytest.mark.parametrize("metadata", [b"<broken", b"\xff\xfe\x00"])
def test_game3_malformed_metadata_is_actionable_value_error(tmp_path, metadata):
    pytest.importorskip("kloppy")
    from soccer_tactics.loaders.metrica import load_metrica_game
    folder = tmp_path / "data" / "Sample_Game_3"
    folder.mkdir(parents=True)
    (folder / "Sample_Game_3_metadata.xml").write_bytes(metadata)
    (folder / "Sample_Game_3_tracking.txt").write_text("nonempty")
    (folder / "Sample_Game_3_events.json").write_text('{"data": []}')
    with pytest.raises(ValueError) as error:
        load_metrica_game(tmp_path, 3)
    message = str(error.value)
    assert "Game 3" in message
    assert str(folder) in message
    assert "metadata" in message
    assert "restore" in message.lower() or "download" in message.lower()
    assert error.value.__cause__ is not None

@pytest.mark.parametrize("failure", ["json", "decode"])
def test_game3_external_parser_errors_name_source_and_recovery(tmp_path, monkeypatch, failure):
    import json
    kloppy = pytest.importorskip("kloppy")
    from soccer_tactics.loaders.metrica import load_metrica_game
    folder = tmp_path / "data" / "Sample_Game_3"
    folder.mkdir(parents=True)
    for suffix in ("metadata.xml", "tracking.txt", "events.json"):
        (folder / f"Sample_Game_3_{suffix}").write_text("nonempty")
    error = (json.JSONDecodeError("Invalid JSON", "{", 1) if failure == "json"
             else UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"))

    def broken_source(*args, **kwargs):
        raise error

    monkeypatch.setattr(kloppy.metrica, "load_tracking_epts", broken_source)
    with pytest.raises(ValueError) as caught:
        load_metrica_game(tmp_path, 3)
    assert str(folder) in str(caught.value)
    assert "restore" in str(caught.value).lower()
    assert caught.value.__cause__ is error


def test_game3_missing_source_retains_actionable_file_error(tmp_path):
    pytest.importorskip("kloppy")
    from soccer_tactics.loaders.metrica import load_metrica_game
    with pytest.raises(FileNotFoundError, match="Complete local Metrica Game 3 source required") as caught:
        load_metrica_game(tmp_path, 3)
    assert "Sample_Game_3_metadata.xml" in str(caught.value)
