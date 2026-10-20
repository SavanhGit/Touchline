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
