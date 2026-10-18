import pytest

from conftest import GAMES
from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.schema import COLUMNS, PITCH_LENGTH, PITCH_WIDTH

pytestmark = pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")


@pytest.fixture(params=GAMES)
def tracking(request, games):
    return games[request.param][0]


def test_schema(tracking):
    assert list(tracking.columns) == COLUMNS
    assert set(tracking["team"]) == {"home", "away", "ball"}


def test_at_most_eleven_players_per_team_per_frame(tracking):
    counts = tracking[tracking["team"] != "ball"].groupby(["frame", "team"]).size()
    assert counts.max() <= 11


def test_player_ids_have_no_spaces(tracking):
    assert not tracking["player_id"].str.contains(" ").any()


def test_positions_stay_near_the_pitch(tracking):
    players = tracking[tracking["team"] != "ball"]
    inside = (players["x"].abs() < PITCH_LENGTH / 2 + 3) & (players["y"].abs() < PITCH_WIDTH / 2 + 3)
    assert inside.mean() > 0.999  # a few frames have players retrieving the ball behind the line
    assert players["x"].abs().max() < PITCH_LENGTH / 2 + 10  # catches scale or origin mistakes
    assert players["y"].abs().max() < PITCH_WIDTH / 2 + 10


def test_kinematics_mask_only_a_tiny_fraction(tracking):
    players = add_kinematics(tracking).query("team != 'ball'")
    assert players["speed"].isna().mean() < 0.005
    assert players["speed"].max() <= 12


def test_home_attacks_positive_x_in_every_period(tracking):
    players = tracking[tracking["team"] != "ball"]
    for _, g in players.groupby("period"):
        mean_x = g.groupby("team")["x"].mean()
        assert mean_x["home"] < mean_x["away"]
