import pandas as pd
import pytest

from soccer_tactics.schema import COLUMNS, orient, validate


def _frame(home_x, away_x, period=1):
    rows = [
        (1, period, 1, 0.04, "home", "h1", home_x, 3.0),
        (1, period, 1, 0.04, "away", "a1", away_x, -3.0),
        (1, period, 1, 0.04, "ball", "ball", 0.0, 0.0),
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_orient_leaves_correct_period_alone():
    df = orient(_frame(-10.0, 10.0))
    assert df.loc[df.team == "home", "x"].iloc[0] == -10.0


def test_orient_flips_period_when_home_is_on_the_right():
    df = orient(_frame(10.0, -10.0, period=2))
    home = df[df.team == "home"].iloc[0]
    assert (home.x, home.y) == (-10.0, -3.0)


def test_validate_rejects_unknown_team():
    df = _frame(0.0, 1.0)
    df.loc[0, "team"] = "ref"
    with pytest.raises(ValueError):
        validate(df)
