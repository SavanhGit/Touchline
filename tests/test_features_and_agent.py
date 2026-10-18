import pandas as pd

from soccer_tactics.agent import tools
from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.schema import COLUMNS, FPS


def _runner(speed=4.0, frames=50):
    rows = [(1, 1, f, f / FPS, "home", "h1", f / FPS * speed, 0.0) for f in range(1, frames + 1)]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_constant_velocity_gives_constant_speed():
    out = add_kinematics(_runner())
    assert out["speed"].iloc[10:-10].between(3.99, 4.01).all()


def test_teleport_glitch_is_masked_but_normal_frames_survive():
    df = _runner()
    df.loc[df["frame"] == 25, "x"] += 50.0  # one-frame teleport
    out = add_kinematics(df).set_index("frame")
    assert out.loc[18:33, "speed"].isna().all()  # glitch frames 23 and 28, plus 5 on each side
    clean = list(range(5, 17)) + list(range(35, 48))
    assert out.loc[clean, "speed"].between(3.99, 4.01).all()


def test_ball_is_not_capped_at_player_speed():
    rows = [(1, 1, f, f / FPS, "ball", "ball", f / FPS * 20.0, 0.0) for f in range(1, 51)]
    out = add_kinematics(pd.DataFrame(rows, columns=COLUMNS))
    assert out["speed"].iloc[10:-10].between(19.9, 20.1).all()


def test_tool_specs_are_api_shaped():
    assert all({"name", "description", "input_schema"} <= set(s) for s in tools.specs())


def test_top_speeds_tool_runs_on_loaded_tracking():
    tools.STATE["tracking"] = _runner(speed=6.0)
    top = tools.call("top_speeds", n=1)
    assert top[0]["player_id"] == "h1"
    assert abs(top[0]["peak_speed"] - 6.0) < 0.05
