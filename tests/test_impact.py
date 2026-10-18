import numpy as np
import pandas as pd
import pytest

from conftest import GAMES
from soccer_tactics.analytics.impact import (
    add_pressure,
    avg_position,
    pass_values,
    player_impact,
    turnover_zones,
)
from soccer_tactics.analytics.xt import fit_xt
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS, FPS

GRID = np.array([[0.1], [0.4]])  # nx=2, ny=1: own half is worth 0.1, attacking half 0.4


def _ev(rows):
    """rows: (team, success, x0, x1, player_id[, frame])"""
    df = pd.DataFrame(
        [(r[0], r[1], r[2], 0.0, r[3], 0.0, r[4], r[5] if len(r) > 5 else 1) for r in rows],
        columns=["team", "success", "x0", "y0", "x1", "y1", "player_id", "frame"],
    )
    df["event"], df["match_id"], df["period"] = "pass", "m", 1
    df["receiver_id"], df["time_s"] = "r", 0.0
    return df[EVENT_COLUMNS]


def test_pass_values_hand_computed():
    p = pass_values(_ev([
        ("home", True, -30, 30, "a"),    # forward completion: +0.3, build-up
        ("home", False, -30, -10, "a"),  # lost in own half: 0.1 thrown away + opponent's 0.4 = 0.5
        ("home", True, 30, -30, "b"),    # backward completion: gain -0.3 clipped to 0, not build-up
        ("home", False, 30, 20, "b"),    # lost in their half: 0.4 + opponent's 0.1 = 0.5, not build-up
        ("away", False, 30, 10, "c"),    # away attacks -x, so this is its own half; mirrors case 2
    ]), GRID)
    assert np.allclose(p["xt_added"], [0.3, 0, 0, 0, 0])
    assert np.allclose(p["turnover_cost"], [0, 0.5, 0, 0.5, 0.5])
    assert list(p["buildup"]) == [True, True, False, False, True]
    assert p["pressure_m"].isna().all()  # no tracking given


def test_add_pressure_uses_nearest_opponent_at_the_pass_frame_only():
    ev = _ev([("home", True, 0, 5, "a", 7)])
    rows = [  # frame, team, x, y
        (7, "home", 1.0, 0.0), (7, "away", 3.0, 4.0), (7, "away", 10.0, 0.0), (7, "ball", 0.5, 0.0),
        (8, "away", 0.5, 0.0),  # a closer opponent, but a different frame
    ]
    tr = pd.DataFrame([("m", 1, f, f / FPS, t, "p", x, y) for f, t, x, y in rows], columns=COLUMNS)
    assert np.isclose(add_pressure(ev, tr)["pressure_m"].iloc[0], 5.0)


def test_player_impact_minutes_and_per90_rates():
    ev = _ev([("home", True, -30, 30, "a"), ("home", False, -30, -10, "a"), ("home", True, 30, 40, "b")])
    p = pass_values(ev, GRID)
    n = FPS * 60 * 30  # 30 minutes on the pitch
    rows = [("m", 1, f, f / FPS, "home", pid, 0.0, 0.0) for pid in ("a", "b") for f in range(n)]
    rows += [("m", 1, f, f / FPS, "home", "sub", 0.0, 0.0) for f in range(FPS * 60 * 5)]  # a 5-minute cameo
    out = player_impact(p, pd.DataFrame(rows, columns=COLUMNS)).set_index("player_id")
    assert set(out.index) == {"a", "b"}  # the cameo is dropped by min_minutes
    a = out.loc["a"]
    assert np.isclose(a["minutes"], 30.0)
    assert (a["passes"], a["completed"], a["buildup_passes"], a["buildup_turnovers"]) == (2, 1, 2, 1)
    assert np.isclose(a["impact_p90"], 0.3 / 30 * 90) and np.isclose(a["detriment_p90"], 0.5 / 30 * 90)
    assert np.isclose(a["buildup_turnovers_p90"], 1 / 30 * 90)
    assert out.loc["b", "detriment_p90"] == 0 and np.isclose(out.loc["b", "impact_p90"], 0)  # 0.4 -> 0.4 adds nothing


def test_avg_position_flags_the_deepest_player_of_each_team_in_its_attacking_frame():
    # home attacks +x, so its keeper stands at x=-45; away attacks -x, so its keeper stands at +45
    spec = [("home", "gk", -45.0), ("home", "d", -20.0), ("home", "m", 0.0),
            ("away", "gk2", 45.0), ("away", "d2", 20.0), ("away", "m2", 0.0), ("ball", "ball", 0.0)]
    tr = pd.DataFrame([("m", 1, 1, 0.04, t, pid, x, 0.0) for t, pid, x in spec], columns=COLUMNS)
    pos = avg_position(tr).set_index("player_id")
    assert pos.loc[["gk", "gk2"], "is_gk"].all() and not pos.loc[["d", "m", "d2", "m2"], "is_gk"].any()
    assert pos.loc["gk2", "avg_x"] == -45.0  # mirrored into the away team's attacking frame
    assert "ball" not in pos.index


def test_player_impact_carries_position_and_goalkeeper_flag():
    ev = _ev([("home", False, -40, -10, "gk"), ("home", True, -20, 30, "d")])
    n = FPS * 60 * 30
    rows = [("m", 1, f, f / FPS, "home", pid, x, 0.0) for pid, x in (("gk", -45.0), ("d", -20.0)) for f in range(n)]
    out = player_impact(pass_values(ev, GRID), pd.DataFrame(rows, columns=COLUMNS)).set_index("player_id")
    assert out.loc["gk", "is_gk"] and not out.loc["d", "is_gk"]
    assert out.loc["gk", "avg_x"] == -45.0
    assert out.loc["gk", "detriment_p90"] > 0 and out.loc["d", "detriment_p90"] == 0


@pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")
def test_real_goalkeepers_are_unambiguous(games):
    """Measured on both games: the deepest player averages -39 to -41 m and sits 24-27 m behind
    the next one, so a loose threshold is safe."""
    for tracking, _ in games.values():
        for _, d in avg_position(tracking).groupby(["match_id", "team"]):
            d = d.sort_values("avg_x")
            assert d["is_gk"].sum() == 1 and d["is_gk"].iloc[0]
            assert d["avg_x"].iloc[0] < -35 and d["avg_x"].iloc[1] - d["avg_x"].iloc[0] > 15


def test_turnover_zones_sum_cost_by_loss_point_in_the_losers_frame():
    p = pass_values(_ev([
        ("home", False, -30, -10, "a"),  # build-up loss at home x=-10 -> cell 0, cost 0.5
        ("away", False, 30, 10, "c"),    # away loss at x=10 -> away frame x=-10 -> cell 0, cost 0.5
        ("home", False, 30, 20, "b"),    # not build-up: excluded by default
    ]), GRID)
    assert np.allclose(turnover_zones(p, 2, 1), [[1.0], [0.0]])
    assert np.allclose(turnover_zones(p, 2, 1, buildup_only=False), [[1.0], [0.5]])  # home loss at x=20 -> cell 1


@pytest.mark.skipif(not GAMES, reason="Metrica sample data not downloaded")
def test_real_scores_are_well_formed(games):
    for g, (tracking, events) in games.items():
        other = [e for k, (_, e) in games.items() if k != g]
        grid = fit_xt(other[0] if other else events)
        p = pass_values(events, grid, tracking)
        assert (p["xt_added"] >= 0).all() and (p["turnover_cost"] >= 0).all()
        assert p["pressure_m"].notna().mean() > 0.99
        assert 0.3 < p["buildup"].mean() < 0.7
        out = player_impact(p, tracking, min_minutes=0)
        assert out["passes"].sum() == len(p)  # every pass is attributed to exactly one player
        assert out["minutes"].between(0, 110).all()
        assert (out["buildup_turnovers"] <= out["buildup_passes"]).all()
        assert np.isclose(out["buildup_cost"].sum(), p.loc[p["buildup"], "turnover_cost"].sum())
