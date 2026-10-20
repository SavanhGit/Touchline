"""Player impact from passes: value created, and damage done to the build-up.

Both scores are in xT units (probability of a goal), so they add up across passes and players:

  impact     xT added by completed passes. Only forward value counts: a safe back-pass is neutral.
  detriment  cost of failed passes that start in the team's own half (the build-up). A failed
             pass costs the possession value thrown away plus the opponent's xT at the point
             where they won the ball, so a loss near your own goal costs far more than one
             near theirs.

Scope: only pass events are loaded, so tackles, interceptions and other defensive work are not
credited. Pass-level pressure (distance to the nearest opponent) is recorded as context rather
than folded into the scores, because it moves completion by about 16 points (75% with an
opponent inside 2 m, 91% at 8-12 m on the Metrica games).

Goalkeepers are flagged (is_gk), not excluded: their long kicks, intercepted far from any
pressure, top the detriment ranking, so rank outfield players separately.

Pass the xT grid in explicitly. A grid fitted on the same matches you score has seen those
passes; for evaluation, fit it on other matches.
"""
import numpy as np
import pandas as pd

from soccer_tactics.analytics.xt import NX, NY, add_xt, attacking, cell, xt_at
from soccer_tactics.schema import FPS

BUILDUP_X = 0.0  # a pass starting in the team's own half (x < 0 when attacking +x) is build-up
KEYS = ["match_id", "team", "player_id"]


def add_pressure(passes: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    """Add pressure_m: distance from the pass origin to the nearest opponent at the pass frame."""
    p = passes.assign(_row=np.arange(len(passes)))
    tp = tracking[(tracking["team"] != "ball") & tracking["frame"].isin(p["frame"].unique())]
    tp = tp[["match_id", "period", "frame", "team", "x", "y"]].rename(columns={"team": "opp", "x": "px", "y": "py"})
    m = p[["_row", "match_id", "period", "frame", "team", "x0", "y0"]].merge(tp, on=["match_id", "period", "frame"])
    m = m[m["opp"] != m["team"]]
    near = np.hypot(m["x0"] - m["px"], m["y0"] - m["py"]).groupby(m["_row"]).min()
    pressure = np.full(len(p), np.nan)
    pressure[near.index.to_numpy()] = near.to_numpy()
    return passes.assign(pressure_m=pressure)


def pass_values(
    events: pd.DataFrame,
    grid: np.ndarray,
    tracking: pd.DataFrame | None = None,
    buildup_x: float = BUILDUP_X,
) -> pd.DataFrame:
    """Pass rows with xt0, xt1, xt_gain (see add_xt) plus:

    xt_added       forward value of a completed pass, max(xt_gain, 0); 0 for failed passes
    turnover_cost  failed pass: xt0 + the opponent's xT where they won the ball; 0 if completed
    buildup        True if the pass starts in the team's own half
    pressure_m     nearest-opponent distance at the pass frame (NaN without `tracking`)
    """
    p = add_xt(events[events["event"] == "pass"], grid)
    a = attacking(p)
    ok = p["success"].to_numpy(bool)
    opp_xt = xt_at(grid, -a["x1"], -a["y1"])  # the opponent attacks the other way: mirror the point
    p["xt_added"] = np.where(ok, np.maximum(p["xt_gain"], 0.0), 0.0)
    p["turnover_cost"] = np.where(ok, 0.0, p["xt0"] + opp_xt)
    p["buildup"] = (a["x0"] < buildup_x).to_numpy()
    if tracking is None:
        p["pressure_m"] = np.nan
        return p
    return add_pressure(p, tracking)


def avg_position(tracking: pd.DataFrame) -> pd.DataFrame:
    """Mean x per player in his team's attacking frame (about -52 is his own goal line), plus
    is_gk: the deepest player of each team in each match, which is the goalkeeper."""
    pl = tracking[tracking["team"] != "ball"]
    ax = pl["x"].where(pl["team"] == "home", -pl["x"])
    out = ax.groupby([pl[k] for k in KEYS]).mean().rename("avg_x").reset_index()
    out["is_gk"] = out["avg_x"] == out.groupby(["match_id", "team"])["avg_x"].transform("min")
    return out


def player_impact(passes: pd.DataFrame, tracking: pd.DataFrame, min_minutes: float = 15.0, *, fps: float | None = None) -> pd.DataFrame:
    """One row per player and match, from `pass_values` output.

    impact_p90 and detriment_p90 are xT added and build-up cost per 90 minutes on the pitch;
    minutes come from how many tracking frames the player appears in. Players under
    `min_minutes` are dropped, since per-90 rates of a cameo are noise. avg_pressure_m is the
    mean nearest-opponent distance over the player's passes (smaller = more pressure faced).
    avg_x and is_gk come from `avg_position`."""
    from soccer_tactics.match import infer_fps

    if fps is not None:
        try:
            valid_fps = bool(np.isfinite(fps) and fps > 0)
        except (TypeError, ValueError):
            valid_fps = False
        if not valid_fps:
            raise ValueError("fps must be positive and finite")
    counts_by_player = tracking[tracking["team"] != "ball"].groupby(KEYS).size()
    rates = {mid: fps if fps is not None else infer_fps(part) for mid, part in tracking.groupby("match_id")}
    minutes = counts_by_player.div(counts_by_player.index.get_level_values("match_id").map(rates).to_numpy() * 60).rename("minutes")
    p = passes.assign(
        buildup_turnover=passes["buildup"] & ~passes["success"],
        buildup_cost=passes["turnover_cost"].where(passes["buildup"], 0.0),
    )
    agg = p.groupby(KEYS).agg(
        passes=("success", "size"),
        completed=("success", "sum"),
        buildup_passes=("buildup", "sum"),
        buildup_turnovers=("buildup_turnover", "sum"),
        xt_added=("xt_added", "sum"),
        buildup_cost=("buildup_cost", "sum"),
        avg_pressure_m=("pressure_m", "mean"),
    )
    out = minutes.to_frame().join(avg_position(tracking).set_index(KEYS)).join(agg, how="left")
    counts = ["passes", "completed", "buildup_passes", "buildup_turnovers"]
    out[counts] = out[counts].fillna(0).astype(int)
    out[["xt_added", "buildup_cost"]] = out[["xt_added", "buildup_cost"]].fillna(0.0)
    out = out[out["minutes"] >= min_minutes].copy()
    out["impact_p90"] = out["xt_added"] / out["minutes"] * 90
    out["detriment_p90"] = out["buildup_cost"] / out["minutes"] * 90
    # the plain count is steadier than the xT-weighted cost (a few costly losses dominate the sum)
    out["buildup_turnovers_p90"] = out["buildup_turnovers"] / out["minutes"] * 90
    return out.reset_index()


def action_values(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Value defensive actions by the opponent threat at the action location.

    This is deliberately separate from pass impact: it credits recoveries,
    tackles, and interceptions without pretending the sparse Metrica sample
    supports a calibrated defensive model.
    """
    actions = events[events["event"].isin(["tackle", "interception", "recovery"])].copy()
    if actions.empty:
        actions["defensive_value"] = pd.Series(dtype=float)
        return actions
    a = attacking(actions)
    actions["defensive_value"] = xt_at(grid, -a["x0"], -a["y0"])
    return actions


def player_impact_with_actions(
    passes: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    grid: np.ndarray,
    min_minutes: float = 15.0,
) -> pd.DataFrame:
    """Extend :func:`player_impact` with shot and defensive-action totals.

    Existing pass-only columns remain unchanged. New columns are raw event
    totals and per-90 rates, intended for multi-match aggregation.
    """
    out = player_impact(passes, tracking, min_minutes=min_minutes).set_index(KEYS)
    events = events[events["event"].isin(["shot", "tackle", "interception", "recovery"])]
    if events.empty:
        out["shots"], out["goals"], out["defensive_actions"] = 0, 0, 0
        out["defensive_value"], out["defensive_value_p90"] = 0.0, 0.0
        return out.reset_index()
    actions = action_values(events, grid)
    shots = events[events["event"] == "shot"].groupby(KEYS).agg(
        shots=("event", "size"), goals=("success", "sum")
    )
    defense = actions.groupby(KEYS).agg(
        defensive_actions=("event", "size"), defensive_value=("defensive_value", "sum")
    )
    out = out.join(shots, how="left").join(defense, how="left")
    out[["shots", "goals", "defensive_actions"]] = out[
        ["shots", "goals", "defensive_actions"]
    ].fillna(0).astype(int)
    out["defensive_value"] = out["defensive_value"].fillna(0.0)
    out["defensive_value_p90"] = out["defensive_value"] / out["minutes"] * 90
    return out.reset_index()


def turnover_zones(passes: pd.DataFrame, nx: int = NX, ny: int = NY, buildup_only: bool = True) -> np.ndarray:
    """Total turnover cost by where the ball was lost, shape (nx, ny), in the losing team's
    attacking frame (its own goal is at -x). Filter `passes` to one team first for its hot spots."""
    a = attacking(passes)
    lost = ~passes["success"].to_numpy(bool)
    if buildup_only:
        lost &= passes["buildup"].to_numpy(bool)
    xi, yi = cell(a["x1"].to_numpy()[lost], a["y1"].to_numpy()[lost], nx, ny)
    grid = np.zeros((nx, ny))
    np.add.at(grid, (xi, yi), passes["turnover_cost"].to_numpy()[lost])
    return grid
