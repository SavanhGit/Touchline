"""Analytics-backed tools for an LLM agent.

The registry is deliberately independent of an LLM SDK. Applications can pass
``specs()`` to their provider and dispatch tool calls through ``call``.
"""
from typing import Any, Callable

import numpy as np
import pandas as pd

TOOLS: dict[str, dict] = {}
STATE: dict[str, Any] = {
    "tracking": None,
    "events": None,
    "xt_grid": None,
    "impact": None,
}


def tool(name: str, description: str, properties: dict, required: list[str]):
    def wrap(fn: Callable):
        TOOLS[name] = {
            "spec": {
                "name": name,
                "description": description,
                "input_schema": {"type": "object", "properties": properties, "required": required},
            },
            "fn": fn,
        }
        return fn

    return wrap


def specs() -> list[dict]:
    """Tool definitions in the format the Anthropic Messages API expects."""
    return [t["spec"] for t in TOOLS.values()]


def call(name: str, **kwargs):
    if name not in TOOLS:
        raise KeyError(f"unknown analytics tool: {name}")
    return TOOLS[name]["fn"](**kwargs)


def set_state(
    *,
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    xt_grid: np.ndarray,
    impact: pd.DataFrame | None = None,
) -> None:
    """Load the canonical match data and fitted analytics into the tool state."""
    STATE.update(tracking=tracking, events=events, xt_grid=np.asarray(xt_grid), impact=impact)


def _require(*names: str) -> dict[str, Any]:
    missing = [name for name in names if STATE.get(name) is None]
    if missing:
        raise RuntimeError(f"agent state is missing: {', '.join(missing)}")
    return {name: STATE[name] for name in names}


def _limit(n: int) -> int:
    if n < 1 or n > 50:
        raise ValueError("n must be between 1 and 50")
    return n


@tool(
    "top_speeds",
    "Peak speeds (m/s) by player in the loaded match, as the 99.9th percentile to ignore tracking noise.",
    {"n": {"type": "integer", "description": "How many players to return"}},
    [],
)
def top_speeds(n: int = 5) -> list[dict]:
    from soccer_tactics.analytics.features import add_kinematics

    n = _limit(n)
    df = add_kinematics(_require("tracking")["tracking"])
    df = df[df["team"] != "ball"]
    out = df.groupby(["match_id", "team", "player_id"])["speed"].quantile(0.999).nlargest(n).rename("peak_speed")
    return out.reset_index().to_dict("records")


@tool(
    "rank_players",
    "Rank players by pass impact or build-up turnovers. Impact is pass-only and "
    "single-match detriment is noisy; use turnover count for a steadier ranking.",
    {
        "metric": {
            "type": "string",
            "enum": ["impact_p90", "detriment_p90", "buildup_turnovers_p90"],
        },
        "n": {"type": "integer", "description": "How many players to return"},
        "outfield_only": {"type": "boolean", "description": "Exclude positional goalkeeper flags"},
    },
    ["metric"],
)
def rank_players(
    metric: str = "impact_p90",
    n: int = 5,
    outfield_only: bool = True,
) -> list[dict]:
    n = _limit(n)
    impact = _require("impact")["impact"]
    if metric not in {"impact_p90", "detriment_p90", "buildup_turnovers_p90"}:
        raise ValueError(f"unsupported player metric: {metric}")
    out = impact if not outfield_only else impact[~impact["is_gk"]]
    return out.nlargest(n, metric).to_dict("records")


@tool(
    "turnover_zones",
    "Top build-up turnover locations in the losing team's attacking frame. "
    "Costs are xT-weighted and should explain, not replace, turnover-count rankings.",
    {
        "team": {"type": "string", "enum": ["home", "away"]},
        "n": {"type": "integer", "description": "How many zones to return"},
    },
    ["team"],
)
def turnover_zones(team: str, n: int = 5) -> list[dict]:
    from soccer_tactics.analytics.impact import pass_values, turnover_zones as grid_zones

    n = _limit(n)
    if team not in {"home", "away"}:
        raise ValueError("team must be home or away")
    state = _require("events", "xt_grid", "tracking")
    passes = pass_values(state["events"], state["xt_grid"], state["tracking"])
    grid = grid_zones(passes[passes["team"] == team])
    flat = np.argsort(grid.ravel())[::-1]
    rows = []
    for index in flat:
        if len(rows) == n or grid.ravel()[index] <= 0:
            break
        x, y = np.unravel_index(index, grid.shape)
        rows.append({"x_cell": int(x), "y_cell": int(y), "turnover_cost": float(grid[x, y])})
    return rows


@tool(
    "threat_map",
    "Return control-weighted home and away threat at the tracking moment nearest to a match time. "
    "Pitch control is time-to-intercept, not a pass-success or shot-prediction model.",
    {
        "match_id": {"type": "integer", "description": "Match identifier"},
        "time_s": {"type": "number", "description": "Match time in seconds"},
        "n": {"type": "integer", "description": "How many highest-threat cells to return"},
    },
    ["match_id", "time_s"],
)
def threat_map(match_id: int, time_s: float, n: int = 10) -> list[dict]:
    from soccer_tactics.analytics.features import add_kinematics
    from soccer_tactics.analytics.pitch_control import pitch_control, zone_threat

    n = _limit(n)
    tracking = add_kinematics(_require("tracking")["tracking"])
    match = tracking[tracking["match_id"] == match_id]
    if match.empty:
        raise ValueError(f"match not found: {match_id}")
    moment = (match[["period", "frame", "time_s"]].drop_duplicates()
                .assign(_distance=lambda x: (x["time_s"] - time_s).abs())
                .sort_values(["_distance", "period", "frame"]).iloc[0])
    frame = match[(match["frame"] == moment["frame"]) & (match["period"] == moment["period"])]
    control = pitch_control(frame)
    home, away = zone_threat(control, _require("xt_grid")["xt_grid"])
    rows = []
    for team, values in (("home", home), ("away", away)):
        for index in np.argsort(values.ravel())[::-1][:n]:
            x, y = np.unravel_index(index, values.shape)
            rows.append({
                "team": team,
                "x_cell": int(x),
                "y_cell": int(y),
                "threat": float(values[x, y]),
            })
    return rows
