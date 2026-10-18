"""Tool registry for the LLM agent. Each tool wraps one analytics function and declares
a JSON schema so the model can call it."""
from typing import Callable

import pandas as pd

TOOLS: dict[str, dict] = {}
STATE: dict[str, pd.DataFrame | None] = {"tracking": None}  # set to a canonical tracking frame


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
    return TOOLS[name]["fn"](**kwargs)


@tool(
    "top_speeds",
    "Peak speeds (m/s) by player in the loaded match, as the 99.9th percentile to ignore tracking noise.",
    {"n": {"type": "integer", "description": "How many players to return"}},
    [],
)
def top_speeds(n: int = 5) -> list[dict]:
    from soccer_tactics.analytics.features import add_kinematics

    df = add_kinematics(STATE["tracking"])
    df = df[df["team"] != "ball"]
    out = df.groupby(["team", "player_id"])["speed"].quantile(0.999).nlargest(n).rename("peak_speed")
    return out.reset_index().to_dict("records")
