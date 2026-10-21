"""Complete Metrica match context alongside the existing filtered modeling loader."""
from pathlib import Path

import pandas as pd

from soccer_tactics.loaders.metrica import load_metrica_game
from soccer_tactics.match import MatchBundle, infer_fps, inspect_quality, validate_bundle


def load_metrica_bundle(root: str | Path, game: int = 1) -> MatchBundle:
    tracking, events = load_metrica_game(root, game)
    path = Path(root) / "data" / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv"
    if game == 3:
        # Recover source goal context kept separately from modeling-event filters.
        raw = _game3_goal_source(path.parent, tracking.attrs["source_team_ids"])
    else:
        raw = pd.read_csv(path)
    scored = (raw.Type == "SHOT") & raw.Subtype.fillna("").str.contains(r"(?:^|-)GOAL$")
    goals = raw.loc[scored, ["Team", "Period", "Start Frame", "Start Time [s]", "From", "Subtype"]].copy()
    goals.columns = ["team", "period", "frame", "time_s", "player_id", "subtype"]
    goals["team"] = goals.team.str.lower()
    goals["player_id"] = goals.player_id.str.replace(" ", "")
    penalty = (raw.Type.shift() == "SET PIECE") & (raw.Subtype.shift() == "PENALTY")
    goals["penalty"] = penalty.loc[scored].to_numpy()
    goals = goals.sort_values(["period", "frame"]).reset_index(drop=True)
    bundle = MatchBundle(
        f"metrica_{game}", tracking, events, goals, "Metrica Sports sample data",
        infer_fps(tracking), score_known=True,
        quality={**inspect_quality(tracking, events),
                 "defensive_exclusions": events.attrs.get("defensive_exclusions", {})},
        metadata=dict(tracking.attrs),
    )
    validate_bundle(bundle)
    return bundle



def _game3_goal_source(folder: Path, teams: dict) -> pd.DataFrame:
    """Goal and preceding set-piece context, including penalties, from the local source."""
    import json
    with (folder / "Sample_Game_3_events.json").open(encoding="utf-8") as stream:
        events = json.load(stream)["data"]
    rows = []
    for event in events:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append({"Team": teams[event["team"]["id"]], "Type": event["type"]["name"],
                     "Subtype": "-".join(item["name"] for item in subtypes),
                     "Period": event["period"], "Start Frame": event["start"]["frame"],
                     "Start Time [s]": event["start"]["time"], "From": event["from"]["id"]})
    return pd.DataFrame(rows)
