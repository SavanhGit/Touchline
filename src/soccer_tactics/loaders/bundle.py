"""Complete Metrica match context alongside the existing filtered modeling loader."""
from pathlib import Path

import pandas as pd

from soccer_tactics.loaders.metrica import load_metrica_game
from soccer_tactics.match import MatchBundle, infer_fps, inspect_quality


def load_metrica_bundle(root: str | Path, game: int = 1) -> MatchBundle:
    tracking, events = load_metrica_game(root, game)
    path = Path(root) / "data" / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv"
    raw = pd.read_csv(path)
    scored = (raw.Type == "SHOT") & raw.Subtype.fillna("").str.contains(r"(?:^|-)GOAL$")
    goals = raw.loc[scored, ["Team", "Period", "Start Frame", "Start Time [s]", "From", "Subtype"]].copy()
    goals.columns = ["team", "period", "frame", "time_s", "player_id", "subtype"]
    goals["team"] = goals.team.str.lower()
    goals["player_id"] = goals.player_id.str.replace(" ", "")
    penalty = (raw.Type.shift() == "SET PIECE") & (raw.Subtype.shift() == "PENALTY")
    goals["penalty"] = penalty.loc[scored].to_numpy()
    goals = goals.sort_values(["period", "frame"]).reset_index(drop=True)
    return MatchBundle(
        f"metrica_{game}", tracking, events, goals, "Metrica Sports sample data",
        infer_fps(tracking), score_known=True, quality=inspect_quality(tracking, events),
    )
