"""Deterministic complete match examples shared by boundary and report tests."""
import numpy as np
import pandas as pd

from soccer_tactics.match import MatchBundle
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS


def make_bundle(match_id: str = "test", *, fps: float = 1.0) -> MatchBundle:
    tracking = pd.DataFrame(
        [(match_id, 1, frame, float(time_s), side, pid, x, 0.)
         for frame, time_s in enumerate(np.arange(round(900 * fps)) / fps)
         for side, pid, x in [("home", "h", -20.), ("home", "g", -45.), ("away", "a", 20.)]],
        columns=COLUMNS,
    )
    rows = [(match_id, 1, "home", "pass", i >= 6, "h", "r", round(i * fps), float(i),
             -20., -20., 30. if i >= 8 else -5., -20.) for i in range(10)]
    rows += [(match_id, 1, "away", "shot", i == 0, "a", None, round((20 + i) * fps),
              float(20 + i), -40., -20. if i < 3 else 0., -52., 0.) for i in range(5)]
    return MatchBundle(
        match_id, tracking, pd.DataFrame(rows, columns=EVENT_COLUMNS),
        pd.DataFrame(columns=["team", "period", "frame", "time_s", "player_id", "penalty"]),
        "synthetic", fps,
    )
