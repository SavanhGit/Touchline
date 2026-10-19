"""Small deterministic matches for exercising the evaluation pipeline."""

import pandas as pd

from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS, FPS


def _match(match_id: str, end_x: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.DataFrame(
        [
            (match_id, 1, "home", "pass", True, "h", "r", 1, 1 / FPS, -20, 0, end_x, 0),
            (match_id, 1, "home", "shot", end_x > 0, "h", None, 2, 2 / FPS, end_x, 0, end_x, 0),
        ],
        columns=EVENT_COLUMNS,
    )
    rows = []
    for frame in range(FPS * 60 * 15):
        rows.extend(
            [
                (match_id, 1, frame, frame / FPS, "home", "h", -20.0, 0.0),
                (match_id, 1, frame, frame / FPS, "away", "a", 20.0, 0.0),
            ]
        )
    return pd.DataFrame(rows, columns=COLUMNS), events


def load_placeholder_matches() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return two canonical matches suitable for a fast cross-match smoke test."""
    return [_match("placeholder-1", 10.0), _match("placeholder-2", 20.0)]
