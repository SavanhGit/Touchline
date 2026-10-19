import numpy as np
import pandas as pd
import pytest

from soccer_tactics.analytics.cross_match import leave_one_match_out, summarize_cross_match
from soccer_tactics.schema import COLUMNS, EVENT_COLUMNS, FPS


def _match(match_id, offset):
    events = pd.DataFrame(
        [
            (match_id, 1, "home", "pass", True, "h", "r", 1, 0.04, -20, 0, offset, 0),
            (match_id, 1, "home", "shot", bool(offset > 0), "h", None, 2, 0.08, offset, 0, offset, 0),
        ],
        columns=EVENT_COLUMNS,
    )
    rows = []
    for frame in range(FPS * 60 * 15):
        rows.extend([
            (match_id, 1, frame, frame / FPS, "home", "h", -20.0, 0.0),
            (match_id, 1, frame, frame / FPS, "away", "a", 20.0, 0.0),
        ])
    return pd.DataFrame(rows, columns=COLUMNS), events


def test_leave_one_match_out_scores_only_held_out_passes():
    passes, players = leave_one_match_out([_match("m1", 10), _match("m2", 20)], min_minutes=15)
    assert set(passes["match_id"]) == {"m1", "m2"}
    assert set(players["match_id"]) == {"m1", "m2"}
    summary = summarize_cross_match(passes, players)
    assert summary["matches"] == 2
    assert summary["passes"] == 2
    assert 0 <= summary["completed_pass_rate"] <= 1


def test_cross_match_requires_two_matches():
    with pytest.raises(ValueError, match="at least two"):
        leave_one_match_out([_match("m1", 10)])
