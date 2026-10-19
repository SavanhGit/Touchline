"""Cross-match evaluation helpers with leave-one-match-out fitting."""
from collections.abc import Iterable

import pandas as pd

from soccer_tactics.analytics.impact import pass_values, player_impact
from soccer_tactics.analytics.xt import fit_xt


def leave_one_match_out(
    matches: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    min_minutes: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each match using an xT grid fitted only on the other matches.

    Returns concatenated pass values and player tables. This prevents a match's
    own passes from contributing to the xT surface used to score it.
    """
    data = list(matches)
    if len(data) < 2:
        raise ValueError("cross-match evaluation needs at least two matches")
    pass_tables, player_tables = [], []
    for index, (tracking, events) in enumerate(data):
        training = pd.concat(
            [other_events for i, (_, other_events) in enumerate(data) if i != index],
            ignore_index=True,
        )
        grid = fit_xt(training)
        values = pass_values(events, grid, tracking)
        table = player_impact(values, tracking, min_minutes=min_minutes)
        pass_tables.append(values)
        player_tables.append(table)
    return pd.concat(pass_tables, ignore_index=True), pd.concat(player_tables, ignore_index=True)


def summarize_cross_match(pass_values_table: pd.DataFrame, player_table: pd.DataFrame) -> dict[str, float]:
    """Return stable, source-agnostic held-out summary metrics."""
    return {
        "matches": float(player_table["match_id"].nunique()),
        "passes": float(len(pass_values_table)),
        "completed_pass_rate": float(pass_values_table["success"].mean()),
        "mean_xt_added": float(pass_values_table["xt_added"].mean()),
        "mean_turnover_cost": float(pass_values_table["turnover_cost"].mean()),
        "players": float(len(player_table)),
    }
