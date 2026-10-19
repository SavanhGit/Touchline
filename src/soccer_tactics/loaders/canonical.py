"""Load already-normalized tracking and event files.

Dataset-specific loaders should convert their source format to the canonical
schemas first. This loader provides a small, reproducible interchange path for
new sources without coupling the analytics package to their file layouts.
"""
from pathlib import Path

import pandas as pd

from soccer_tactics.schema import orient, validate, validate_events


def load_canonical(
    tracking_path: str | Path,
    events_path: str | Path | None = None,
    *,
    orient_tracking: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load parquet or CSV files containing the canonical schemas."""
    read = lambda path: pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    tracking = validate(read(tracking_path))
    if orient_tracking:
        tracking = orient(tracking)
    events = None
    if events_path is not None:
        events = validate_events(read(events_path))
        if orient_tracking:
            from soccer_tactics.schema import attack_flips, flip_periods

            events = flip_periods(events, attack_flips(tracking), ["x0", "y0", "x1", "y1"])
    return tracking, events
