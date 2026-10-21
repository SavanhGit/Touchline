"""IDSSE match context with source event origins and unsynchronized tracking."""
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_tactics.loaders import idsse
from soccer_tactics.match import MatchBundle, infer_fps, inspect_quality, validate_bundle
from soccer_tactics.schema import EVENT_COLUMNS


_SEGMENTS = {"firstHalf": 1, "secondHalf": 2}


def _metadata(source) -> tuple[dict, dict, tuple[int, int]]:
    information = source.match_information.find("MatchInformation")
    if information is None:
        raise ValueError("IDSSE match information is missing")
    general = information.find("General")
    environment = information.find("Environment")
    if general is None or environment is None:
        raise ValueError("IDSSE General or Environment metadata is missing")
    required = ("MatchId", "KickoffTime", "CompetitionName", "HomeTeamId",
                "GuestTeamId", "HomeTeamName", "GuestTeamName", "Result")
    if any(not general.get(key) for key in required):
        raise ValueError("IDSSE match metadata is incomplete")
    if general.get("MatchId") != source.match_id:
        raise ValueError("IDSSE metadata match ID differs from requested match")
    try:
        length, width = float(environment.get("PitchX")), float(environment.get("PitchY"))
        scores = tuple(int(value) for value in general.get("Result").split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("IDSSE pitch dimensions or source score are invalid") from exc
    if not np.isfinite([length, width]).all() or min(length, width) <= 0 or len(scores) != 2:
        raise ValueError("IDSSE pitch dimensions or source score are invalid")
    names = {"home": general.get("HomeTeamName"), "away": general.get("GuestTeamName")}
    metadata = {
        "match_date": pd.Timestamp(general.get("KickoffTime")).date().isoformat(),
        "competition": general.get("CompetitionName"),
        "source_team_ids": {"home": general.get("HomeTeamId"),
                            "away": general.get("GuestTeamId")},
        "source_team_labels": names.copy(),
        "source_pitch_dimensions_m": {"length": length, "width": width},
        "source_score": {"home": scores[0], "away": scores[1]},
    }
    return metadata, names, scores


def _observed_clocks(source) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Index raw N/T by period and require an unambiguous observed clock."""
    clocks = {}
    for segment, period in _SEGMENTS.items():
        if segment not in source.positions:
            continue
        rows = source.position_clocks.get(segment)
        if rows is None or rows.empty:
            raise ValueError(f"missing observed IDSSE clock for {segment}")
        frames = rows.frame.to_numpy(dtype=int)
        timestamps = pd.to_datetime(rows.timestamp, utc=True, errors="raise")
        nanoseconds = timestamps.astype("int64").to_numpy()
        if len(np.unique(frames)) != len(frames) or len(np.unique(nanoseconds)) != len(nanoseconds):
            raise ValueError(f"duplicate observed IDSSE clock keys for {segment}")
        if (np.diff(frames) <= 0).any() or (np.diff(nanoseconds) <= 0).any():
            raise ValueError(f"non-increasing observed IDSSE clock for {segment}")
        clocks[period] = frames, nanoseconds
    return clocks


def _source_missing_positions(source) -> int:
    """Count absent coordinate pairs in every object column at observed frames.

    This includes an object missing in every observed frame; tracking omits all
    of its rows, but its source samples still count as missing.
    """
    missing = 0
    for segment in _SEGMENTS:
        if segment not in source.positions:
            continue
        frames = source.position_clocks[segment].frame.to_numpy(dtype=int)
        offsets = frames - int(frames[0])
        for team in ("Home", "Away", "Ball"):
            xy = source.positions[segment][team]
            x, y = np.asarray(xy.x)[offsets], np.asarray(xy.y)[offsets]
            missing += int((~(np.isfinite(x) & np.isfinite(y))).sum())
    return missing


def load_idsse_bundle(root: str | Path, match_id: str) -> MatchBundle:
    """Load a match with countable source origins outside canonical endpoint analytics."""
    source = idsse._read_idsse_source(root, match_id)
    metadata, names, _ = _metadata(source)
    raw_tracking = idsse._tracking(source)
    fps = infer_fps(raw_tracking)
    clocks = _observed_clocks(source)
    source_fps = {
        period: round(float(np.median(np.diff(frames) * 1_000_000_000 / np.diff(times))), 8)
        for period, (frames, times) in clocks.items() if len(times) > 1
    }
    tracking = raw_tracking
    selection = idsse.select_idsse_events(source.events, match_id)
    # The public tuple loader retains selected source events with unavailable
    # endpoints. A bundle keeps them in a separate origin-only table so no
    # canonical endpoint consumer can mistake them for modeled passes.
    events = pd.DataFrame(columns=EVENT_COLUMNS)
    goals = selection.goals.copy()
    length = metadata["source_pitch_dimensions_m"]["length"]
    width = metadata["source_pitch_dimensions_m"]["width"]
    origin_rows = []
    for source_order, source_row in enumerate(selection.source_events.itertuples(index=False)):
        if pd.isna(source_row.canonical_index):
            continue
        selected = selection.events.iloc[int(source_row.canonical_index)]
        origin_rows.append({
            "origin_id": f"IDSSE:{match_id}:{source_order:06d}",
            "match_id": match_id, "period": int(source_row.period),
            "team": source_row.team, "event": selected.event,
            "success": selected.success, "player_id": selected.player_id,
            "source_code": source_row.eID,
            "source_timestamp": pd.Timestamp(source_row.timestamp),
            "gameclock_s": float(source_row.gameclock),
            "source_x_m": float(source_row.at_x), "source_y_m": float(source_row.at_y),
            "x_m": float(source_row.at_x) - length / 2,
            "y_m": float(source_row.at_y) - width / 2,
        })
    event_origins = pd.DataFrame(origin_rows)
    metadata["transforms"] = {
        "tracking": "source centered metres; no attack-direction rotation",
        "event_origins": "subtract half source pitch length/width in metres; no rotation",
    }
    metadata["event_capabilities"] = {
        "origin_coordinates": True, "pass_endpoints": False,
        "attack_direction": False, "tracking_sync": False,
    }
    metadata["clock_provenance"] = {
        "source_timestamp": "source event timestamp; no validated link to tracking",
        "gameclock_s": "source period-relative event gameclock",
        "tracking_time_s": "seconds since the first observed tracking timestamp in each period",
    }
    quality = inspect_quality(tracking, events)
    for key in ("event_tracking_coverage", "alignment_median_m", "alignment_p90_m",
                "shots_in_attacking_half"):
        quality.pop(key, None)
    quality.update({
        "source_events_total": selection.total,
        "source_event_codes": selection.source_events.get(
            "eID", pd.Series(dtype="string")).value_counts().to_dict(),
        "event_exclusion_reasons": selection.exclusions.reason.value_counts().to_dict(),
        "event_origin_count": len(event_origins),
        "source_fps_by_period": source_fps,
        "source_missing_positions": _source_missing_positions(source),
        "source_score": metadata["source_score"],
        "scored_goals": len(goals),
    })
    bundle = MatchBundle(match_id, tracking, events, goals, "IDSSE/DFL", fps,
                         score_known=True, team_names=names, quality=quality,
                         metadata=metadata, event_origins=event_origins)
    validate_bundle(bundle)
    if bundle.score != metadata["source_score"]:
        raise ValueError(f"IDSSE scored goals {bundle.score} differ from source score {metadata['source_score']}")
    return bundle
