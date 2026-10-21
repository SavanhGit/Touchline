"""Observed possession and transition summaries for canonical Metrica events."""
import pandas as pd

POSSESSION_COLUMNS = ["match_id", "period", "team", "possession_id", "start_time_s", "end_time_s",
 "start_event_id", "end_event_id", "event_ids", "terminal_reason", "duration_s", "passes",
 "completed_passes", "final_third_entries", "shots", "goals"]
TRANSITION_COLUMNS = ["match_id", "period", "team", "transition_id", "start_time_s", "end_time_s",
 "source_turnover_event_id", "event_ids", "final_third_entries", "shots", "goals", "window_s"]
DEFENSE = {"tackle", "interception", "recovery"}


def _empty(columns):
    return pd.DataFrame(columns=columns)


def _metric_lookup(metrics):
    if metrics is None or metrics.empty or "event_id" not in metrics:
        return {}
    return metrics.drop_duplicates("event_id", keep="last").set_index("event_id").to_dict("index")


def _count(row, key, default=False):
    return bool(row.get(key, default))


def segment_possessions(events: pd.DataFrame, pass_metrics: pd.DataFrame) -> pd.DataFrame:
    """Group source-ordered events into observed same-team, same-period possessions."""
    if events.empty:
        return _empty(POSSESSION_COLUMNS)
    metrics = _metric_lookup(pass_metrics)
    rows, seq = [], []

    def flush(reason):
        nonlocal seq
        if not seq:
            return
        first, last = seq[0], seq[-1]
        ids = [r.event_id for r in seq]
        passes = [r for r in seq if r.event == "pass"]
        shots = [r for r in seq if r.event == "shot"]
        rows.append(dict(match_id=first.match_id, period=first.period, team=first.team,
            possession_id=len(rows), start_time_s=first.time_s, end_time_s=last.time_s,
            start_event_id=first.event_id, end_event_id=last.event_id, event_ids=ids,
            terminal_reason=reason, duration_s=max(0.0, float(last.time_s-first.time_s)),
            passes=len(passes), completed_passes=sum(bool(metrics.get(r.event_id, {}).get("success", r.success)) for r in passes),
            final_third_entries=sum(bool(metrics.get(r.event_id, {}).get("entry", False)) for r in passes),
            shots=len(shots), goals=sum(bool(r.success) for r in shots)))
        seq = []

    previous = None
    for r in events.itertuples(index=False):
        if not hasattr(r, "event_id"):
            raise ValueError("events must include stable event_id values")
        key = (r.match_id, r.period, r.team)
        if seq and key != (seq[0].match_id, seq[0].period, seq[0].team):
            flush("period_end" if (r.match_id, r.period) != (seq[0].match_id, seq[0].period) else "team_change")
        if r.event in DEFENSE and bool(r.success):
            flush("turnover")
            seq = [r]
        else:
            seq.append(r)
        if r.event == "pass" and not bool(r.success):
            flush("turnover")
        elif r.event == "shot":
            flush("goal" if bool(r.success) else "shot")
        previous = r
    flush("period_end" if previous is not None else "end_of_data")
    return pd.DataFrame(rows, columns=POSSESSION_COLUMNS)


def summarize_transitions(events: pd.DataFrame, possessions: pd.DataFrame, pass_metrics: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame:
    """Summarize observed events in up to `window_s` after qualifying turnovers."""
    if window_s < 0:
        raise ValueError("window_s must be non-negative")
    if events.empty or possessions.empty:
        return _empty(TRANSITION_COLUMNS)
    metrics = _metric_lookup(pass_metrics)
    source = events.reset_index(drop=True)
    # A transition is identified by the observed possession it starts. Several triggers
    # can point to that possession; a successful defensive action is the stronger source.
    candidates = {}
    for i, r in enumerate(source.itertuples(index=False)):
        triggers = []
        if r.event in DEFENSE and bool(r.success):
            triggers.append((i, r.event_id, True, True))
        elif r.event == "pass" and not bool(r.success):
            for j in range(i + 1, len(source)):
                q = source.iloc[j]
                if (q.match_id, q.period) != (r.match_id, r.period):
                    break
                if q.team != r.team:
                    if float(q.time_s) <= float(r.time_s) + window_s:
                        triggers.append((j, r.event_id, False, False))
                    break
        for start_i, source_id, require_following, is_defensive in triggers:
            start = source.iloc[start_i]
            matches = possessions[(possessions.match_id == start.match_id) &
                (possessions.period == start.period) & (possessions.team == start.team) &
                possessions.event_ids.map(lambda ids: start.event_id in ids)]
            if matches.empty:
                continue
            possession = matches.iloc[0]
            key = (possession.match_id, possession.period, possession.team, possession.possession_id)
            existing = candidates.get(key)
            candidate = (start_i, source_id, require_following, possession, is_defensive)
            if existing is None or (is_defensive and not existing[4]):
                candidates[key] = candidate

    rows = []
    for start_i, turnover_id, require_following, possession, _ in candidates.values():
        start = source.iloc[start_i]
        limit = float(start.time_s) + window_s
        if require_following:
            following = source[(source.index > start_i) & (source.match_id == start.match_id) &
                (source.period == start.period) & (source.time_s <= limit)]
            if following.empty:
                continue
        # Restrict to this possession before clipping by clock. Time alone can pick up
        # a later possession when distinct events share the same timestamp.
        possession_ids = set(possession.event_ids)
        later = source[(source.index >= start_i) & (source.match_id == start.match_id) &
            (source.period == start.period) & (source.event_id.isin(possession_ids)) &
            (source.time_s <= limit)]
        if later.empty:
            continue
        ids = later.event_id.tolist()
        ps = [metrics.get(eid, {}) for eid in ids]
        shots = later[later.event == "shot"]
        rows.append(dict(match_id=start.match_id, period=start.period, team=start.team,
            transition_id=len(rows), start_time_s=start.time_s, end_time_s=later.iloc[-1].time_s,
            source_turnover_event_id=turnover_id, event_ids=ids,
            final_third_entries=sum(bool(m.get("entry", False)) for m in ps), shots=len(shots),
            goals=int(shots.success.astype(bool).sum()), window_s=window_s))
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)
