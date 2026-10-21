"""Deterministic match explanations with source-addressable evidence."""
from dataclasses import dataclass, field
from collections.abc import Iterable

import numpy as np
import pandas as pd

from soccer_tactics.analytics.sequences import segment_possessions, summarize_transitions, DEFENSE
from soccer_tactics.analytics.impact import pass_values, player_impact
from soccer_tactics.analytics.xt import attacking, fit_xt
from soccer_tactics.match import MatchBundle, validate_bundle
from soccer_tactics.schema import validate_events

CHANNELS = ("right", "centre", "left")  # y increases toward the attacking team's left
LIMITATIONS = [
    "These are patterns that may have contributed, not a causal explanation of the result.",
    "Progression counts passes only: carries and dribbles are unavailable; possessions contain observed events only.",
    "Single-match player rankings are noisy; impact credits passes only and goalkeeper roles are inferred.",
    "Pitch coordinates use a nominal 105 x 68 m field. Pitch control ignores ball flight and acceleration.",
    "Modeling shots exclude penalties; the score and goal timeline include them.",
    "Shot counts and outcomes do not estimate shot quality; small samples limit comparisons.",
    "Only three public Metrica matches are available; descriptive patterns do not establish significance.",
    "Five-event denominators and three supporting events are display safeguards, not significance tests.",
]


@dataclass
class Evidence:
    event_id: str
    match_id: str
    period: int
    frame: int
    time_s: float
    team: str
    player_id: str
    event: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def label(self) -> str:
        return f"P{self.period} · source {self.time_s / 60:.2f} min · {self.player_id} · frame {self.frame}"


@dataclass
class Finding:
    category: str
    title: str
    observation: str
    suggestion: str
    tradeoff: str
    metrics: dict
    evidence: list[Evidence]


@dataclass
class MatchReport:
    match_id: str
    team: str
    score: dict | None
    outcome: str
    goals: pd.DataFrame
    comparisons: pd.DataFrame
    channels: pd.DataFrame
    players: pd.DataFrame
    passes: pd.DataFrame
    findings: list[Finding]
    provenance: dict
    limitations: list[str]
    possessions: pd.DataFrame = field(default_factory=pd.DataFrame)
    transitions: pd.DataFrame = field(default_factory=pd.DataFrame)
    defensive_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    shot_summaries: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class TrainingMatch:
    """Canonical events and identity needed to train xT without tracking frames."""
    match_id: str
    source: str
    events: pd.DataFrame


def channel(y) -> np.ndarray:
    return np.asarray(CHANNELS)[np.digitize(np.asarray(y, float), [-68 / 6, 68 / 6])]


def _evidence(rows: pd.DataFrame, metric: str | None = None) -> list[Evidence]:
    ordered = rows.sort_values([metric, "period", "frame"], ascending=[False, True, True]) if metric else rows.sort_values(["period", "frame"], kind="stable")
    if metric:
        selected = ordered.head(3)
    else:
        # Sample across the match rather than presenting three adjacent kickoff passes.
        positions = ((np.arange(min(3, len(ordered))) + .5) * len(ordered) / min(3, len(ordered))).astype(int) if len(ordered) else []
        selected = ordered.iloc[positions]
    return [Evidence(**{k: row[k] for k in Evidence.__dataclass_fields__}) for row in selected.to_dict("records")]


def analyze_match(match_bundle: MatchBundle, team: str,
                  training_matches: Iterable[MatchBundle | TrainingMatch]) -> MatchReport:
    if team not in {"home", "away"}:
        raise ValueError("team must be home or away")
    bundle = match_bundle
    if bundle.metadata.get("event_capabilities", {}).get("pass_endpoints") is False:
        raise ValueError("origin-only events cannot enter endpoint-based match analysis")
    validate_bundle(bundle)
    training_events = []
    training_match_ids = []
    seen_ids = {str(bundle.match_id)}
    for other in training_matches:
        if isinstance(other, MatchBundle):
            validate_bundle(other)
        elif not isinstance(other, TrainingMatch):
            raise TypeError("training matches must be MatchBundle or TrainingMatch")
        validate_events(other.events)
        if other.events.empty:
            raise ValueError("training events cannot be empty")
        if other.events.match_id.isna().any() or set(other.events.match_id.astype(str)) != {str(other.match_id)}:
            raise ValueError("training events contain a different match from the declared match")
        if str(other.match_id) in seen_ids:
            raise ValueError("held-out match cannot be used for xT training, and training match IDs must be distinct")
        if other.source != bundle.source:
            raise ValueError("xT training source must match the selected match source")
        seen_ids.add(str(other.match_id))
        training_match_ids.append(other.match_id)
        training_events.append(other.events)
    if not training_events:
        raise ValueError("at least one other training match is required")
    grid = fit_xt(pd.concat(training_events, ignore_index=True))
    events = bundle.events.reset_index(drop=True).copy()
    events["event_id"] = [f"{bundle.match_id}:{int(row.period)}:{int(row.frame)}:{i}" for i, row in events.iterrows()]
    passes = pass_values(events, grid, bundle.tracking)
    a = attacking(passes)
    passes["channel"] = channel(a.y0)
    passes["entry"] = passes.success & (a.x0 < 17.5) & (a.x1 >= 17.5)
    passes["outside_third"] = a.x0 < 17.5
    passes["buildup_loss"] = passes.buildup & ~passes.success
    players = player_impact(passes, bundle.tracking, min_minutes=15, fps=bundle.fps)
    comparisons, channels = [], []
    for side in ("home", "away"):
        p = passes[passes.team == side]
        sh = attacking(events[(events.team == side) & (events.event == "shot")])
        b = p[p.buildup]
        comparisons.append({
            "team": side, "passes": len(p), "completed": int(p.success.sum()),
            "completion_rate": float(p.success.mean()) if len(p) else None,
            "buildup_passes": len(b), "buildup_losses": int(b.buildup_loss.sum()),
            "buildup_loss_rate": float(b.buildup_loss.mean()) if len(b) else None,
            "buildup_cost": float(b.turnover_cost.sum()), "final_third_entries": int(p.entry.sum()),
            "pass_value_added": float(p.xt_added.sum()), "modeling_shots": len(sh),
        })
        for name in CHANNELS:
            pc = p[p.channel == name]
            bc = pc[pc.buildup]
            outside = pc[pc.outside_third]
            channels.append({
                "team": side, "channel": name, "passes": len(pc),
                "buildup_passes": len(bc), "buildup_losses": int(bc.buildup_loss.sum()),
                "buildup_loss_rate": float(bc.buildup_loss.mean()) if len(bc) else None,
                "turnover_cost": float(bc.turnover_cost.sum()),
                "outside_third_passes": len(outside), "entries": int(pc.entry.sum()),
                "entry_rate": float(outside.entry.mean()) if len(outside) else None,
                "pass_value_added": float(pc.xt_added.sum()),
                "shots": int((channel(sh.y0) == name).sum()),
            })
    possessions = segment_possessions(events, passes)
    transitions = summarize_transitions(events, possessions, passes)
    defensive_rows, shot_rows = [], []
    for side in ("home", "away"):
        # Count observed team frames once, irrespective of player count or clock gaps.
        observed = bundle.tracking[bundle.tracking.team == side]
        minutes = len(observed.drop_duplicates(["match_id", "period", "frame"])) / bundle.fps / 60
        for action in sorted(DEFENSE):
            rows = events[(events.team == side) & (events.event == action)]
            successful = int(rows.success.astype(bool).sum())
            defensive_rows.append(dict(team=side, action_type=action, total_actions=len(rows),
                successful_outcomes=successful, unsuccessful_outcomes=len(rows)-successful,
                observed_tracking_minutes=minutes, actions_per_90=len(rows)*90/minutes if minutes else None,
                event_ids=rows.event_id.tolist(), evidence=_evidence(rows)))
        shots = events[(events.team == side) & (events.event == "shot")].copy()
        shots["channel"] = channel(attacking(shots).y0)
        for name in CHANNELS:
            rows = shots[shots.channel == name]
            shot_rows.append(dict(team=side, channel=name, shots=len(rows),
                goals=int(rows.success.astype(bool).sum()), event_ids=rows.event_id.tolist(),
                evidence=_evidence(rows)))
    defensive_actions = pd.DataFrame(defensive_rows)
    shot_summaries = pd.DataFrame(shot_rows)
    channels = pd.DataFrame(channels)
    own = passes[passes.team == team]
    own_channels = channels[channels.team == team]
    findings = []
    eligible = own_channels[(own_channels.buildup_passes >= 5) & (own_channels.buildup_losses >= 3)]
    if not eligible.empty:
        c = eligible.sort_values(["buildup_loss_rate", "turnover_cost"], ascending=False).iloc[0]
        rows = own[(own.channel == c.channel) & own.buildup_loss]
        pressure = rows.pressure_m.median()
        pressure_text = f" Median nearest-opponent distance was {pressure:.1f} m." if pd.notna(pressure) else " Tracking pressure was unavailable."
        findings.append(Finding(
            "buildup", f"Buildup losses on the {c.channel}",
            f"{int(c.buildup_losses)} of {int(c.buildup_passes)} buildup passes in this channel were lost ({c.buildup_loss_rate:.0%}); total modeled cost {c.turnover_cost:.3f} xT." + pressure_text,
            f"Investigate offering a closer passing option on the {c.channel}, or varying the buildup route.",
            "Bringing support toward the ball can leave less space occupied further forward.",
            {"numerator": int(c.buildup_losses), "denominator": int(c.buildup_passes), "rate": float(c.buildup_loss_rate), "cost": float(c.turnover_cost)},
            _evidence(rows, "turnover_cost"),
        ))
    eligible = own_channels[own_channels.outside_third_passes >= 5].copy()
    eligible["nonentries"] = eligible.outside_third_passes - eligible.entries
    eligible = eligible[eligible.nonentries >= 3]
    if not eligible.empty:
        c = eligible.sort_values(["entry_rate", "outside_third_passes"], ascending=[True, False]).iloc[0]
        rows = own[(own.channel == c.channel) & own.outside_third & ~own.entry]
        findings.append(Finding(
            "progression", f"Limited direct entries from the {c.channel}",
            f"{int(c.entries)} of {int(c.outside_third_passes)} passes starting outside the final third crossed into it successfully ({c.entry_rate:.0%}). Other passes may be useful circulation; this does not establish poor decisions.",
            "Review the supporting moments for a forward passing option, and compare routes through the other channels.",
            "More direct progression can increase turnover exposure; low entry rate alone does not justify forcing a pass.",
            {"numerator": int(c.nonentries), "denominator": int(c.outside_third_passes), "entries": int(c.entries), "entry_rate": float(c.entry_rate)},
            _evidence(rows),
        ))
    opponent = "away" if team == "home" else "home"
    shots = events[(events.team == opponent) & (events.event == "shot")].copy()
    shots["channel"] = channel(attacking(shots).y0)
    entries = passes[(passes.team == opponent) & passes.entry]
    for label, rows in (("shots", shots), ("completed final-third entries", entries)):
        if len(rows) < 5:
            continue
        counts = rows.channel.value_counts().reindex(CHANNELS, fill_value=0)
        name = counts.idxmax()
        count = int(counts[name])
        if count < 3:
            continue
        findings.append(Finding(
            "danger", f"Opponent danger through their {name}",
            f"{count} of {len(rows)} opponent {label} came through their attacking {name} channel ({count / len(rows):.0%}). Channels are measured in the opponent's attacking direction.",
            f"Investigate earlier pressure or additional cover in the opponent's {name} attacking channel.",
            "Extra cover in one channel can free space elsewhere; these counts do not identify the responsible defender.",
            {"numerator": count, "denominator": len(rows), "event_type": label},
            _evidence(rows[rows.channel == name], "success" if label == "shots" else None),
        ))
        break
    own_transitions = transitions[transitions.team == team]
    stalled = own_transitions[(own_transitions.final_third_entries == 0) & (own_transitions.shots == 0)]
    if len(own_transitions) >= 5 and len(stalled) >= 3:
        # Cite the first observed moment of each supporting transition.
        starts = {event_ids[0] for event_ids in stalled.event_ids}
        findings.append(Finding("transition", "Transitions without a direct entry or shot",
            f"{len(stalled)} of {len(own_transitions)} observed transitions had no completed final-third entry or shot within the clipped 10-second window.",
            "Review forward support at these transition starts as a tactical hypothesis.",
            "Sending support forward can reduce cover after another turnover; these observations do not establish causality.",
            {"numerator": len(stalled), "denominator": len(own_transitions)},
            _evidence(events[events.event_id.isin(starts)])))
    actions = events[(events.team == team) & events.event.isin(DEFENSE)]
    failed = actions[~actions.success.astype(bool)]
    if len(actions) >= 5 and len(failed) >= 3:
        findings.append(Finding("defense", "Unsuccessful observed defensive actions",
            f"{len(failed)} of {len(actions)} recognized defensive actions had unsuccessful source outcomes. Other defensive work is outside this sample.",
            "Investigate closer cover around these defensive attempts as a tactical hypothesis.",
            "Closer cover can concede space elsewhere; action outcomes alone do not identify an individual error or establish causality.",
            {"numerator": len(failed), "denominator": len(actions)}, _evidence(failed)))
    order = {name: i for i, name in enumerate(("transition", "defense", "buildup", "progression", "danger"))}
    findings.sort(key=lambda f: (-f.metrics["numerator"], order[f.category]))
    findings = findings[:3]
    limitations = LIMITATIONS.copy()
    if actions.empty:
        limitations.append("No recognized defensive actions are observed for the selected team; zero counts do not mean no defensive work occurred.")
    score = bundle.score
    outcome = "unknown" if score is None else "win" if score[team] > score[opponent] else "loss" if score[team] < score[opponent] else "draw"
    return MatchReport(
        bundle.match_id, team, score, outcome, bundle.goals.copy(), pd.DataFrame(comparisons),
        channels, players[players.team == team].copy(), passes, findings,
        {"source": bundle.source, "training_match_ids": training_match_ids,
         "fps": bundle.fps, "quality": bundle.quality, "xt_grid": grid.tolist()},
        limitations, possessions, transitions, defensive_actions, shot_summaries,
        [Evidence(**{key: row[key] for key in Evidence.__dataclass_fields__})
         for row in events.to_dict("records")],
    )


def analyze_catalogued_metrica(game: int, team: str, catalogue, load_bundle) -> MatchReport:
    """Train on every other ready Metrica match, never the selected match.

    The caller owns local loading/caching; unready and other-provider entries
    cannot enter the calibration set.
    """
    selected = load_bundle(game)
    ids = sorted({entry.match_id for entry in catalogue
                  if entry.provider == "Metrica" and entry.ready
                  and entry.match_id != selected.match_id})
    training = [load_bundle(int(mid.removeprefix("metrica_"))) for mid in ids]
    return analyze_match(selected, team, training)
