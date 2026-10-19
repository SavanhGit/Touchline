"""Deterministic match explanations with source-addressable evidence."""
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd

from soccer_tactics.analytics.impact import pass_values, player_impact
from soccer_tactics.analytics.xt import attacking, fit_xt
from soccer_tactics.match import MatchBundle, validate_bundle

CHANNELS = ("right", "centre", "left")  # y increases toward the attacking team's left
LIMITATIONS = [
    "These are patterns that may have contributed, not a causal explanation of the result.",
    "Progression counts passes only: carries, dribbles and complete possessions are unavailable.",
    "Single-match player rankings are noisy; impact credits passes only and goalkeeper roles are inferred.",
    "Pitch coordinates use a nominal 105 x 68 m field. Pitch control ignores ball flight and acceleration.",
    "Modeling shots exclude penalties; the score and goal timeline include them.",
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


def channel(y) -> np.ndarray:
    return np.asarray(CHANNELS)[np.digitize(np.asarray(y, float), [-68 / 6, 68 / 6])]


def _evidence(rows: pd.DataFrame, metric: str | None = None) -> list[Evidence]:
    ordered = rows.sort_values([metric, "period", "frame"], ascending=[False, True, True]) if metric else rows.sort_values(["period", "frame"])
    if metric:
        selected = ordered.head(3)
    else:
        # Sample across the match rather than presenting three adjacent kickoff passes.
        positions = ((np.arange(min(3, len(ordered))) + .5) * len(ordered) / min(3, len(ordered))).astype(int) if len(ordered) else []
        selected = ordered.iloc[positions]
    return [Evidence(**{k: row[k] for k in Evidence.__dataclass_fields__}) for row in selected.to_dict("records")]


def analyze_match(match_bundle: MatchBundle, team: str, training_matches: Iterable[MatchBundle]) -> MatchReport:
    if team not in {"home", "away"}:
        raise ValueError("team must be home or away")
    bundle = match_bundle
    validate_bundle(bundle)
    training = list(training_matches)
    if not training:
        raise ValueError("at least one other training match is required")
    for other in training:
        validate_bundle(other)
        if str(other.match_id) == str(bundle.match_id):
            raise ValueError("held-out match cannot be used for xT training")
    grid = fit_xt(pd.concat([other.events for other in training], ignore_index=True))
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
    score = bundle.score
    outcome = "unknown" if score is None else "win" if score[team] > score[opponent] else "loss" if score[team] < score[opponent] else "draw"
    return MatchReport(
        bundle.match_id, team, score, outcome, bundle.goals.copy(), pd.DataFrame(comparisons),
        channels, players[players.team == team].copy(), passes, findings,
        {"source": bundle.source, "training_match_ids": [other.match_id for other in training],
         "fps": bundle.fps, "quality": bundle.quality, "xt_grid": grid.tolist()},
        LIMITATIONS.copy(),
    )
