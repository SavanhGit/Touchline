"""Are the player scores signal or noise? Split-half reliability: does a player's first-half
score predict his second-half score? Also checks how much of 'impact' is just pass volume, and
prints the leaders with their average position as a face-validity check.

Goalkeepers are reported separately: their long kicks top the detriment ranking, and including
them changes the reliability numbers.

xT is fitted on the *other* game for each match, so no pass values its own build-up.
Usage: python scripts/validate_player_impact.py data/raw/metrica
"""
import sys

import numpy as np
import pandas as pd

from soccer_tactics.analytics.impact import avg_position, pass_values, player_impact
from soccer_tactics.analytics.xt import fit_xt
from soccer_tactics.loaders import load_metrica_game
from soccer_tactics.schema import FPS

root = sys.argv[1]
data = {g: load_metrica_game(root, g) for g in (1, 2)}
rng = np.random.default_rng(0)
KEYS = ["match_id", "team", "player_id"]

passes, tracks, tables = [], [], []
for g, (tr, ev) in data.items():
    p = pass_values(ev, fit_xt(data[3 - g][1]), tr)  # cross-fitted
    passes.append(p)
    tracks.append(tr)
    tables.append(player_impact(p, tr, min_minutes=30))
passes, tracking, full = pd.concat(passes, ignore_index=True), pd.concat(tracks, ignore_index=True), pd.concat(tables, ignore_index=True)
print(f"{len(passes)} passes, {len(full)} player-matches with >= 30 min ({int(full.is_gk.sum())} goalkeepers)\n")


def spearman(a, b):
    return np.corrcoef(pd.Series(np.asarray(a)).rank(), pd.Series(np.asarray(b)).rank())[0, 1]


# ---- per-half metrics -------------------------------------------------------------------
players = tracking[tracking.team != "ball"]
pos = avg_position(tracking)
is_gk = pos.set_index(KEYS)["is_gk"]
minutes = (players.groupby(KEYS + ["period"]).size() / (FPS * 60)).rename("minutes")
agg = passes.assign(cost=passes.turnover_cost.where(passes.buildup, 0.0), bt=passes.buildup & ~passes.success).groupby(KEYS + ["period"]).agg(
    passes=("success", "size"), bpasses=("buildup", "sum"), xt=("xt_added", "sum"), cost=("cost", "sum"), bt=("bt", "sum")
)
h = minutes.to_frame().join(agg).fillna(0).reset_index()
h = h[h.minutes >= 20]
h["passes_p90"] = h.passes / h.minutes * 90
h["impact_p90"] = h.xt / h.minutes * 90
h["impact_per_pass"] = h.xt / h.passes.where(h.passes > 0)
h["detriment_p90"] = h.cost / h.minutes * 90
h["detriment_per_buildup_pass"] = h.cost / h.bpasses.where(h.bpasses > 0)
h["buildup_turnovers_p90"] = h.bt / h.minutes * 90  # exploratory: count instead of xT cost
h["buildup_turnover_rate"] = h.bt / h.bpasses.where(h.bpasses > 0)  # exploratory
metrics = [
    "passes_p90", "impact_p90", "impact_per_pass", "detriment_p90", "detriment_per_buildup_pass",
    "buildup_turnovers_p90", "buildup_turnover_rate",
]
wide = h.pivot(index=KEYS, columns="period", values=metrics)


def reliability(d):
    r = spearman(d[1], d[2])
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(d), len(d))
        boots.append(spearman(d[1].to_numpy()[i], d[2].to_numpy()[i]))
    perm = [spearman(d[1], rng.permutation(d[2].to_numpy())) for _ in range(2000)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r, lo, hi, (np.sum(np.abs(perm) >= abs(r)) + 1) / (len(perm) + 1)


for label, outfield in [("all players", False), ("outfield players only (goalkeepers excluded)", True)]:
    print(f"split-half reliability, {label}: period 1 vs period 2, players with >= 20 min in both halves")
    print(f"{'metric':<28} {'n':>3} {'spearman':>9} {'95% CI (bootstrap)':>20} {'perm p':>7} {'full-match est.':>16}")
    for m in metrics:
        d = wide[m].dropna()
        if outfield:
            d = d[~is_gk.reindex(d.index).to_numpy()]
        r, lo, hi, pval = reliability(d)
        print(f"{m:<28} {len(d):>3} {r:9.2f} {f'[{lo:.2f}, {hi:.2f}]':>20} {pval:7.3f} {2 * r / (1 + r):16.2f}")
    print()
print("(full-match est. = Spearman-Brown: the reliability a whole-match score should have, from the half-vs-half r)")
print("(the last two rows are exploratory alternatives to detriment, shown so the best of several is not cherry-picked)")

# ---- why is detriment noisy? ------------------------------------------------------------
lost = passes[passes.buildup & ~passes.success].turnover_cost.sort_values(ascending=False)
print(f"\nfailed build-up passes: {len(lost)} | cost per turnover: mean {lost.mean():.3f}, median {lost.median():.3f}")
print(f"top 10 turnovers = {lost.head(10).sum() / lost.sum():.0%} of all build-up cost | top decile = {lost.head(len(lost) // 10).sum() / lost.sum():.0%}")
per_player_turnovers = passes[passes.buildup & ~passes.success].groupby(KEYS).size()
print(f"build-up turnovers per player-match: median {per_player_turnovers.median():.0f}, max {per_player_turnovers.max()}")

# ---- volume dependence + face validity --------------------------------------------------
full["passes_p90"] = full.passes / full.minutes * 90
out = full[~full.is_gk]
print(f"\noutfield players: impact_p90 vs passes_p90 spearman {spearman(out.impact_p90, out.passes_p90):.2f}")
print(f"outfield players: detriment_p90 vs impact_p90 spearman {spearman(out.detriment_p90, out.impact_p90):.2f}")

cols = ["match_id", "team", "player_id", "minutes", "passes", "avg_x", "impact_p90", "buildup_turnovers", "buildup_passes", "detriment_p90", "avg_pressure_m"]
pd.set_option("display.width", 220)
fmt = lambda d: d[cols].round(3).to_string(index=False)
print("\nhighest impact, outfield (avg_x = mean position, attacking frame: -52 is own goal, +52 theirs):")
print(fmt(out.nlargest(6, "impact_p90")))
print("\nhighest build-up detriment, outfield:")
print(fmt(out.nlargest(6, "detriment_p90")))
print("\ngoalkeepers, for contrast:")
print(fmt(full[full.is_gk].sort_values("detriment_p90", ascending=False)))
print("\nbuild-up cost share: biggest single contributor per team-match")
share = full.groupby(["match_id", "team"]).apply(
    lambda d: (lambda r: f"{r.player_id}{' [GK]' if r.is_gk else ''} ({r.buildup_cost / d.buildup_cost.sum():.0%})")(d.loc[d.buildup_cost.idxmax()]),
    include_groups=False,
)
print(share.to_string())
