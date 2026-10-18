"""Does the zone-threat balance (control x xT) rise *before* shots? xT is fitted on the other game
so no shot is used to value its own build-up. Compared against random moments of the same match."""
import sys

import numpy as np
import pandas as pd

from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.pitch_control import pitch_control, zone_threat
from soccer_tactics.analytics.xt import fit_xt
from soccer_tactics.loaders import load_metrica_game

root = sys.argv[1]
data = {g: load_metrica_game(root, g) for g in (1, 2)}
rng = np.random.default_rng(0)


def share(frame, grid):
    h, a = zone_threat(pitch_control(frame), grid)
    return h.sum() / (h.sum() + a.sum())  # home's share of the total threat on the pitch


def pct_rank(value, pool):
    return (np.asarray(pool) < value).mean() + 0.5 * (np.asarray(pool) == value).mean()


rows = []
for g in (1, 2):
    tr, ev = data[g]
    grid = fit_xt(data[3 - g][1])  # fitted on the other game
    tr = add_kinematics(tr)
    ball_x = tr[tr.team == "ball"].drop_duplicates("frame").set_index("frame")["x"]
    tr = tr[tr.team != "ball"]
    by_frame = {f: d for f, d in tr.groupby("frame")}
    period_of = tr.drop_duplicates("frame").set_index("frame")["period"]
    all_frames = period_of.index.to_numpy()

    pool, ball_pool = {}, {}
    for f in rng.choice(all_frames, 400, replace=False):
        fr = by_frame[f]
        if fr.team.nunique() == 2 and f in ball_x.index:
            pool[f] = share(fr, grid)
            ball_pool[f] = ball_x[f]
    pool_home = np.array(list(pool.values()))
    ball_home = np.array(list(ball_pool.values()))  # ball x seen from home (+x = toward its goal)

    for s in ev[ev.event == "shot"].itertuples():
        for lead in (0, 75, 125):  # frames before the shot: 0 s, 3 s, 5 s
            f = s.frame - lead
            if f not in by_frame or period_of.get(f) != s.period or by_frame[f].team.nunique() < 2 or f not in ball_x.index:
                continue
            h = share(by_frame[f], grid)
            mine = h if s.team == "home" else 1 - h
            base = pool_home if s.team == "home" else 1 - pool_home
            sign = 1 if s.team == "home" else -1
            rows.append(dict(
                game=g, lead_s=lead / 25, team=s.team, goal=bool(s.success), share=mine, pct=pct_rank(mine, base),
                pct_ball=pct_rank(sign * ball_x[f], sign * ball_home),
            ))

df = pd.DataFrame(rows)
print("shooting team's share of pitch-wide threat, as a percentile of random moments (0.5 = typical):")
print(f"{'seconds before shot':>20} {'shots':>6} {'predictor':>16} {'mean pct':>9} {'95% CI':>16} {'above median':>13}")
for lead, d in df.groupby("lead_s"):
    for name, col in [("threat share", "pct"), ("ball x only", "pct_ball")]:
        p = d[col].to_numpy()
        boots = [rng.choice(p, len(p)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{lead:>20.0f} {len(d):>6} {name:>16} {p.mean():9.3f} {f'[{lo:.3f}, {hi:.3f}]':>16} {(p > 0.5).mean():13.2f}")
    diff = (d.pct - d.pct_ball).to_numpy()
    b = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    print(f"{'':>20} paired difference (threat - ball): {diff.mean():+.3f}  95% CI [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]")
print("\nmean threat share at 3 s before shot:", round(df[df.lead_s == 3].share.mean(), 3), "| random-moment mean (home perspective):", round(float(np.mean(pool_home)), 3))
