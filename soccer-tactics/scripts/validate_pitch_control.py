"""Does pitch control predict pass outcomes? Control is never fitted to passes, so this is an
out-of-sample check. For each pass, take the frame at its start and ask how much the passing team
controls the point where the ball ended up (receiver for completed passes, interception point for
failed ones)."""
import sys
import time

import numpy as np
import pandas as pd

from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.pitch_control import control_at
from soccer_tactics.loaders import load_metrica_game

root = sys.argv[1]


def auc(score, label):
    """P(score of a random completed pass > score of a random failed one), ties count half."""
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = int(label.sum()), int((~label).sum())
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(score, label, n=300, seed=0):
    rng, idx = np.random.default_rng(seed), np.arange(len(score))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        vals.append(auc(score[s], label[s]))
    return np.percentile(vals, [2.5, 97.5])


rows = []
t0 = time.time()
for g in (1, 2):
    tr, ev = load_metrica_game(root, g)
    ev = ev[ev.event == "pass"]
    tr = add_kinematics(tr)
    tr = tr[(tr.team != "ball") & tr.frame.isin(ev.frame.unique())]
    frames = {f: d for f, d in tr.groupby("frame")}
    for e in ev.itertuples():
        fr = frames.get(e.frame)
        if fr is None or fr.team.nunique() < 2:
            continue
        still = fr.assign(vx=0.0, vy=0.0)
        mine = 1.0 if e.team == "home" else -1.0
        p_vel = float(control_at(fr, e.x1, e.y1))
        p_still = float(control_at(still, e.x1, e.y1))
        d = np.hypot(fr.x - e.x1, fr.y - e.y1)
        d_team, d_opp = d[fr.team == e.team].min(), d[fr.team != e.team].min()
        rows.append(dict(
            game=g, success=bool(e.success),
            ctrl_vel=p_vel if e.team == "home" else 1 - p_vel,
            ctrl_still=p_still if e.team == "home" else 1 - p_still,
            near_diff=d_opp - d_team,
            length=np.hypot(e.x1 - e.x0, e.y1 - e.y0),
        ))
df = pd.DataFrame(rows)
print(f"{len(df)} passes ({int(df.success.sum())} completed, {int((~df.success).sum())} failed) in {time.time() - t0:.1f}s")

y = df.success.to_numpy()
print(f"\n{'predictor':<44} {'AUC':>6}  95% CI (bootstrap)")
for name, col in [
    ("nearest-opponent minus nearest-teammate dist", "near_diff"),
    ("pitch control, positions only", "ctrl_still"),
    ("pitch control, with velocity extrapolation", "ctrl_vel"),
]:
    s = df[col].to_numpy()
    lo, hi = boot_ci(s, y)
    print(f"{name:<44} {auc(s, y):6.3f}  [{lo:.3f}, {hi:.3f}]")
print("\nper game (pitch control with velocity):", {g: round(auc(d.ctrl_vel.to_numpy(), d.success.to_numpy()), 3) for g, d in df.groupby("game")})

df["bin"] = pd.cut(df.ctrl_vel, [0, .2, .4, .6, .8, .95, 1.0], include_lowest=True)
t = df.groupby("bin", observed=True).success.agg(["size", "mean"]).rename(columns={"size": "passes", "mean": "completion rate"})
print("\ncompletion rate by passing team's control at the ball's end point:"); print(t.round(3).to_string())
