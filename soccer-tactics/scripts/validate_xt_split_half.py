import sys

import numpy as np
import pandas as pd

from soccer_tactics.analytics.xt import fit_xt, smooth_grid
from soccer_tactics.loaders import load_metrica_game

root = sys.argv[1]
e1, e2 = (load_metrica_game(root, g)[1] for g in (1, 2))
raw1, raw2 = fit_xt(e1), fit_xt(e2)


def rank_corr(a, b):
    return pd.Series(a.ravel()).corr(pd.Series(b.ravel()), method="spearman")


def pearson(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


print("split-half agreement of xT surfaces fitted on game 1 vs game 2 (192 cells)")
print(f"{'smoothing passes':>18} {'spearman':>9} {'pearson':>9} {'min xT':>8} {'max xT':>8}")
for p in (0, 1, 2, 3):
    a, b = (smooth_grid(raw1, p), smooth_grid(raw2, p)) if p else (raw1, raw2)
    print(f"{p:>18} {rank_corr(a, b):9.3f} {pearson(a, b):9.3f} {min(a.min(), b.min()):8.4f} {max(a.max(), b.max()):8.4f}")

# permutation baseline: how high does agreement get by chance for unrelated surfaces?
rng = np.random.default_rng(0)
perm = [rank_corr(raw1, rng.permutation(raw2.ravel()).reshape(raw2.shape)) for _ in range(200)]
print("\nshuffled-cells baseline spearman: mean %.3f, 95th pct %.3f" % (np.mean(perm), np.percentile(perm, 95)))

both = pd.concat([e1, e2], ignore_index=True)
full = fit_xt(both)
np.set_printoptions(linewidth=200, suppress=True)
print("\nfull-data final-third rows (x cells 12-15), raw x1000:")
print((full[12:] * 1000).round(0).astype(int))
for p in (1, 2):
    print(f"\nsame, smoothed {p} pass(es):")
    print((smooth_grid(full, p)[12:] * 1000).round(0).astype(int))
print("\nzero cells raw:", int((full == 0).sum()), "| after 1 pass:", int((smooth_grid(full, 1) == 0).sum()))
