"""
Per-ZETA summary scatters: pairwise relationships among
  - mean number of starving agents
  - mean number of dominant hypergraph edges
  - mean Gini coefficient of rho

Each metric is averaged over the second half of each simulation
(steady-state regime). Each point is one ZETA value.
"""

import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


ZETA_VALUES = [round(0.5 + 0.1 * i, 2) for i in range(16)]  # 0.5..2.0
RHO_THR = 0.05
MU_THR = 0.05
TAIL_FRAC = 0.5  # average over the last 50% of snapshots


def find_sims(zeta):
    """Return all seed runs for this zeta (e.g. ..._Z0.5_s0.jsonl, _s1, ...)."""
    pat = f"sim_N*_T*_C*_Z{zeta}_s*.jsonl"
    matches = sorted(glob.glob(pat))
    # Restrict to the C=1.0 sweep when present.
    c1 = [m for m in matches if re.search(r"_C1\.0_", m)]
    matches = c1 or matches
    if not matches:
        raise FileNotFoundError(f"no sim file matching {pat}")
    return matches


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    s = x.sum()
    if s <= 0 or x.size == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1) / n)


def load_snapshots(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summary_for_run(path):
    m = re.search(r"_C([0-9.]+)_Z", path)
    C_param = float(m.group(1)) if m else 1.0
    snaps = load_snapshots(path)
    N = len(snaps[0]["mu"])

    groups = sorted(
        {g for a in snaps[0]["mu"] for g in a},
        key=lambda g: (len(g.split(",")), g),
    )
    group_members = {g: [int(m) for m in g.split(",")] for g in groups}
    group_size = {g: len(group_members[g]) for g in groups}

    agent_keys = [list(snaps[0]["mu"][n].keys()) for n in range(N)]
    agent_inv_size = [
        np.array([1.0 / group_size[k] for k in agent_keys[n]])
        for n in range(N)
    ]

    start = int(len(snaps) * (1 - TAIL_FRAC))
    starving, edge_sizes, ginis = [], [], []
    for pkt in snaps[start:]:
        rho = pkt["rho"]
        mu = pkt["mu"]
        ginis.append(gini(np.fromiter(rho.values(), dtype=float)))
        sizes = []
        for g in groups:
            members = group_members[g]
            avg = float(np.mean([mu[mi][g] for mi in members]))
            if rho[g] > RHO_THR or avg > MU_THR:
                sizes.append(group_size[g])
        edge_sizes.append(float(np.mean(sizes)) if sizes else 0.0)
        s = 0
        for n in range(N):
            r_n = np.array([rho[k] for k in agent_keys[n]])
            if (r_n * agent_inv_size[n]).sum() < C_param:
                s += 1
        starving.append(s)

    return (float(np.mean(starving)),
            float(np.mean(edge_sizes)),
            float(np.mean(ginis)))


# Per-zeta: collect (starv, esize, gini) for every seed run, plus per-zeta
# mean and std across seeds.
runs_by_z = {z: [] for z in ZETA_VALUES}
for z in tqdm(ZETA_VALUES, desc="zeta"):
    for path in find_sims(z):
        runs_by_z[z].append(summary_for_run(path))

starv_mean = np.array([np.mean([r[0] for r in runs_by_z[z]]) for z in ZETA_VALUES])
starv_std  = np.array([np.std ([r[0] for r in runs_by_z[z]]) for z in ZETA_VALUES])
esize_mean = np.array([np.mean([r[1] for r in runs_by_z[z]]) for z in ZETA_VALUES])
esize_std  = np.array([np.std ([r[1] for r in runs_by_z[z]]) for z in ZETA_VALUES])
gini_mean  = np.array([np.mean([r[2] for r in runs_by_z[z]]) for z in ZETA_VALUES])
gini_std   = np.array([np.std ([r[2] for r in runs_by_z[z]]) for z in ZETA_VALUES])

for i, z in enumerate(ZETA_VALUES):
    n_runs = len(runs_by_z[z])
    print(f"ZETA={z}  ({n_runs} runs):  "
          f"starv={starv_mean[i]:.2f}±{starv_std[i]:.2f}  "
          f"esize={esize_mean[i]:.2f}±{esize_std[i]:.2f}  "
          f"gini={gini_mean[i]:.4f}±{gini_std[i]:.4f}")


fig, axes = plt.subplots(3, 1, figsize=(7, 15))
zetas = np.array(ZETA_VALUES)
norm = plt.Normalize(vmin=zetas.min(), vmax=zetas.max())
cmap = plt.get_cmap("viridis")
colors = cmap(norm(zetas))


def scatter_panel(ax, x_mean, x_std, y_mean, y_std, xlabel, ylabel):
    ax.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std,
                fmt="none", ecolor="gray", alpha=0.6, capsize=3, zorder=2)
    sc = ax.scatter(x_mean, y_mean, c=zetas, cmap=cmap, norm=norm,
                    s=110, edgecolor="black", linewidth=0.6, zorder=3)
    for xi, yi, z in zip(x_mean, y_mean, zetas):
        ax.annotate(f"{z}", (xi, yi), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return sc


sc = scatter_panel(axes[0], starv_mean, starv_std, esize_mean, esize_std,
                   "mean # starving agents",
                   "mean dominant hyperedge size")
axes[0].set_title("starving vs hyperedge size")

scatter_panel(axes[1], starv_mean, starv_std, gini_mean, gini_std,
              "mean # starving agents",
              r"mean Gini of $\rho$")
axes[1].set_title("starving vs Gini")

scatter_panel(axes[2], esize_mean, esize_std, gini_mean, gini_std,
              "mean dominant hyperedge size",
              r"mean Gini of $\rho$")
axes[2].set_title("hyperedge size vs Gini")

fig.suptitle(
    fr"Steady-state summary across $\zeta$ sweep" + "\n" +
    fr"(C=1.0, last {int(TAIL_FRAC*100)}% of each run, "
    fr"means ± std across seeds)",
    fontsize=12,
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "sweep_summary.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"saved {out}")
