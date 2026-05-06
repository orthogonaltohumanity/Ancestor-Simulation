"""
Hypergraph evolution GIF across the full ZETA sweep, one panel per ZETA.
Picks a single seed run for each ZETA value and renders a 4x4 grid of
xgi-drawn hypergraphs (dominant edges only). Nodes are colored by
starving status (red = starving, green = fed).
"""

import io
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import xgi


ZETA_VALUES = [round(0.5 + 0.1 * i, 2) for i in range(16)]  # 0.5..2.0
SEED = 0
RHO_THR = 0.05
MU_THR = 0.05
PANEL_COLS = 4
PANEL_ROWS = int(np.ceil(len(ZETA_VALUES) / PANEL_COLS))


def find_sim(zeta, seed):
    pat = f"sim_N*_T*_C*_Z{zeta}_s{seed}.jsonl"
    matches = sorted(glob.glob(pat))
    c1 = [m for m in matches if re.search(r"_C1\.0_", m)]
    matches = c1 or matches
    if not matches:
        # Fall back to any seed for this zeta if SEED isn't available.
        pat2 = f"sim_N*_T*_C*_Z{zeta}_s*.jsonl"
        matches = sorted(glob.glob(pat2))
        c1 = [m for m in matches if re.search(r"_C1\.0_", m)]
        matches = c1 or matches
    if not matches:
        raise FileNotFoundError(f"no sim file for ZETA={zeta}")
    return matches[-1]


def load_snapshots(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


sim_paths = {z: find_sim(z, SEED) for z in ZETA_VALUES}
print("using:")
for z, p in sim_paths.items():
    print(f"  ZETA={z}: {p}")

snaps_by_z = {z: load_snapshots(p) for z, p in sim_paths.items()}
n_frames = min(len(s) for s in snaps_by_z.values())
print(f"frames per gif: {n_frames}")

first = snaps_by_z[ZETA_VALUES[0]][0]
N = len(first["mu"])
groups = sorted(
    {g for a in first["mu"] for g in a},
    key=lambda g: (len(g.split(",")), g),
)
group_idx_of = {g: i for i, g in enumerate(groups)}
group_members = {g: [int(m) for m in g.split(",")] for g in groups}
group_sizes = np.array([len(group_members[g]) for g in groups], dtype=float)

# Per-zeta agent group indices and inverse sizes (for starving check).
agent_keys = {z: [list(snaps_by_z[z][0]["mu"][n].keys()) for n in range(N)]
              for z in ZETA_VALUES}
agent_group_pos = {}
agent_inv_size = {}
C_by_z = {}
for z in ZETA_VALUES:
    m = re.search(r"_C([0-9.]+)_Z", sim_paths[z])
    C_by_z[z] = float(m.group(1)) if m else 1.0
    agent_group_pos[z] = [
        np.array([group_idx_of[k] for k in agent_keys[z][n]], dtype=int)
        for n in range(N)
    ]
    agent_inv_size[z] = [1.0 / group_sizes[v] for v in agent_group_pos[z]]


def rho_array(pkt):
    rho = pkt["rho"]
    return np.array([rho[g] for g in groups])


def dominant_edges(pkt):
    rho = pkt["rho"]
    mu = pkt["mu"]
    edges = []
    for g in groups:
        members = group_members[g]
        avg = float(np.mean([mu[mi][g] for mi in members]))
        if rho[g] > RHO_THR or avg > MU_THR:
            edges.append(members)
    return edges


def starving_mask(pkt, z):
    r = rho_array(pkt)
    C_p = C_by_z[z]
    return np.array([
        (r[agent_group_pos[z][n]] * agent_inv_size[z][n]).sum() < C_p
        for n in range(N)
    ])


angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos = {n: (np.cos(a), np.sin(a)) for n, a in enumerate(angles)}


def render_frame(i):
    fig, axes = plt.subplots(PANEL_ROWS, PANEL_COLS,
                             figsize=(2.7 * PANEL_COLS, 2.8 * PANEL_ROWS))
    flat = np.array(axes).flatten()
    for ax in flat[len(ZETA_VALUES):]:
        ax.axis("off")
    t_now = snaps_by_z[ZETA_VALUES[0]][i]["t"]
    for z, ax in zip(ZETA_VALUES, flat[:len(ZETA_VALUES)]):
        pkt = snaps_by_z[z][i]
        edges = dominant_edges(pkt)
        starving = starving_mask(pkt, z)
        H = xgi.Hypergraph()
        H.add_nodes_from(range(N))
        H.add_edges_from(edges)
        if H.num_edges > 0:
            xgi.draw(H, pos=pos, ax=ax, node_size=0, node_labels=False,
                     edge_fc="steelblue", dyad_color="steelblue")
        node_colors = ["tab:red" if starving[n] else "tab:green"
                       for n in range(N)]
        xs = [pos[n][0] for n in range(N)]
        ys = [pos[n][1] for n in range(N)]
        ax.scatter(xs, ys, s=90, c=node_colors,
                   edgecolor="black", linewidth=0.6, zorder=5)
        for n in range(N):
            ax.annotate(str(n), pos[n], xytext=(4, 4),
                        textcoords="offset points", fontsize=7, zorder=6)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        n_starv = int(starving.sum())
        ax.set_title(fr"$\zeta$={z}  |E|={H.num_edges}  "
                     f"starv={n_starv}/{N}", fontsize=9)
    fig.suptitle(fr"Dominant-group hypergraph  —  C=1.0,  seed={SEED},  "
                 fr"t={t_now}  ($\rho>${RHO_THR}, avg $\mu>${MU_THR})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=65)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").quantize(colors=128,
                                                   method=Image.MEDIANCUT)


# Subsample frames if needed; gif size scales linearly with frame count.
FRAME_STRIDE = max(1, n_frames // 280)
frame_indices = list(range(0, n_frames, FRAME_STRIDE))
print(f"rendering {len(frame_indices)} frames "
      f"(stride={FRAME_STRIDE} of {n_frames})")
frames = [render_frame(i) for i in tqdm(frame_indices, desc="rendering")]
out = "sweep_hypergraph_full.gif"
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=140, loop=0, optimize=True)
import os
size_mb = os.path.getsize(out) / 1e6
print(f"saved {out}  ({len(frames)} frames, {size_mb:.2f} MB)")
