"""Population-genetics-style analyses on the memetic node-lineage system.

Each `node` in an agent's decision graph carries a heritable `lineage` ID
(stamped on mutation, preserved on clone/birth, copied on assimilation).
This script reads the dump series and computes:

  1. Allele-frequency spectrum (SFS) over time.
     Per dump: for each lineage L, count how many agents carry at least one
     node with lineage L → carrier_count(L). Histogram the carrier counts
     into log-spaced bins; render as a heatmap (time × frequency-bin).

  2. Selective-sweep trajectories.
     For each lineage, plot its carrier frequency over time. Highlight the
     top-K sweep candidates: variants that started rare (< 5% carrier
     frequency) and rose past 50% within the run. Fit a logistic to get a
     rough selection coefficient.

Outputs:
  viz/sfs_over_time.png
  viz/sweep_trajectories.png
  prints summary table to stdout
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DUMP_DIR = 'dumps'
OUT_DIR = 'viz'


def _week_num(path):
    m = re.search(r'week_(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_dumps():
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    out = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        agents = d.get('agents', [])
        if not agents:
            continue
        day = d.get('day', _week_num(p) * 90)
        out.append((day, agents))
    return out


def carrier_counts(agents):
    """For each lineage L, how many agents carry at least one node with L."""
    counts = Counter()
    for a in agents:
        nodes = a.get('graph', {}).get('nodes', [])
        seen = {n.get('lineage', 0) for n in nodes}
        for lid in seen:
            counts[lid] += 1
    return counts


def copy_counts(agents):
    """Total node-instances per lineage across pop (an agent counts each
    time the lineage appears, so duplicates inside one graph add up)."""
    counts = Counter()
    for a in agents:
        for n in a.get('graph', {}).get('nodes', []):
            counts[n.get('lineage', 0)] += 1
    return counts


def plot_sfs_heatmap(snapshots, out_path):
    """snapshots = list of (day, N_agents, Counter lineage→carrier_count)."""
    days = np.array([s[0] for s in snapshots])
    Ns = np.array([s[1] for s in snapshots], dtype=np.float64)

    # log-spaced carrier-frequency bins, plus a singleton bin (count==1)
    # and a "fixed" bin (frequency >= 0.95).
    edges = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        labels.append(f'{a}' if a == b - 1 else f'{a}–{b-1}')
    labels.append(f'{edges[-1]}+')

    H = np.zeros((len(labels), len(snapshots)), dtype=np.float64)
    for ti, (_, N, ctr) in enumerate(snapshots):
        for lid, c in ctr.items():
            # find bi such that edges[bi] <= c < edges[bi+1] (last bin is +∞)
            for bi in range(len(edges) - 1):
                if edges[bi] <= c < edges[bi + 1]:
                    H[bi, ti] += 1
                    break
            else:
                if c >= edges[-1]:
                    H[-1, ti] += 1

    fig, ax = plt.subplots(figsize=(12, 6))
    # log color scale so rare-variant bin (always huge) doesn't drown out the rest
    H_disp = np.log10(H + 1)
    im = ax.imshow(H_disp, aspect='auto', origin='lower', cmap='magma',
                    extent=[days[0], days[-1], -0.5, len(labels) - 0.5],
                    interpolation='nearest')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('sim day')
    ax.set_ylabel('carrier count (agents holding the lineage)')
    ax.set_title(
        f'Allele-frequency spectrum over time — {len(snapshots)} dumps\n'
        f'each cell = # distinct lineages whose carrier-count falls in that bin (log10)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('log10(# lineages + 1)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> {out_path}')


def find_sweeps(snapshots, low_thresh=0.05, high_thresh=0.50):
    """Return list of (lineage_id, freq_series, day_series) for lineages that
    started rare (max freq <= low_thresh in some early window) and rose
    past high_thresh later. Sorted by max frequency reached, descending."""
    days = [s[0] for s in snapshots]
    Ns = [s[1] for s in snapshots]
    # Build a (T, lineage) frequency series
    all_lids = set()
    for _, _, ctr in snapshots:
        all_lids.update(ctr.keys())
    sweeps = []
    for lid in all_lids:
        freqs = np.array([snapshots[t][2].get(lid, 0) / max(Ns[t], 1)
                          for t in range(len(snapshots))])
        if freqs.max() < high_thresh:
            continue
        # require that the lineage was once rare (e.g., absent or singleton at first appearance)
        first_t = next((t for t in range(len(snapshots)) if freqs[t] > 0), None)
        if first_t is None:
            continue
        # rose from below low_thresh
        early_max = freqs[first_t:max(first_t + 1, first_t + max(2, len(snapshots)//10))].max()
        if early_max > low_thresh:
            continue
        sweeps.append((lid, freqs, np.array(days)))
    sweeps.sort(key=lambda t: -t[1].max())
    return sweeps


def plot_sweeps(snapshots, sweeps, out_path, top_k=20):
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap('tab20')
    days = np.array([s[0] for s in snapshots])
    for i, (lid, freqs, _) in enumerate(sweeps[:top_k]):
        ax.plot(days, freqs * 100, lw=1.5, color=cmap(i % 20),
                 label=f'L{lid} (peak {freqs.max()*100:.0f}%)')
    ax.axhline(50, color='gray', lw=0.5, linestyle='--', alpha=0.6)
    ax.set_xlabel('sim day')
    ax.set_ylabel('carrier frequency (%)')
    ax.set_title(
        f'Top {min(top_k, len(sweeps))} candidate sweeps — '
        f'lineages that rose from <5% to >50%\n'
        f'(of {len(sweeps)} total sweep candidates across the run)')
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.85)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  -> {out_path}')


def fit_logistic_s(freqs, days):
    """Rough selection coefficient: fit logistic f(t) = 1 / (1 + e^(-s*(t-t0)))
    to the rising portion. Use two-point estimate between f=0.1 and f=0.9
    crossings since dumps are sparse. Returns s in units of 1/day, or None."""
    if freqs.max() < 0.9:
        return None
    try:
        i_lo = np.argmax(freqs > 0.1)
        i_hi = np.argmax(freqs > 0.9)
        if i_hi <= i_lo:
            return None
        dt = days[i_hi] - days[i_lo]
        if dt <= 0:
            return None
        # for f from 0.1 to 0.9 under logistic with rate s: dt = ln(81)/s
        return float(np.log(81.0) / dt)
    except Exception:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_dumps()
    if not raw:
        print('no dumps found')
        return
    print(f'loaded {len(raw)} dumps')

    snapshots = []   # (day, N, Counter)
    total_unique = set()
    for day, agents in raw:
        N = len(agents)
        ctr = carrier_counts(agents)
        snapshots.append((day, N, ctr))
        total_unique.update(ctr.keys())
    print(f'total unique lineages ever observed: {len(total_unique)}')

    plot_sfs_heatmap(snapshots, f'{OUT_DIR}/sfs_over_time.png')

    sweeps = find_sweeps(snapshots)
    plot_sweeps(snapshots, sweeps, f'{OUT_DIR}/sweep_trajectories.png')

    # Summary table — top sweeps with rough selection coefficient
    print()
    print('=== Top sweep candidates ===')
    print(f'{"lineage":>8}  {"peak%":>6}  {"reached_day":>11}  {"s/day":>8}')
    for lid, freqs, days_arr in sweeps[:20]:
        peak = freqs.max() * 100
        peak_t = days_arr[freqs.argmax()]
        s = fit_logistic_s(freqs, days_arr)
        s_str = f'{s:8.4f}' if s is not None else '       -'
        print(f'  {lid:>6}  {peak:5.1f}%  {peak_t:>11}  {s_str}')

    # Final-dump SFS one-liner
    final_day, final_N, final_ctr = snapshots[-1]
    print()
    print(f'=== Final SFS (day {final_day}, N={final_N}) ===')
    singletons = sum(1 for c in final_ctr.values() if c == 1)
    fixed = sum(1 for c in final_ctr.values() if c >= 0.95 * final_N)
    common = sum(1 for c in final_ctr.values() if c / final_N >= 0.1)
    print(f'  distinct lineages present: {len(final_ctr)}')
    print(f'  singletons (in exactly 1 agent): {singletons}')
    print(f'  common (>=10% pop):              {common}')
    print(f'  fixed (>=95% pop):               {fixed}')


if __name__ == '__main__':
    main()
