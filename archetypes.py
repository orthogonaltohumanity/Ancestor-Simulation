"""Multi-subject archetypal analysis on anthrosim agent populations.

Each weekly dump is a "subject"; agents are columns of X; the 11 ranking-weight
features are the shared row axis. ArchePy finds K archetypal personality
vectors that explain agents across the entire run, plus per-agent mixing
coefficients. Each agent ends up as "0.4 archetype A + 0.5 B + 0.1 C".

Usage:
  python archetypes.py                 # K=4, all weekly dumps
  python archetypes.py --K 6           # 6 archetypes
  python archetypes.py --weeks 30 50   # only dumps in week range
  python archetypes.py --include crests   # also include 256-d crest in features
"""
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from archepy import SubjectT, multi_subject_aa_T


FEATURE_LABELS = (
    'net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
    'age', 'is_pregnant', 'is_menopausal',
    'matings', 'gifts', 'family', 'camp$',
)


def load_week(path, include_crests=False, signed=False):
    with open(path) as f:
        d = json.load(f)
    ag = d['agents']
    if not ag:
        return None
    s = [a['state'] for a in ag]
    W = np.array([x['weights'] for x in s], dtype=np.float64)
    if include_crests:
        Cv = np.array([x['crest'] for x in s], dtype=np.float64)
        feat = np.concatenate([W, Cv], axis=1)
    else:
        feat = W
    if signed:
        # Sign-split each feature into (+col, −col), both nonneg, so AA can
        # find archetypes in both directions of feature space.
        feat = np.concatenate([np.maximum(feat, 0.0),
                               np.maximum(-feat, 0.0)], axis=1)
    # ArchePy temporal convention: X has shape (V, T). V (rows) varies per
    # subject; T (cols) is the SHARED axis across subjects. Archetypes live
    # in T-space (length-T vectors).
    # → V = agents (variable), T = features (shared = 11).
    # Each agent gets a K-dim mixture over K archetypal feature-vectors.
    X = feat   # (n_agents, n_features) = (V, T) ✓
    return dict(
        path=path,
        day=d['day'],
        year=d['day'] / 365.0,
        N=len(ag),
        X=np.ascontiguousarray(X),
        ages=np.array([x['age'] for x in s], dtype=np.float32),
        fem=np.array([x['is_female'] for x in s]),
    )


def label_archetype(vec, labels, top_k=4):
    """Build a short human label by listing the strongest features (signed)."""
    order = np.argsort(-np.abs(vec))[:top_k]
    parts = []
    for j in order:
        sign = '+' if vec[j] > 0 else '−'
        parts.append(f"{sign}{labels[j]}")
    return '  '.join(parts)


def plot_simplex(weeks, results, K, out_path, age_color=True):
    """Project each agent's K-dim archetype mixture onto a 3-simplex (triangle)
    using the top-3 archetypes (by total population participation across
    weeks). Each agent is one dot. Color encodes age (or sex if requested)."""
    # Pick top 3 archetypes by total normalized mass across all agents
    total_mass = np.zeros(K)
    for r in results:
        # Use ReLU + row-normalize for sXC (it can have negatives in unsigned mode)
        sXC = np.maximum(r['sXC'], 0)
        total_mass += sXC.sum(axis=0)
    top3 = np.argsort(-total_mass)[:3]
    print(f"  simplex axes: archetypes {[int(k+1) for k in top3]} "
          f"(by total mass)")

    # Use latest week as the snapshot
    latest = weeks[-1]
    sXC = np.maximum(results[-1]['sXC'], 0)            # (N, K)
    sub = sXC[:, top3]                                  # (N, 3)
    row_sum = sub.sum(axis=1, keepdims=True) + 1e-12
    bary = sub / row_sum                                # convex coords (sum to 1)
    # Drop agents with no signal in any of the 3 (numerically zero rows)
    keep = sub.sum(axis=1) > 1e-6
    bary = bary[keep]; ages = latest['ages'][keep]; fem = latest['fem'][keep]

    # Map barycentric (a, b, c) to 2D triangle:
    # vertex 0 → (0, 0); vertex 1 → (1, 0); vertex 2 → (0.5, √3/2)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (np.sqrt(3) / 2) * bary[:, 2]

    fig, ax = plt.subplots(figsize=(9, 8))
    # triangle edges
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color='#666', lw=1.2)

    # interior gridlines (decile)
    for f in np.arange(0.1, 1.0, 0.1):
        # lines parallel to each edge
        for v_a, v_b, v_c in [
            ([(1-f), f, 0], [(1-f), 0, f]),  # parallel to edge opposite vertex 0
            ([f, (1-f), 0], [0, (1-f), f]),
            ([f, 0, (1-f)], [0, f, (1-f)]),
        ]:
            p1 = np.array(v_a); p2 = np.array(v_b)
            x1 = p1[1] + 0.5*p1[2]; y1 = (np.sqrt(3)/2)*p1[2]
            x2 = p2[1] + 0.5*p2[2]; y2 = (np.sqrt(3)/2)*p2[2]
            ax.plot([x1, x2], [y1, y2], color='#dddddd', lw=0.4, zorder=0)

    # scatter agents
    if age_color:
        c = ages
        sc = ax.scatter(x, y, c=c, cmap='viridis', s=22,
                         edgecolor='white', linewidth=0.3, alpha=0.85,
                         zorder=2)
        cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('age (years)')
    else:
        colors = ['#e94560' if f else '#3a6ea5' for f in fem]
        ax.scatter(x, y, c=colors, s=22, edgecolor='white',
                    linewidth=0.3, alpha=0.85, zorder=2)

    # vertex labels (archetype names)
    label_offset = 0.04
    ax.text(0 - label_offset, 0 - label_offset,
            f'Arch #{top3[0]+1}', fontsize=11, ha='right', va='top',
            fontweight='bold', color='#222')
    ax.text(1 + label_offset, 0 - label_offset,
            f'Arch #{top3[1]+1}', fontsize=11, ha='left', va='top',
            fontweight='bold', color='#222')
    ax.text(0.5, np.sqrt(3)/2 + label_offset,
            f'Arch #{top3[2]+1}', fontsize=11, ha='center', va='bottom',
            fontweight='bold', color='#222')

    ax.set_aspect('equal')
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
    ax.axis('off')
    ax.set_title(
        f"Agents on the simplex of top-3 archetypes — yr {latest['year']:.1f}, "
        f"pop {latest['N']}\n(each dot = one agent; corners = pure archetype)",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


def plot_archetype_bars(C, labels, out_path):
    """C: (n_features, K). Each col is a probability over features (sums to 1).
    Bar chart per archetype, with consistent feature ordering across cols."""
    K = C.shape[1]
    n_feat = C.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(3.4 * K + 1, 5))
    if K == 1:
        axes = [axes]
    vmax = float(C.max() * 1.1)
    for k in range(K):
        ax = axes[k]
        vals = C[:, k]
        ax.barh(range(n_feat), vals, color='#e94560',
                 edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', lw=0.6)
        ax.set_xlim(0, vmax)
        # annotate values > 0.05
        for j, v in enumerate(vals):
            if v > 0.05:
                ax.text(v + vmax * 0.02, j, f"{v:.2f}",
                         va='center', fontsize=7, color='#666')
        ax.set_title(f'Archetype #{k+1}', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle("Archetypal personality coalitions — each column is a probability "
                  "distribution over features (sums to 1)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=4, help='number of archetypes')
    p.add_argument('--weeks', nargs=2, type=int, default=None,
                    metavar=('LO','HI'),
                    help='week range (inclusive). default: all dumps')
    p.add_argument('--include-crests', action='store_true',
                    help='concatenate 256-d crest into the feature vector')
    p.add_argument('--signed', action='store_true',
                    help='sign-split each feature into (+, −) halves so AA '
                         'can find archetypes in both directions')
    p.add_argument('--maxiter', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='viz/archetypes')
    args = p.parse_args()

    def _week_num(p):
        try: return int(p.rsplit('week_', 1)[1].split('.')[0])
        except Exception: return 0
    paths = sorted(glob.glob('/mnt/anthrosim/dumps/week_*.json'), key=_week_num)
    if args.weeks:
        lo, hi = args.weeks
        paths = [p for p in paths
                 if lo <= int(p.split('week_')[1].split('.')[0]) <= hi]
    if not paths:
        raise SystemExit("no dumps in range")

    print(f"loading {len(paths)} dumps...")
    weeks = []
    for p in paths:
        w = load_week(p, include_crests=args.include_crests,
                       signed=args.signed)
        if w is None or w['N'] < args.K:
            continue
        weeks.append(w)
    print(f"  {len(weeks)} usable weeks (each pop ≥ K={args.K})")

    # Temporal MS-AA: SubjectT(X, sX) where X is (V, T). V varies per subject
    # (= agents that week), T is shared (= features). sX = X (no split-half).
    subjects = [SubjectT(X=w['X'], sX=w['X']) for w in weeks]
    print(f"  feature dim T (shared) = {weeks[0]['X'].shape[1]}")
    print(f"  agents per subject V: min={min(w['N'] for w in weeks)}, "
          f"max={max(w['N'] for w in weeks)}")

    print(f"\nrunning multi_subject_aa_T with K={args.K}, maxiter={args.maxiter}...")
    results, C, cost, varexpl, elapsed = multi_subject_aa_T(
        subjects, noc=args.K,
        opts={'maxiter': args.maxiter, 'rngSEED': args.seed,
              'heteroscedastic': True})

    print(f"\nvariance explained: {varexpl*100:.1f}%   "
          f"({elapsed:.1f}s, {len(cost)} iters)")

    # In MS-AA-T, the shared output is C (T, K) — each COLUMN is a probability
    # distribution over the T features, characterizing which features
    # compose archetype k. C cols sum to 1.
    base_labels = list(FEATURE_LABELS)
    if args.include_crests:
        n_crest_dims = ((C.shape[0] // (2 if args.signed else 1))
                        - len(FEATURE_LABELS))
        base_labels += [f'crest{i}' for i in range(n_crest_dims)]
    if args.signed:
        feat_labels = ([f'+{n}' for n in base_labels]
                        + [f'−{n}' for n in base_labels])
    else:
        feat_labels = base_labels

    print("\n=== Archetype profiles (feature coalition; each column is a prob over features) ===")
    for k in range(args.K):
        col = C[:, k]
        order = np.argsort(-col)
        top = [(feat_labels[j], col[j]) for j in order if col[j] > 0.02][:6]
        desc = ' + '.join(f"{p*100:.0f}% {n}" for n, p in top)
        print(f"\nArchetype #{k+1}: {desc}")

    # Per-agent score on each archetype: sXC[i] = X[i] · C is each agent's
    # raw score on the K archetype dimensions. Higher → more aligned with
    # that archetype. We softmax into a proper mixture for plotting.
    def softmax_rows(X, t=1.0):
        Z = X - X.max(axis=1, keepdims=True)
        e = np.exp(Z / t)
        return e / e.sum(axis=1, keepdims=True)

    # Latest week breakdown by dominant archetype (raw argmax)
    latest = weeks[-1]
    sXC_latest = results[-1]['sXC']                  # (N, K)
    dominant = sXC_latest.argmax(axis=1)
    print(f"\n=== Latest week (yr {latest['year']:.1f}, pop {latest['N']}): "
          f"dominant-archetype counts ===")
    for k in range(args.K):
        n = int((dominant == k).sum())
        print(f"  Archetype #{k+1}: {n} agents ({100*n/latest['N']:.1f}%)")

    # Save outputs
    import os
    os.makedirs(args.out, exist_ok=True)
    plot_archetype_bars(C, feat_labels, f'{args.out}/archetypes.png')
    plot_simplex(weeks, results, args.K, f'{args.out}/simplex.png')

    # Per-week dominant-archetype shares (stacked area over time)
    means = []
    for w, r in zip(weeks, results):
        d = r['sXC'].argmax(axis=1)
        m = np.array([(d == k).mean() for k in range(args.K)])
        means.append(m)
    means = np.stack(means, axis=0)
    years = np.array([w['year'] for w in weeks])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.Set2(np.linspace(0, 1, args.K))
    ax.stackplot(years, means.T,
                  labels=[f'Archetype #{k+1}' for k in range(args.K)],
                  colors=colors, edgecolor='white', linewidth=0.3)
    ax.set_xlabel('sim year'); ax.set_ylabel('share of population')
    ax.set_ylim(0, 1); ax.set_xlim(years.min(), years.max())
    ax.set_title("Dominant-archetype share of population over time", fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(f'{args.out}/mixture_over_time.png',
                dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {args.out}/mixture_over_time.png")


if __name__ == '__main__':
    main()
