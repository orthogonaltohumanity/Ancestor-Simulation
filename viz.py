"""Visualizations for anthrosim dumps.

Usage:
  python viz.py                    # use latest dump in dumps/
  python viz.py week_18            # specific dump
  python viz.py week_18 -o myviz   # custom output dir

Produces three figures in viz/<week>/:
  family_clusters.png   — crest cosine heatmap, sorted into clans
  cohort_cohesion.png   — feature-weight cosine similarity matrix between cohorts
  population_pyramid.png — back-to-back age × sex histogram
"""
import json
import sys
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DUMP_DIR = '/mnt/anthrosim/dumps'
COHORTS = [
    ("kids 0-6",     0,  6),
    ("forager 6-13", 6,  13),
    ("teens 13-16", 13, 16),
    ("young 16-30", 16, 30),
    ("middle 30-50", 30, 50),
    ("elders 50+",  50, 999),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _week_num(p):
    try: return int(os.path.basename(p).split('_')[1].split('.')[0])
    except Exception: return 0


def latest_dump():
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    if not paths:
        raise SystemExit("no dumps found")
    return paths[-1]


def load_arrays(path):
    with open(path) as f:
        d = json.load(f)
    ag = d['agents']
    if not ag:
        raise SystemExit(f"{path}: empty population")
    s = [a['state'] for a in ag]
    return dict(
        path=path,
        day=d['day'],
        year=d['day'] / 365.0,
        N=len(ag),
        states=s,
        age=np.array([x['age'] for x in s], dtype=np.float32),
        fem=np.array([x['is_female'] for x in s]),
        meno=np.array([x.get('is_menopausal', False) for x in s]),
        preg=np.array([x.get('is_pregnant', False) for x in s]),
        weights=np.array([x['weights'] for x in s], dtype=np.float32),
        crests=np.array([x['crest'] for x in s], dtype=np.float32),
        cache=np.array([x['cache'] for x in s], dtype=np.float32),
        hunger=np.array([x['hunger'] for x in s], dtype=np.float32),
        agents=ag,
    )


# ---------------------------------------------------------------------------
# 1. Family clusters: crest cosine heatmap, ordered by single-link clans
# ---------------------------------------------------------------------------

def find_clans(S, threshold=0.6):
    """Single-link clustering of agents by crest cosine similarity > threshold."""
    N = S.shape[0]
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    iu, ju = np.triu_indices(N, k=1)
    mask = S[iu, ju] > threshold
    for i, j in zip(iu[mask], ju[mask]):
        union(int(i), int(j))

    clan_of = [find(i) for i in range(N)]
    clans = {}
    for i, c in enumerate(clan_of):
        clans.setdefault(c, []).append(i)
    return list(clans.values())


def plot_family_clusters(a, out_path, threshold=0.6):
    crests = a['crests']
    N = a['N']
    S = crests @ crests.T
    clans = find_clans(S, threshold=threshold)
    # sort clans by size, biggest first
    clans.sort(key=len, reverse=True)
    order = [i for c in clans for i in c]
    S_ord = S[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                              gridspec_kw={'width_ratios': [3, 1]})

    # heatmap
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        'kin', [(0, '#1a1a2e'), (0.3, '#16213e'), (0.5, '#0f3460'),
                (0.7, '#e94560'), (1.0, '#ffd700')])
    im = ax.imshow(S_ord, cmap=cmap, vmin=-0.3, vmax=1.0,
                    interpolation='nearest', aspect='equal')

    # draw clan boundaries
    cum = 0
    boundaries = [0]
    for c in clans:
        cum += len(c)
        boundaries.append(cum)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', lw=0.6, alpha=0.5)
        ax.axvline(b - 0.5, color='white', lw=0.6, alpha=0.5)

    ax.set_title(f"Family clusters — crest cosine similarity\n"
                  f"yr {a['year']:.1f}, pop {N}, "
                  f"single-link clans @ cos > {threshold}: {len(clans)} clans",
                  fontsize=11)
    ax.set_xlabel("agent (sorted by clan)")
    ax.set_ylabel("agent (sorted by clan)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cosine similarity")

    # clan size bar chart
    ax2 = axes[1]
    sizes = sorted([len(c) for c in clans], reverse=True)[:20]
    colors = ['#e94560'] + ['#16213e'] * (len(sizes) - 1)
    ax2.barh(range(len(sizes)), sizes, color=colors)
    ax2.set_yticks(range(len(sizes)))
    ax2.set_yticklabels([f"#{i+1}" for i in range(len(sizes))], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("members")
    ax2.set_title(f"Top 20 clan sizes\n(biggest: {sizes[0]}, "
                   f"singletons: {sum(1 for c in clans if len(c)==1)})",
                   fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for i, v in enumerate(sizes):
        ax2.text(v + 0.2, i, str(v), va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 2. Cohort cohesion: feature-weight cosine similarity matrix
# ---------------------------------------------------------------------------

def plot_cohort_cohesion(a, out_path):
    W = a['weights']
    age = a['age']
    cohort_means = []
    labels = []
    counts = []
    for label, lo, hi in COHORTS:
        m = (age >= lo) & (age < hi)
        if not m.any():
            continue
        cohort_means.append(W[m].mean(axis=0))
        labels.append(f"{label}\n(n={int(m.sum())})")
        counts.append(int(m.sum()))
    if len(cohort_means) < 2:
        print(f"  (skip cohesion: only {len(cohort_means)} non-empty cohort)")
        return
    M = np.stack(cohort_means)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    cross = Mn @ Mn.T

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cross, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cross[i, j]
            color = 'white' if abs(v) > 0.5 else 'black'
            ax.text(j, i, f"{v:+.2f}", ha='center', va='center',
                    fontsize=10, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cosine similarity of cohort-mean weight vectors")

    # pop-wide cohesion as subtitle
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    upper = (Wn @ Wn.T)[np.triu_indices(a['N'], k=1)]
    ax.set_title(f"Cultural cohesion between cohorts — yr {a['year']:.1f}, "
                  f"pop {a['N']}\npop-wide pairwise cohesion: "
                  f"mean={upper.mean():+.3f}, std={upper.std():.3f}",
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Population pyramid: back-to-back age × sex histogram
# ---------------------------------------------------------------------------

def plot_population_pyramid(a, out_path, bin_width=2):
    age = a['age']
    fem = a['fem']
    preg = a['preg']
    meno = a['meno']
    max_age = max(int(age.max()) + bin_width, 60)
    bins = np.arange(0, max_age + bin_width, bin_width)
    centers = (bins[:-1] + bins[1:]) / 2.0

    male_counts, _ = np.histogram(age[~fem], bins=bins)
    female_counts, _ = np.histogram(age[fem], bins=bins)
    # decompose female into preg / meno / fertile / pre-fertile for layered bars
    female_preg = np.histogram(age[fem & preg], bins=bins)[0]
    female_meno = np.histogram(age[fem & meno & ~preg], bins=bins)[0]
    female_fertile = np.histogram(
        age[fem & ~preg & ~meno & (age >= 16)], bins=bins)[0]
    female_pre = np.histogram(age[fem & (age < 16)], bins=bins)[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    h = bin_width * 0.92

    # male bar (left, negative values)
    ax.barh(centers, -male_counts, height=h,
            color='#3a6ea5', edgecolor='white', linewidth=0.5,
            label=f"male (n={int((~fem).sum())})")

    # female stacked from right
    cum = np.zeros_like(centers)
    ax.barh(centers, female_pre, height=h, left=cum,
             color='#fce5d0', edgecolor='white', linewidth=0.5,
             label=f"female pre-fertile (n={int((fem & (age<16)).sum())})")
    cum += female_pre
    ax.barh(centers, female_fertile, height=h, left=cum,
             color='#e94560', edgecolor='white', linewidth=0.5,
             label=f"female fertile (n={int((fem & ~preg & ~meno & (age>=16)).sum())})")
    cum += female_fertile
    ax.barh(centers, female_preg, height=h, left=cum,
             color='#ffd700', edgecolor='white', linewidth=0.5,
             label=f"female pregnant (n={int((fem & preg).sum())})")
    cum += female_preg
    ax.barh(centers, female_meno, height=h, left=cum,
             color='#7d6b91', edgecolor='white', linewidth=0.5,
             label=f"female menopausal (n={int((fem & meno & ~preg).sum())})")

    # zero line
    ax.axvline(0, color='black', lw=0.8)

    # life stage horizontal lines
    for thr, label in [(6, 'forage'), (13, 'leave-camp'), (16, 'mate-age')]:
        ax.axhline(thr, color='gray', lw=0.4, ls=':', alpha=0.6)
        ax.text(ax.get_xlim()[1], thr + 0.1, label, fontsize=7,
                va='bottom', ha='right', color='gray')

    # symmetric x-axis with absolute-valued ticks
    xmax = max(male_counts.max(), female_counts.max()) + 1
    ax.set_xlim(-xmax, xmax)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t))}" for t in ticks])

    ax.set_xlabel("count  (← male       female →)")
    ax.set_ylabel("age (years)")
    ax.set_title(f"Population pyramid — yr {a['year']:.1f}, "
                  f"pop {a['N']}  ({int((~fem).sum())}M / {int(fem.sum())}F)",
                  fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 4. Weight values by age cohort × feature (heatmap)
# ---------------------------------------------------------------------------

FEATURE_LABELS = (
    'net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
    'age', 'is_pregnant', 'is_menopausal',
    'matings', 'gifts', 'family', 'camp$',
    'opinion_sim',
)


def plot_weight_heatmap(a, out_path):
    W = a['weights']
    age = a['age']
    n_feat = W.shape[1]
    labels = FEATURE_LABELS[:n_feat]

    rows = []
    row_stds = []
    row_labels = []
    for label, lo, hi in COHORTS:
        m = (age >= lo) & (age < hi)
        if not m.any():
            continue
        rows.append(W[m].mean(axis=0))
        # std across the cohort per feature; captures intra-cohort cultural
        # variance — a tight ±0.05 means everyone agrees, a loose ±0.4 means
        # the cohort is split on this preference.
        row_stds.append(W[m].std(axis=0, ddof=0) if m.sum() > 1
                         else np.zeros_like(W[m].mean(axis=0)))
        row_labels.append(f"{label}\n(n={int(m.sum())})")
    if not rows:
        print("  (skip weight heatmap: no cohorts)")
        return
    M = np.stack(rows)
    S = np.stack(row_stds)

    # symmetric color range driven by the data's own scale, capped at ±1
    vmax = float(min(1.0, max(0.2, np.abs(M).max() * 1.05)))

    fig, ax = plt.subplots(figsize=(13, 5 + 0.3 * len(rows)))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    aspect='auto')
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; sd = S[i, j]
            tcolor = 'white' if abs(v) > vmax * 0.55 else 'black'
            ax.text(j, i, f"{v:+.2f}\n±{sd:.2f}", ha='center', va='center',
                    fontsize=7.5, color=tcolor, linespacing=0.95)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean linear weight (♥ + / ✗ −)")
    ax.set_title(
        f"Cohort weight values — yr {a['year']:.1f}, pop {a['N']}\n"
        f"row = age cohort,  col = ranking feature  "
        f"(red = liked, blue = avoided)",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 5. Cache + hunger distributions (histograms split by life stage)
# ---------------------------------------------------------------------------

def plot_resource_distributions(a, out_path):
    cache = a['cache']
    hunger = a['hunger']
    age = a['age']

    # life-stage masks
    stages = [
        ('kid <13',     age < 13,            '#fce5d0'),
        ('teen 13-15',  (age >= 13) & (age < 16), '#f5b942'),
        ('adult 16+',   age >= 16,           '#3a6ea5'),
    ]

    # try to grab thresholds from main; fallback to literals
    try:
        import main as ref
        CACHE_LIMIT = ref.CACHE_LIMIT
        HUNGER_DEATH = ref.HUNGER_DEATH
        EAT_GATE = 250          # canonical "hunger > 250 → eat"
        WITHDRAW_GATE = 1500    # common evolved "hunger > 1500 → withdraw"
        DEPOSIT_GATE = 500      # canonical "cache > 500 → deposit"
    except Exception:
        CACHE_LIMIT = 12000
        HUNGER_DEATH = 120000
        EAT_GATE = 250
        WITHDRAW_GATE = 1500
        DEPOSIT_GATE = 500

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))

    # --- cache ---
    ax = axes[0]
    bins = np.linspace(0, max(CACHE_LIMIT, cache.max() + 1), 41)
    bottom = np.zeros(len(bins) - 1)
    for label, mask, color in stages:
        if not mask.any():
            continue
        h, _ = np.histogram(cache[mask], bins=bins)
        ax.bar((bins[:-1] + bins[1:]) / 2, h, width=np.diff(bins),
               bottom=bottom, color=color, edgecolor='white',
               linewidth=0.4, label=f"{label} (n={int(mask.sum())})")
        bottom += h
    ax.axvline(DEPOSIT_GATE, color='gray', lw=0.7, ls=':', alpha=0.7)
    ax.text(DEPOSIT_GATE, ax.get_ylim()[1] * 0.95,
            f' deposit gate\n cache>{DEPOSIT_GATE}',
            fontsize=7, color='gray', va='top')
    ax.axvline(CACHE_LIMIT, color='red', lw=0.7, ls=':', alpha=0.7)
    ax.text(CACHE_LIMIT, ax.get_ylim()[1] * 0.95,
            f' CACHE_LIMIT\n {CACHE_LIMIT}',
            fontsize=7, color='red', va='top', ha='right')
    ax.set_xlabel('cache (kcal)')
    ax.set_ylabel('agent count')
    ax.set_title(
        f"Personal cache distribution — yr {a['year']:.1f}, pop {a['N']}\n"
        f"mean={cache.mean():.0f}  median={np.median(cache):.0f}  "
        f"empty (≤0): {int((cache <= 0).sum())}  "
        f"at cap (≥{int(0.95*CACHE_LIMIT)}): {int((cache >= 0.95*CACHE_LIMIT).sum())}",
        fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- hunger ---
    ax = axes[1]
    # log-ish bin edges since hunger is unbounded; cap at 1.5× DEATH for plotting
    upper = max(HUNGER_DEATH * 1.1, hunger.max() + 1)
    bins = np.linspace(0, upper, 41)
    bottom = np.zeros(len(bins) - 1)
    for label, mask, color in stages:
        if not mask.any():
            continue
        h, _ = np.histogram(hunger[mask], bins=bins)
        ax.bar((bins[:-1] + bins[1:]) / 2, h, width=np.diff(bins),
               bottom=bottom, color=color, edgecolor='white',
               linewidth=0.4, label=f"{label} (n={int(mask.sum())})")
        bottom += h
    ax.axvline(EAT_GATE, color='gray', lw=0.7, ls=':', alpha=0.7)
    ax.text(EAT_GATE, ax.get_ylim()[1] * 0.95,
            f' eat gate\n hunger>{EAT_GATE}',
            fontsize=7, color='gray', va='top')
    ax.axvline(WITHDRAW_GATE, color='gray', lw=0.7, ls=':', alpha=0.7)
    ax.text(WITHDRAW_GATE, ax.get_ylim()[1] * 0.95,
            f' withdraw gate\n hunger>{WITHDRAW_GATE}',
            fontsize=7, color='gray', va='top')
    ax.axvline(HUNGER_DEATH, color='red', lw=0.7, ls=':', alpha=0.7)
    ax.text(HUNGER_DEATH, ax.get_ylim()[1] * 0.95,
            f' HUNGER_DEATH\n {HUNGER_DEATH}',
            fontsize=7, color='red', va='top', ha='right')
    ax.set_xlabel('hunger (kcal accumulated)')
    ax.set_ylabel('agent count')
    starving = int((hunger > HUNGER_DEATH * 0.5).sum())
    near_death = int((hunger > HUNGER_DEATH * 0.85).sum())
    ax.set_title(
        f"Hunger distribution — mean={hunger.mean():.0f}  "
        f"median={np.median(hunger):.0f}  "
        f"starving (>{int(HUNGER_DEATH*0.5)}): {starving}  "
        f"near death (>{int(HUNGER_DEATH*0.85)}): {near_death}",
        fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 6. Camp density & composition
# ---------------------------------------------------------------------------

def plot_camp_density(a, dump, out_path):
    """Four panels describing the multi-camp structure of the population:
      (1) per-camp population bar chart (sorted by size, descending)
      (2) per-camp larder cache bar chart (same camp order as #1)
      (3) per-camp mean within-camp crest-cosine — internal kinship density
      (4) per-camp mean family-weight (idx 10) — culture per camp
    """
    states = a['states']
    N = a['N']
    if 'camp_id' not in states[0]:
        print("  (skip camp_density: dumps don't have camp_id; old format)")
        return
    cids = np.array([s['camp_id'] for s in states], dtype=np.int64)
    W = a['weights']
    crests = a['crests']
    age = a['age']
    fem = a['fem']

    # Per-camp larders from dump['camps'] if present, else infer keys from cids
    larder = {}
    for c in dump.get('camps', []):
        larder[int(c['id'])] = float(c['cache'])

    # Build per-camp metrics in a single pass
    unique_camps = np.unique(cids)
    records = []
    for cid in unique_camps:
        m = cids == cid
        size = int(m.sum())
        if size == 0:
            continue
        # within-camp pairwise crest cosine (mean over upper-triangle)
        if size >= 2:
            sub = crests[m]
            S = sub @ sub.T
            iu, ju = np.triu_indices(size, k=1)
            mean_kin = float(S[iu, ju].mean())
        else:
            mean_kin = float('nan')
        rec = dict(
            id=int(cid),
            size=size,
            n_fem=int((m & fem).sum()),
            n_kids=int((m & (age < 13)).sum()),
            n_mate_age=int((m & (age >= 16)).sum()),
            cache=larder.get(int(cid), 0.0),
            mean_kin=mean_kin,
            mean_family_w=float(W[m, 10].mean()) if W.shape[1] > 10 else 0.0,
            mean_ndf_w=float(W[m, 0].mean()),
        )
        records.append(rec)

    # sort largest → smallest
    records.sort(key=lambda r: -r['size'])
    labels = [f"#{r['id']}" for r in records]
    sizes = np.array([r['size'] for r in records])
    caches = np.array([r['cache'] for r in records])
    kins = np.array([r['mean_kin'] for r in records])
    fams = np.array([r['mean_family_w'] for r in records])
    ndfs = np.array([r['mean_ndf_w'] for r in records])
    n_fems = np.array([r['n_fem'] for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (1) population per camp
    ax = axes[0, 0]
    pos = np.arange(len(records))
    n_males = sizes - n_fems
    ax.bar(pos, n_males, color='#3a6ea5', label='M', edgecolor='white', linewidth=0.4)
    ax.bar(pos, n_fems, bottom=n_males, color='#e94560', label='F',
            edgecolor='white', linewidth=0.4)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('members')
    ax.set_title(f"Population per camp  (pop {a['N']}, {len(records)} camps)",
                  fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # annotate
    for i, sz in enumerate(sizes):
        ax.text(i, sz + 0.3, str(int(sz)), ha='center', fontsize=7, color='#333')

    # (2) per-camp larder
    ax = axes[0, 1]
    ax.bar(pos, caches, color='#f5b942', edgecolor='white', linewidth=0.4)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('larder (kcal)')
    ax.set_title(f"Per-camp larder  total={caches.sum():,.0f}", fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (3) within-camp kinship density (mean crest cosine over pairs)
    ax = axes[1, 0]
    valid = ~np.isnan(kins)
    colors = ['#e94560' if k > 0.3 else '#3a6ea5' for k in kins]
    ax.bar(pos[valid], kins[valid], color=[c for c, v in zip(colors, valid) if v],
            edgecolor='white', linewidth=0.4)
    ax.axhline(0.6, color='red', lw=0.5, ls=':', alpha=0.6)
    ax.text(len(records)-1, 0.62, 'incest gate', ha='right', fontsize=7,
             color='red', alpha=0.8)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('mean pairwise crest cos')
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Within-camp kinship density  (red = clannish camp)",
                  fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # (4) per-camp cultural values (mean family-weight & ndf-weight)
    ax = axes[1, 1]
    w = 0.4
    ax.bar(pos - w/2, fams, width=w, color='#7d6b91',
            edgecolor='white', linewidth=0.4, label='family weight')
    ax.bar(pos + w/2, ndfs, width=w, color='#16213e',
            edgecolor='white', linewidth=0.4, label='ndf weight')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('mean weight')
    ax.set_title("Per-camp cultural lean (family vs net-debt-flow weights)",
                  fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle(f"Camp density & composition — yr {a['year']:.1f}, "
                  f"pop {a['N']}, {len(records)} camp{'s' if len(records)!=1 else ''}",
                  fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 7. Sex × cohort × action heatmap (graph composition by sex/life-stage)
# ---------------------------------------------------------------------------

ACTION_ORDER = (
    'agent_sleep', 'agent_wake', 'agent_eat', 'agent_hunt', 'agent_fish',
    'agent_gather', 'agent_go_to_camp', 'agent_leave_camp', 'agent_talk',
    'agent_listen', 'agent_propose_mate', 'agent_gift',
    'agent_deposit', 'agent_withdraw', 'agent_watch_children',
    'agent_idle', None,
)


def plot_action_by_sex(a, dump, out_path):
    """Two panels:
      Top:    sex × cohort × action — per-capita true_action frequency in graphs
              (rows: M-cohort, F-cohort; cols: action). Identifies who DOES what.
      Bottom: 2×2 cross-sex interaction matrices — gift flow and mating-event
              counts between (M, F) pairs. Identifies who INTERACTS with whom.
    """
    age = a['age']; fem = a['fem']
    cohort_defs = [
        ('forager 6-13',  6, 13),
        ('teen 13-16',   13, 16),
        ('young 16-30',  16, 30),
        ('middle 30-50', 30, 50),
        ('elders 50+',   50, 999),
    ]
    # Build sex × cohort cells of mean per-agent true_action counts
    rows = []     # list of (label, frac_vector_over_actions)
    for sex_label, sex_mask in [('M', ~fem), ('F', fem)]:
        for label, lo, hi in cohort_defs:
            mask = sex_mask & (age >= lo) & (age < hi)
            if not mask.any():
                continue
            # per-agent action counts → per-capita
            counts = np.zeros(len(ACTION_ORDER), dtype=np.float64)
            n_agents = 0
            for ag, m in zip(dump['agents'], mask):
                if not m:
                    continue
                for n in ag['graph']['nodes']:
                    ta = n['true_action']
                    if ta in ACTION_ORDER:
                        counts[ACTION_ORDER.index(ta)] += 1
                n_agents += 1
            if n_agents == 0:
                continue
            counts /= n_agents
            rows.append((f"{sex_label} {label}  (n={int(mask.sum())})", counts))

    if not rows:
        print("  (skip action-by-sex: no rows)")
        return

    M = np.stack([r[1] for r in rows], axis=0)
    labels = [r[0] for r in rows]
    col_labels = [a or '·none' for a in ACTION_ORDER]
    col_labels = [c.replace('agent_', '') for c in col_labels]

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], width_ratios=[1, 1],
                            hspace=0.5, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])

    im = ax.imshow(M, cmap='magma', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    # divider line between M and F blocks
    n_m = sum(1 for l in labels if l.startswith('M '))
    if 0 < n_m < len(labels):
        ax.axhline(n_m - 0.5, color='white', lw=2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("nodes per agent")
    ax.set_title(
        f"Sex × cohort × action — yr {a['year']:.1f}, pop {a['N']}\n"
        f"per-agent frequency of each true_action in their decision graph",
        fontsize=10)

    # --- bottom-left: cross-sex gift flow ---
    gift_pairs = dump.get('gift_pairs', [])
    G = np.zeros((2, 2), dtype=np.float64)        # rows=giver, cols=recip
    for i, j, amt in gift_pairs:
        # gift_matrix is directional (i=giver, j=recip)
        if i < a['N'] and j < a['N']:
            G[int(fem[i]), int(fem[j])] += float(amt)
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(G, cmap='Reds')
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['M', 'F'], fontsize=10)
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(['M', 'F'], fontsize=10)
    ax2.set_xlabel('recipient'); ax2.set_ylabel('giver')
    ax2.set_title(f"Gift flow (kcal)  total={G.sum():,.0f}", fontsize=10)
    for i in (0, 1):
        for j in (0, 1):
            ax2.text(j, i, f"{G[i,j]:,.0f}",
                      ha='center', va='center', fontsize=11,
                      color=('white' if G[i,j] > G.max()*0.55 else 'black'))
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)

    # --- bottom-right: mating events ---
    mp = dump.get('mating_pairs', [])
    Mp = np.zeros((2, 2), dtype=np.float64)
    for i, j, w in mp:
        if i < a['N'] and j < a['N']:
            si, sj = int(fem[i]), int(fem[j])
            # mating_matrix is symmetric → split half/half between (si,sj) cells
            Mp[si, sj] += float(w) * 0.5
            Mp[sj, si] += float(w) * 0.5
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(Mp, cmap='Blues')
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(['M', 'F'], fontsize=10)
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(['M', 'F'], fontsize=10)
    ax3.set_xlabel('partner B'); ax3.set_ylabel('partner A')
    ax3.set_title(f"Mating events (decayed)  total={Mp.sum():,.1f}", fontsize=10)
    for i in (0, 1):
        for j in (0, 1):
            ax3.text(j, i, f"{Mp[i,j]:.1f}",
                      ha='center', va='center', fontsize=11,
                      color=('white' if Mp[i,j] > Mp.max()*0.55 else 'black'))
    plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)

    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# 8. Time-of-day action rhythm — replay each agent's chain over 24 hours and
#    aggregate which actions fire at each hour. Reveals emergent daily
#    rhythms (siesta, dawn forage, midnight social, etc.) without needing
#    per-visit traces in the dump — we replay the chain walk statically
#    against each agent's dumped state.
# ---------------------------------------------------------------------------

# Truth-table bits for chunk_links (must match main.py)
_LINK_AND = 8
_LINK_OR  = 14
_LINK_TRUE = 15
_MAX_ACTION_REPLAY = 30      # visits per hour, matches main.MAX_ACTION
ORDERED_VARS = {'hunger', 'cache', 'tired', 'time', 'age', 'season', 'rng'}


def _eval_chunk_py(chunk, state, hour, season_val):
    """Evaluate one chunk against an agent's dumped state."""
    var = chunk['var']
    val = chunk['value']
    op = chunk['op']
    if var == 'rng':
        cur_val = np.random.random()
    elif var == 'time':
        cur_val = hour
    elif var == 'season':
        cur_val = season_val
    else:
        cur_val = state.get(var, 0)
    if var in ORDERED_VARS:
        if op == '<':  return cur_val < val
        if op == '>':  return cur_val > val
        if op == '==': return cur_val == val
        return False
    # bool var
    cur_b = bool(cur_val)
    if val is None: return cur_b
    return cur_b == bool(val)


def _eval_seq_py(seq, state, hour, season_val):
    """Left-fold a chunk_seq with chunk_links."""
    if not seq:
        return True
    chunks = seq['chunks']
    links = seq.get('links', [])
    if not chunks:
        return True
    acc = _eval_chunk_py(chunks[0], state, hour, season_val)
    for i, link in enumerate(links):
        if i + 1 >= len(chunks): break
        b = _eval_chunk_py(chunks[i + 1], state, hour, season_val)
        idx = (int(acc) << 1) | int(b)
        acc = bool((link >> idx) & 1)
    return acc


def _replay_agent_day(agent_dict, hours=range(24)):
    """For one agent, simulate its chain walk for each hour-of-day and tally
    which action-id fires at each hour. State is held FROZEN at the dump
    snapshot — we don't update hunger/cache/cur as the walk progresses
    (which would require fully reimplementing _apply_actions). This means
    the histogram reflects 'what behaviors would fire given current state
    at each time-of-day', not a true diurnal trace. Time-gated patterns
    (sleep at night, forage at noon) are still captured correctly since
    those gates only depend on the time chunks."""
    nodes = agent_dict['graph']['nodes']
    if not nodes: return None
    state = dict(agent_dict['state'])
    # Always use 'season=1' (mid-summer) to neutralize seasonality —
    # otherwise an off-season dump misleadingly suppresses forage.
    season_val = 1.0
    cur = int(state.get('cur', 0))
    if cur < 0 or cur >= len(nodes):
        cur = 0
    counts_per_hour = np.zeros((len(list(hours)), len(ACTION_ORDER)), dtype=np.int32)
    for h_idx, hour in enumerate(hours):
        for _ in range(_MAX_ACTION_REPLAY):
            n = nodes[cur]
            cond = _eval_seq_py(n.get('seq'), state, hour, season_val)
            action = n.get('true_action') if cond else n.get('false_action')
            if action in ACTION_ORDER:
                counts_per_hour[h_idx, ACTION_ORDER.index(action)] += 1
            nxt = n.get('true_node') if cond else n.get('false_node')
            ended = n.get('true_end') if cond else n.get('false_end')
            if nxt is None or nxt >= len(nodes):
                cur = 0
            else:
                cur = int(nxt)
            if ended:
                break
    return counts_per_hour


def plot_time_of_day_actions(dump, out_path, sample_n=200):
    """Stacked-area chart: hour-of-day vs population-wide action-firing rate.
    Replays each (sampled) agent's chain through a 24-hour cycle to aggregate
    action firings per hour. Reveals diurnal rhythms — sleep bunched at night,
    forage at midday, social-economic in evenings, plus any emergent siesta
    or other multi-modal patterns."""
    agents = dump['agents']
    if not agents:
        return
    if sample_n and sample_n < len(agents):
        rng = np.random.default_rng(0)
        sample = [agents[i] for i in rng.choice(len(agents), sample_n, replace=False)]
    else:
        sample = agents

    np.random.seed(0)
    totals = np.zeros((24, len(ACTION_ORDER)), dtype=np.int64)
    n_replayed = 0
    for a in sample:
        c = _replay_agent_day(a)
        if c is None: continue
        totals += c
        n_replayed += 1
    if n_replayed == 0:
        return
    # per-agent average per-hour firings, normalized to fractions
    per_hour = totals.astype(np.float64) / n_replayed   # avg fires per agent per hour
    # Group actions into semantic categories for the headline chart
    cats = {
        'sleep':     ['agent_sleep'],
        'forage':    ['agent_hunt', 'agent_fish', 'agent_gather'],
        'eat':       ['agent_eat'],
        'migrate':   ['agent_leave_camp', 'agent_make_camp', 'agent_join_camp', 'agent_go_to_camp'],
        'social':    ['agent_talk', 'agent_listen', 'agent_propose_mate', 'agent_gift', 'agent_watch_children'],
        'economic':  ['agent_deposit', 'agent_withdraw'],
        'other':     ['agent_wake', 'agent_idle'],
    }
    cat_data = np.zeros((24, len(cats)), dtype=np.float64)
    for ci, (_, members) in enumerate(cats.items()):
        for act in members:
            if act in ACTION_ORDER:
                cat_data[:, ci] += per_hour[:, ACTION_ORDER.index(act)]

    fig, (ax_cat, ax_act) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                          gridspec_kw={'hspace': 0.18})

    # Top: stacked area by semantic category
    hours_x = np.arange(24)
    cat_names = list(cats.keys())
    cat_colors = ['#1b3a6b', '#2a9d8f', '#e76f51', '#9b6f47',
                  '#a86fcf', '#d4a017', '#888888']
    ax_cat.stackplot(hours_x, cat_data.T, labels=cat_names,
                      colors=cat_colors[:len(cat_names)], alpha=0.92,
                      edgecolor='white', linewidth=0.4)
    ax_cat.set_xlim(0, 23)
    ax_cat.set_xticks(range(0, 24, 2))
    ax_cat.set_ylabel('avg action-fires per agent per hour')
    day = dump.get('day', 0); pop = dump.get('n_agents', len(agents))
    ax_cat.set_title(f"Daily rhythm — day {day}, pop {pop} (n={n_replayed} sampled)\n"
                      f"action-fires per agent at each hour-of-day, by category",
                      fontsize=11)
    # Shaded night band
    for x0, x1 in [(0, 5), (20, 23)]:
        ax_cat.axvspan(x0, x1, color='#0a0a3a', alpha=0.08, zorder=0)
    ax_cat.legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.95)
    ax_cat.grid(True, alpha=0.25, linewidth=0.5)

    # Bottom: heatmap of individual actions
    M = per_hour.T   # (action, hour)
    # Drop the None row
    keep = [i for i, a in enumerate(ACTION_ORDER) if a is not None]
    M = M[keep]
    act_labels = [ACTION_ORDER[i].replace('agent_', '') for i in keep]
    im = ax_act.imshow(M, cmap='magma', aspect='auto', interpolation='nearest')
    ax_act.set_yticks(range(len(act_labels)))
    ax_act.set_yticklabels(act_labels, fontsize=8)
    ax_act.set_xlabel('hour of day')
    ax_act.set_xticks(range(0, 24, 2))
    cb = plt.colorbar(im, ax=ax_act, fraction=0.025, pad=0.02)
    cb.set_label('avg fires / agent / hr', fontsize=8)
    ax_act.set_title('per-action heatmap (same data, ungrouped)', fontsize=10)

    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path} (replayed {n_replayed} agents × 24 hours)")


def make_grid_animation(out_path, fps=8, dpi=110, sample_every=1, max_frames=None):
    """Build a GIF of the spatial grid over time. One frame per daily dump.
    Layout is a 2×2 panel to keep density-heavy grids readable:
      TL: resources (RGB: R=hunt, G=gather, B=fish — black=depleted, bright=full)
      TR: total population per cell (viridis heatmap)
      BL: ♀ density per cell (reds)
      BR: ♂ density per cell (blues)
    Camps overlaid on all four panels as yellow squares (size ∝ log larder).
    Per-cell counts are written in-cell when small enough.
    Reads every dump in DUMP_DIR; assumes daily cadence."""
    import matplotlib.animation as anim
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    if not paths:
        print('  (no dumps to animate)')
        return
    if sample_every > 1:
        paths = paths[::sample_every]
    if max_frames:
        paths = paths[:max_frames]
    d0 = json.load(open(paths[0]))
    if 'resources_grid' in d0:
        grid_shape = (len(d0['resources_grid']), len(d0['resources_grid'][0]))
    else:
        grid_shape = (10, 10)
    GW, GH = grid_shape

    import main as m
    K = np.array([m.RESOURCE_K_HUNT_PER_CELL,
                  m.RESOURCE_K_FISH_PER_CELL,
                  m.RESOURCE_K_GATHER_PER_CELL])
    # Load the terrain mask (impassable cells rendered as gray on every panel).
    terrain_mask = None
    terrain_path = f'{DUMP_DIR}/terrain.json'
    if os.path.exists(terrain_path):
        with open(terrain_path) as _tf:
            terrain_mask = np.array(json.load(_tf)['mask'], dtype=bool)
    impassable_T = (~terrain_mask).T if terrain_mask is not None else None

    # Pre-pass to compute max counts for stable color scales.
    print(f'  pre-loading {len(paths)} frames...')
    raw_frames = []
    max_pop = 1
    for p in paths:
        try:
            with open(p) as f: d = json.load(f)
        except json.JSONDecodeError:
            continue
        if not d.get('agents'):
            continue
        pop = np.zeros((GH, GW), dtype=np.int32)
        fem = np.zeros((GH, GW), dtype=np.int32)
        mal = np.zeros((GH, GW), dtype=np.int32)
        for ag in d['agents']:
            sa = ag['state']
            ax_ = int(sa.get('x', 0)) % GW
            ay_ = int(sa.get('y', 0)) % GH
            pop[ay_, ax_] += 1
            if sa.get('is_female'):
                fem[ay_, ax_] += 1
            else:
                mal[ay_, ax_] += 1
        if 'resources_grid' in d:
            res = np.array(d['resources_grid'], dtype=np.float64)
        else:
            res = np.tile(K.reshape(1, 1, 3), (GW, GH, 1))
        rk = np.clip(res / np.maximum(K.reshape(1, 1, 3), 1e-9), 0, 1)
        bg_rgb = np.zeros((GH, GW, 3), dtype=np.float32)
        bg_rgb[..., 0] = rk[..., 0].T   # R = hunt
        bg_rgb[..., 1] = rk[..., 2].T   # G = gather
        bg_rgb[..., 2] = rk[..., 1].T   # B = fish
        # Impassable terrain → black on every panel.
        if impassable_T is not None:
            bg_rgb[impassable_T] = 0.0
            pop = pop.astype(np.float32); pop[impassable_T] = np.nan
            fem = fem.astype(np.float32); fem[impassable_T] = np.nan
            mal = mal.astype(np.float32); mal[impassable_T] = np.nan
        camps = [(c.get('x', 0), c.get('y', 0), c.get('cache', 0))
                 for c in d.get('camps', [])]
        # Season field: a sin wave over y whose phase sweeps with time.
        # season_at_y is a function of row y only, so tile it across x.
        season_col = np.asarray(m.season_at_y(np.arange(GH), d['day'] * 24),
                                 dtype=np.float32)
        season_grid = np.tile(season_col.reshape(GH, 1), (1, GW))
        if impassable_T is not None:
            season_grid = season_grid.copy()
            season_grid[impassable_T] = np.nan
        max_pop = max(max_pop, int(np.nanmax(pop)) if pop.size else 1)
        raw_frames.append((d['day'], d.get('n_agents', 0), bg_rgb,
                            pop, fem, mal, camps, season_grid))
    print(f'  {len(raw_frames)} frames ready (max cell pop = {max_pop})')

    # Use a fixed display-cap so the color scale is stable across the run.
    pop_cap = max(4, max_pop)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5))
    (ax_res, ax_pop, ax_season), (ax_fem, ax_mal, ax_blank) = axes
    ax_blank.axis('off')
    panels = [
        (ax_res, 'resources',   None,         None,  None),
        (ax_pop, 'population',  'viridis',    pop_cap, '#000'),
        (ax_fem, '♀ density',   'Reds',       pop_cap, '#fff'),
        (ax_mal, '♂ density',   'Blues',      pop_cap, '#fff'),
    ]
    artists = {}
    for ax, label, cmap, vmax, cell_text_color in panels:
        if cmap is None:
            im = ax.imshow(np.zeros((GH, GW, 3), dtype=np.float32),
                            origin='upper', extent=(-0.5, GW - 0.5, GH - 0.5, -0.5),
                            interpolation='nearest', zorder=0)
        else:
            # PowerNorm(gamma=0.5) = sqrt scaling; brings out single-agent
            # cells against mostly-empty grids while compressing the high end.
            from matplotlib.colors import PowerNorm
            import copy as _copy
            cmap_obj = _copy.copy(plt.get_cmap(cmap))
            cmap_obj.set_bad('black')   # NaN cells → black (impassable terrain)
            im = ax.imshow(np.zeros((GH, GW), dtype=np.float32),
                            cmap=cmap_obj,
                            norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax),
                            origin='upper', extent=(-0.5, GW - 0.5, GH - 0.5, -0.5),
                            interpolation='nearest', zorder=0)
        # Per-cell count labels — only drawn for small enough grids.
        cell_labels = []
        if cmap is not None and GW <= 25 and GH <= 25:
            for yy in range(GH):
                row = []
                for xx in range(GW):
                    t = ax.text(xx, yy, '', ha='center', va='center',
                                fontsize=max(5, 9 - max(GW, GH) // 5),
                                color=cell_text_color or 'white',
                                zorder=4)
                    row.append(t)
                cell_labels.append(row)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(0, GW, max(1, GW // 10)))
        ax.set_yticks(range(0, GH, max(1, GH // 10)))
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, GW - 0.5)
        ax.set_ylim(GH - 0.5, -0.5)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        artists[label] = (im, cell_labels)
        if cmap is not None:
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.tick_params(labelsize=7)

    # Season panel — [0,1] field, diverging colormap (blue=winter, red=summer).
    # The band is a sin wave over y whose phase sweeps through time, so this
    # panel animates as the seasons migrate along the vertical axis.
    from matplotlib.colors import Normalize
    import copy as _copy2
    season_cmap = _copy2.copy(plt.get_cmap('RdBu_r'))
    season_cmap.set_bad('black')   # impassable terrain → black, as on other panels
    season_im = ax_season.imshow(np.zeros((GH, GW), dtype=np.float32),
                                  cmap=season_cmap, norm=Normalize(vmin=0.0, vmax=1.0),
                                  origin='upper', extent=(-0.5, GW - 0.5, GH - 0.5, -0.5),
                                  interpolation='nearest', zorder=0)
    ax_season.set_title('season  (red=summer  blue=winter)', fontsize=11)
    ax_season.set_xticks(range(0, GW, max(1, GW // 10)))
    ax_season.set_yticks(range(0, GH, max(1, GH // 10)))
    ax_season.tick_params(labelsize=7)
    ax_season.set_xlim(-0.5, GW - 0.5)
    ax_season.set_ylim(GH - 0.5, -0.5)
    ax_season.set_aspect('equal')
    for spine in ax_season.spines.values():
        spine.set_edgecolor('#444')
    cbar = fig.colorbar(season_im, ax=ax_season, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    artists['season'] = (season_im, [])

    fig.text(0.02, 0.97, '', fontsize=13, fontweight='bold',
              color='#222')
    sup = fig.texts[-1]
    fig.text(0.02, 0.01,
              'resources panel — R=hunt  G=gather  B=fish  (R/K per cell)',
              fontsize=8, color='#444')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    def update(frame_idx):
        day, n_ag, bg_rgb, pop, fem, mal, camps, season_grid = raw_frames[frame_idx]
        updated = []
        for label, grid, fmt in [
            ('resources',  bg_rgb,      None),
            ('population', pop,         '{:d}'),
            ('♀ density',  fem,         '{:d}'),
            ('♂ density',  mal,         '{:d}'),
            ('season',     season_grid, None),
        ]:
            im, cell_labels = artists[label]
            im.set_data(grid)
            if cell_labels and fmt is not None:
                for yy in range(GH):
                    for xx in range(GW):
                        raw = grid[yy, xx]
                        if raw != raw:   # NaN — works for numpy scalars too
                            cell_labels[yy][xx].set_text('')
                            continue
                        v = int(raw)
                        cell_labels[yy][xx].set_text(fmt.format(v) if v > 0 else '')
            updated.append(im)
        sup.set_text(f'day {day}   •   pop {n_ag}   •   camps {len(camps)}')
        return updated

    ani = anim.FuncAnimation(fig, update, frames=len(raw_frames),
                              interval=1000 // fps, blit=False)
    print(f'  rendering gif → {out_path} ({len(raw_frames)} frames @ {fps} fps)...')
    ani.save(out_path, writer=anim.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f'  wrote {out_path}')


def make_culture_animation(out_path, fps=8, dpi=110, sample_every=1, max_frames=None):
    """Build a GIF of the spatial distribution of CULTURE — one heatmap per
    social-ranking feature. For each grid cell, average that feature's weight
    across every agent standing on the cell (all camps pooled). Cells with no
    agents and impassable terrain render black.

    Weights are signed (born ~N(0, WEIGHT_INIT_STD), drifting via cultural
    learning), so each panel uses a diverging colormap centred at 0 with a
    per-feature symmetric scale held fixed across the run."""
    import matplotlib.animation as anim
    from matplotlib.colors import Normalize
    import copy as _copy
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    if not paths:
        print('  (no dumps to animate)')
        return
    if sample_every > 1:
        paths = paths[::sample_every]
    if max_frames:
        paths = paths[:max_frames]
    d0 = json.load(open(paths[0]))
    if 'resources_grid' in d0:
        grid_shape = (len(d0['resources_grid']), len(d0['resources_grid'][0]))
    else:
        grid_shape = (10, 10)
    GW, GH = grid_shape

    import main as m
    feat_names = list(m.SOCIAL_FEATURES)
    n_feat = len(feat_names)

    terrain_mask = None
    terrain_path = f'{DUMP_DIR}/terrain.json'
    if os.path.exists(terrain_path):
        with open(terrain_path) as _tf:
            terrain_mask = np.array(json.load(_tf)['mask'], dtype=bool)
    impassable_T = (~terrain_mask).T if terrain_mask is not None else None

    # Pre-pass: per frame, build (n_feat, GH, GW) per-cell mean-weight grids.
    print(f'  pre-loading {len(paths)} frames...')
    raw_frames = []
    feat_absmax = np.full(n_feat, 1e-6)
    for p in paths:
        try:
            with open(p) as f: d = json.load(f)
        except json.JSONDecodeError:
            continue
        if not d.get('agents'):
            continue
        wsum = np.zeros((n_feat, GH, GW), dtype=np.float64)
        cnt  = np.zeros((GH, GW), dtype=np.int32)
        for ag in d['agents']:
            sa = ag['state']
            w = sa.get('weights')
            if not w:
                continue
            ax_ = int(sa.get('x', 0)) % GW
            ay_ = int(sa.get('y', 0)) % GH
            wsum[:, ay_, ax_] += w[:n_feat]
            cnt[ay_, ax_] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_w = wsum / cnt[None, :, :]
        mean_w[:, cnt == 0] = np.nan          # empty cells → black
        if impassable_T is not None:
            mean_w[:, impassable_T] = np.nan  # impassable terrain → black
        for i in range(n_feat):
            finite = mean_w[i][np.isfinite(mean_w[i])]
            if finite.size:
                feat_absmax[i] = max(feat_absmax[i], float(np.abs(finite).max()))
        raw_frames.append((d['day'], d.get('n_agents', 0), mean_w))
    print(f'  {len(raw_frames)} frames ready')

    ncols = 4
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    ims = []
    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_feat:
            ax.axis('off')
            ims.append(None)
            continue
        cmap = _copy.copy(plt.get_cmap('RdBu_r'))
        cmap.set_bad('black')
        im = ax.imshow(np.full((GH, GW), np.nan, dtype=np.float32),
                        cmap=cmap,
                        norm=Normalize(vmin=-feat_absmax[i], vmax=feat_absmax[i]),
                        origin='upper', extent=(-0.5, GW - 0.5, GH - 0.5, -0.5),
                        interpolation='nearest', zorder=0)
        ax.set_title(feat_names[i], fontsize=10)
        ax.set_xticks(range(0, GW, max(1, GW // 6)))
        ax.set_yticks(range(0, GH, max(1, GH // 6)))
        ax.tick_params(labelsize=6)
        ax.set_xlim(-0.5, GW - 0.5)
        ax.set_ylim(GH - 0.5, -0.5)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        ims.append(im)

    fig.text(0.02, 0.985, '', fontsize=13, fontweight='bold', color='#222')
    sup = fig.texts[-1]
    fig.text(0.02, 0.005,
              'culture — per-cell mean social-ranking weight (all agents pooled).  '
              'red = positive  •  blue = negative  •  black = empty / impassable',
              fontsize=8, color='#444')
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))

    def update(frame_idx):
        day, n_ag, mean_w = raw_frames[frame_idx]
        updated = []
        for i in range(n_feat):
            ims[i].set_data(mean_w[i])
            updated.append(ims[i])
        sup.set_text(f'day {day}   •   pop {n_ag}   •   culture map')
        return updated

    ani = anim.FuncAnimation(fig, update, frames=len(raw_frames),
                              interval=1000 // fps, blit=False)
    print(f'  rendering gif → {out_path} ({len(raw_frames)} frames @ {fps} fps)...')
    ani.save(out_path, writer=anim.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_camp_encounter_distribution(dump, out_path):
    """Histogram: per-agent, how many OTHER agents share their camp at this
    snapshot. For an agent in a 17-person camp, that's 16 daily camp-mates.
    Wanderers (camp_id == -1) get 0 (they encounter no one in-camp).
    Captures the realized social-pool size each agent operates within —
    Dunbar-style. Side panel: unique social-tie count from gift+mating
    matrices (counts distinct partners with any recent (non-decayed)
    interaction). Together: passive co-residence vs active social network."""
    agents = dump['agents']
    if not agents:
        return
    N = len(agents)
    # passive: camp co-residence
    from collections import Counter
    camp_counts = Counter()
    for a in agents:
        cid = a['state'].get('camp_id', -1)
        if cid >= 0:
            camp_counts[cid] += 1
    encounters = np.zeros(N, dtype=np.int32)
    for idx, a in enumerate(agents):
        cid = a['state'].get('camp_id', -1)
        if cid >= 0:
            encounters[idx] = camp_counts[cid] - 1   # exclude self
    # active: distinct tie partners (gift OR mating matrix entries above eps)
    # Build per-agent sets of partner indices from the pair lists.
    partners = [set() for _ in range(N)]
    eps = 1e-3
    for src, dst, val in dump.get('gift_pairs', []):
        if val > eps:
            partners[int(src)].add(int(dst))
            partners[int(dst)].add(int(src))   # gift is directional but the
                                                # tie itself is mutual knowledge
    for src, dst, val in dump.get('mating_pairs', []):
        if val > eps:
            partners[int(src)].add(int(dst))
            partners[int(dst)].add(int(src))
    tie_counts = np.array([len(p) for p in partners], dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Left: camp encounters
    enc_max = int(encounters.max()) if encounters.size else 0
    bins = np.arange(0, max(2, enc_max + 2)) - 0.5
    ax1.hist(encounters, bins=bins, color='#2a9d8f', edgecolor='white',
              linewidth=0.6, alpha=0.92)
    med = float(np.median(encounters))
    mean = float(encounters.mean())
    n_wand = int((encounters == 0).sum())
    ax1.axvline(med, color='black', linestyle='--', linewidth=1, alpha=0.7,
                 label=f'median = {med:.0f}')
    ax1.axvline(mean, color='#e76f51', linestyle=':', linewidth=1.5,
                 label=f'mean = {mean:.1f}')
    ax1.set_xlabel('camp-mates (other agents in same camp)')
    ax1.set_ylabel('agents')
    day = dump.get('day', 0)
    ax1.set_title(f'In-camp encounters per agent — day {day}\n'
                   f'{N} agents, {n_wand} wanderers (0 encounters)',
                   fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    # Right: active social ties
    tie_max = int(tie_counts.max()) if tie_counts.size else 0
    bins2 = np.arange(0, max(2, tie_max + 2)) - 0.5
    ax2.hist(tie_counts, bins=bins2, color='#a86fcf', edgecolor='white',
              linewidth=0.6, alpha=0.92)
    med2 = float(np.median(tie_counts))
    mean2 = float(tie_counts.mean())
    ax2.axvline(med2, color='black', linestyle='--', linewidth=1, alpha=0.7,
                 label=f'median = {med2:.0f}')
    ax2.axvline(mean2, color='#e76f51', linestyle=':', linewidth=1.5,
                 label=f'mean = {mean2:.1f}')
    ax2.set_xlabel('distinct social-tie partners (gift OR mating matrix > 0)')
    ax2.set_ylabel('agents')
    ax2.set_title('Active social ties per agent (decay-weighted)', fontsize=10)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


def plot_ties_by_sex_age(dump, out_path):
    """Heatmap grid: rows = sex (M, F), cols = age cohort, cells = mean number
    of distinct social-tie partners per agent in that bin. Three panels —
    mating partners, gift partners (gift_matrix has direction so any
    in-or-out tie counts as a partner), and the union (any tie).
    Reveals life-stage shape of the social network: which sex-age cells
    are hubs vs. periphery."""
    agents = dump['agents']
    if not agents:
        return
    N = len(agents)
    sex = np.array([a['state']['is_female'] for a in agents], dtype=bool)
    age = np.array([a['state']['age'] for a in agents], dtype=np.float32)

    mate_p = [set() for _ in range(N)]
    gift_p = [set() for _ in range(N)]
    eps = 1e-3
    for i, j, v in dump.get('mating_pairs', []):
        if v > eps:
            mate_p[int(i)].add(int(j))
            mate_p[int(j)].add(int(i))
    for src, dst, v in dump.get('gift_pairs', []):
        if v > eps:
            gift_p[int(src)].add(int(dst))
            gift_p[int(dst)].add(int(src))
    union_p = [mate_p[i] | gift_p[i] for i in range(N)]
    mate_n = np.array([len(p) for p in mate_p], dtype=np.float32)
    gift_n = np.array([len(p) for p in gift_p], dtype=np.float32)
    union_n = np.array([len(p) for p in union_p], dtype=np.float32)

    age_bins = [(0, 6),   (6, 13),  (13, 16), (16, 22),
                (22, 30), (30, 40), (40, 55), (55, 999)]
    bin_labels = ['0–5\nbaby/kid', '6–12\nforager', '13–15\nteen',
                   '16–21\nyoung adult', '22–29\nyoung', '30–39\nmid',
                   '40–54\nelder', '55+\nold']

    def build_grid(values):
        grid = np.zeros((2, len(age_bins)), dtype=np.float32)
        counts = np.zeros((2, len(age_bins)), dtype=np.int32)
        for r, sex_mask in enumerate([~sex, sex]):     # row 0=M, row 1=F
            for c, (lo, hi) in enumerate(age_bins):
                m = sex_mask & (age >= lo) & (age < hi)
                n = int(m.sum())
                counts[r, c] = n
                grid[r, c] = float(values[m].mean()) if n else np.nan
        return grid, counts

    g_mate, _    = build_grid(mate_n)
    g_gift, _    = build_grid(gift_n)
    g_union, cnt = build_grid(union_n)

    # Shared color scale across panels for direct comparison
    vmax = float(np.nanmax([np.nanmax(g_mate), np.nanmax(g_gift), np.nanmax(g_union)]))
    vmax = max(vmax, 1.0)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8.5),
                             gridspec_kw={'hspace': 0.55})
    panels = [
        ('Mating-pair partners (mating_matrix > 0)', g_mate),
        ('Gift partners (gift_matrix in or out > 0)', g_gift),
        ('Union of all social ties (mating OR gift)',  g_union),
    ]
    for ax, (title, grid) in zip(axs, panels):
        im = ax.imshow(grid, cmap='viridis', aspect='auto',
                        vmin=0, vmax=vmax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['M', 'F'], fontsize=11)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        for r in range(2):
            for c in range(len(age_bins)):
                v = grid[r, c]; n = cnt[r, c]
                if np.isnan(v):
                    ax.text(c, r, f'∅\nn=0', ha='center', va='center',
                            color='gray', fontsize=8)
                else:
                    tcolor = 'white' if v < vmax * 0.5 else 'black'
                    ax.text(c, r, f'{v:.0f}\nn={n}', ha='center', va='center',
                            color=tcolor, fontsize=9)
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('mean partners', fontsize=8)

    day = dump.get('day', 0)
    fig.suptitle(f'Social-tie count by sex × age — day {day}, pop {N}\n'
                  f'cell = mean distinct partners (mating + gift matrices); '
                  f'n = agents in cell',
                  fontsize=12, y=1.0)
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  wrote {out_path}')


def plot_time_of_day_by_camp(dump, out_path, n_camps=6, min_members=3):
    """Per-camp daily-rhythm small-multiples. Each panel is one camp's
    population-averaged action-fires-per-hour, broken down by semantic
    category. Reveals inter-camp cultural divergence: do different camps
    have different daily schedules? (Different forage/social/sleep
    rhythms, different "meeting hours", etc.)"""
    agents = dump['agents']
    if not agents:
        return
    # group by camp_id
    by_camp = {}
    for a in agents:
        cid = a['state'].get('camp_id', -1)
        if cid < 0:
            continue   # skip wanderers
        by_camp.setdefault(cid, []).append(a)
    # pick top n_camps by size with at least min_members
    real_camps = sorted(
        ((cid, members) for cid, members in by_camp.items()
         if len(members) >= min_members),
        key=lambda kv: -len(kv[1]))[:n_camps]
    if not real_camps:
        print("  (skip time-of-day-by-camp: no camps ≥ min_members)")
        return

    cats = {
        'sleep':     ['agent_sleep'],
        'forage':    ['agent_hunt', 'agent_fish', 'agent_gather'],
        'eat':       ['agent_eat'],
        'migrate':   ['agent_leave_camp', 'agent_make_camp', 'agent_join_camp', 'agent_go_to_camp'],
        'social':    ['agent_talk', 'agent_listen', 'agent_propose_mate', 'agent_gift', 'agent_watch_children'],
        'economic':  ['agent_deposit', 'agent_withdraw'],
        'other':     ['agent_wake', 'agent_idle'],
    }
    cat_names = list(cats.keys())
    cat_colors = ['#1b3a6b', '#2a9d8f', '#e76f51', '#9b6f47',
                  '#a86fcf', '#d4a017', '#888888']

    # determine global ymax across camps so panels share scale
    np.random.seed(0)
    per_camp_data = []
    global_ymax = 0.0
    for cid, members in real_camps:
        totals = np.zeros((24, len(ACTION_ORDER)), dtype=np.int64)
        n_rep = 0
        for ag in members:
            c = _replay_agent_day(ag)
            if c is None: continue
            totals += c
            n_rep += 1
        if n_rep == 0:
            per_camp_data.append((cid, members, None))
            continue
        per_hour = totals.astype(np.float64) / n_rep
        cat_data = np.zeros((24, len(cats)), dtype=np.float64)
        for ci, (_, mem) in enumerate(cats.items()):
            for act in mem:
                if act in ACTION_ORDER:
                    cat_data[:, ci] += per_hour[:, ACTION_ORDER.index(act)]
        global_ymax = max(global_ymax, float(cat_data.sum(axis=1).max()))
        per_camp_data.append((cid, members, cat_data))

    K = len(real_camps)
    cols = min(3, K)
    rows_n = (K + cols - 1) // cols
    fig, axs = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.0 * rows_n),
                              sharex=True, sharey=True,
                              squeeze=False)
    hours_x = np.arange(24)
    for idx, (cid, members, cat_data) in enumerate(per_camp_data):
        ax = axs[idx // cols][idx % cols]
        if cat_data is None:
            ax.set_visible(False); continue
        ax.stackplot(hours_x, cat_data.T, colors=cat_colors[:len(cat_names)],
                      alpha=0.92, edgecolor='white', linewidth=0.3)
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        if global_ymax > 0:
            ax.set_ylim(0, global_ymax * 1.05)
        for x0, x1 in [(0, 5), (20, 23)]:
            ax.axvspan(x0, x1, color='#0a0a3a', alpha=0.07, zorder=0)
        ax.set_title(f"camp {cid}  (n={len(members)})", fontsize=10)
        ax.grid(True, alpha=0.2, linewidth=0.4)
    # hide unused axes
    for idx in range(len(per_camp_data), rows_n * cols):
        axs[idx // cols][idx % cols].set_visible(False)
    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[i]) for i in range(len(cat_names))]
    fig.legend(handles, cat_names, loc='lower center',
                ncol=len(cat_names), bbox_to_anchor=(0.5, -0.02), fontsize=9)
    day = dump.get('day', 0)
    fig.suptitle(f"Daily rhythm by camp — day {day} "
                  f"(top {len(real_camps)} camps by size)",
                  fontsize=12, y=1.0)
    fig.text(0.5, 0.01, 'hour of day', ha='center', fontsize=10)
    fig.text(0.005, 0.5, 'avg action-fires / agent / hr', va='center',
              rotation='vertical', fontsize=10)
    plt.tight_layout(rect=(0.02, 0.04, 1, 0.98))
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path} (top {len(real_camps)} camps)")


# ---------------------------------------------------------------------------
# 9. Mating-pair sex composition over time (MM / MF / FF)
# ---------------------------------------------------------------------------

def plot_forage_over_time(out_path):
    """Two stacked panels across all dumps:
      top    — per-capita absolute count of hunt/fish/gather nodes per agent
      bottom — proportion of forage-mass split between the three
    Tracks dietary regime shifts: when gather/fish pools deplete, lineages
    with high hunt-node counts gain relative advantage and can spread the
    practice via subgraph adoption."""
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    years = []
    hunt_per = []; fish_per = []; gath_per = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        agents = d.get('agents', [])
        if not agents:
            continue
        h = f_ = g = 0
        for a in agents:
            for n in a['graph']['nodes']:
                ta = n.get('true_action')
                if ta == 'agent_hunt':   h += 1
                elif ta == 'agent_fish': f_ += 1
                elif ta == 'agent_gather': g += 1
        n = len(agents)
        years.append(d.get('day', 0) / 365.0)
        hunt_per.append(h / n)
        fish_per.append(f_ / n)
        gath_per.append(g / n)
    if not years:
        return
    H = np.array(hunt_per); F = np.array(fish_per); G = np.array(gath_per)
    tot = H + F + G
    safe = np.maximum(tot, 1e-9)
    H_pct = H / safe; F_pct = F / safe; G_pct = G / safe

    fig, (ax_abs, ax_pct) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                          gridspec_kw={'hspace': 0.12})
    # absolute counts
    ax_abs.plot(years, G, color='#2a9d8f', lw=2.0, label='gather (yield∝R/K)')
    ax_abs.plot(years, F, color='#577590', lw=1.6, label='fish   (yield∝R/K)')
    ax_abs.plot(years, H, color='#e76f51', lw=1.6, label='hunt   (prob∝R/K)')
    ax_abs.plot(years, tot, color='black', lw=0.8, linestyle=':',
                 label='total forage', alpha=0.5)
    ax_abs.set_ylabel('forage nodes per agent\n(true_action count)')
    ax_abs.set_title(
        f'Forage action composition over time — {len(years)} dumps, '
        f'day {int(years[-1]*365)}\n'
        f'per-agent count of hunt/fish/gather nodes in their decision graphs',
        fontsize=11)
    ax_abs.legend(fontsize=9, loc='upper right')
    ax_abs.grid(True, alpha=0.3, linewidth=0.5)

    # stacked composition
    ax_pct.stackplot(years, [G_pct, F_pct, H_pct],
                      labels=['gather', 'fish', 'hunt'],
                      colors=['#2a9d8f', '#577590', '#e76f51'],
                      alpha=0.92, edgecolor='white', linewidth=0.3)
    ax_pct.set_xlabel('year')
    ax_pct.set_ylabel('fraction of forage-node mass')
    ax_pct.set_ylim(0, 1)
    ax_pct.legend(fontsize=9, loc='center right')
    ax_pct.grid(True, alpha=0.3, linewidth=0.5)
    ax_pct.set_title('composition (stacked fractions)', fontsize=10)
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


def plot_death_causes_inferred(out_path):
    """Death-cause counts aren't serialized in the dumps (only printed to
    stdout per-day). Reconstruct approximately from consecutive-dump
    diffs: agents that disappeared between dump N and dump N+1 are
    classified using their last-known state in dump N.

    Heuristics (with 180-day dump cadence and ~1 mortality event per
    agent over that interval):
      - **old_age**: age in dump N + 0.5y ≥ MAX_AGE_YEARS (would
        likely hit lifespan within the interval)
      - **starvation**: hunger in dump N ≥ 50000 (trending toward
        HUNGER_DEATH at typical accumulation rate)
      - **neglect/exposure**: kid (age < WATCH_AGE_YEARS) in dump N who
        died and wasn't old (kids ~never die of old age). Caveat: also
        catches some kid hazard deaths.
      - **hazard/other**: residual — healthy adults who vanished. In
        a real run most of these are HAZARD_PER_HOUR random deaths,
        but also includes deaths from late-window mortality causes
        that the snapshot-state heuristic misses.

    Plotted both as absolute per-180-day counts and as composition shares."""
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    if len(paths) < 2:
        return
    import main as m
    MAX_AGE = float(m.MAX_AGE_YEARS)
    WATCH_AGE = float(m.WATCH_AGE_YEARS)
    DAYS_PER_DUMP = float(m.DUMP_EVERY_DAYS)
    YEARS_PER_DUMP = DAYS_PER_DUMP / 365.0
    cats = ['old_age', 'starvation', 'neglect/exposure', 'hazard/other']
    cat_idx = {c: i for i, c in enumerate(cats)}
    results = []   # list of (year, dict cat→count, total_loss)
    prev = None; prev_year = None
    for p in paths:
        try:
            with open(p) as f: d = json.load(f)
        except json.JSONDecodeError:
            continue
        agents = d.get('agents', [])
        year = d.get('day', 0) / 365.0
        if not agents:
            prev = None
            continue
        ag_by_id = {a['state']['id']: a for a in agents}
        if prev is not None:
            disappeared = set(prev) - set(ag_by_id.keys())
            cnt = np.zeros(len(cats), dtype=np.int32)
            for aid in disappeared:
                ag = prev[aid]
                age = ag['state']['age']
                hng = ag['state']['hunger']
                if age + YEARS_PER_DUMP >= MAX_AGE:
                    cnt[cat_idx['old_age']] += 1
                elif hng >= 50000:
                    cnt[cat_idx['starvation']] += 1
                elif age < WATCH_AGE:
                    cnt[cat_idx['neglect/exposure']] += 1
                else:
                    cnt[cat_idx['hazard/other']] += 1
            results.append((year, cnt, int(cnt.sum()), len(prev)))
        prev = ag_by_id
        prev_year = year
    if not results:
        return

    years_arr = np.array([r[0] for r in results])
    counts_mat = np.stack([r[1] for r in results])    # (W, 4)
    prev_pops = np.array([r[3] for r in results], dtype=np.float64)

    fig, (ax_abs, ax_per, ax_pct) = plt.subplots(3, 1, figsize=(11, 11),
                                                   sharex=True,
                                                   gridspec_kw={'hspace': 0.18})
    colors = ['#577590', '#e76f51', '#9b6f47', '#888888']
    # Top: absolute losses per interval
    ax_abs.stackplot(years_arr, counts_mat.T, labels=cats, colors=colors,
                      alpha=0.92, edgecolor='white', linewidth=0.3)
    ax_abs.set_ylabel('inferred deaths per 180-day interval')
    ax_abs.set_title(
        f'Inferred death-cause composition over time — '
        f'{len(results)} intervals, day {int(years_arr[-1]*365)}\n'
        f'reconstructed from dump-to-dump agent-id diffs + last-known state',
        fontsize=11)
    ax_abs.legend(loc='upper left', fontsize=9)
    ax_abs.grid(True, alpha=0.3, linewidth=0.5)

    # Middle: per-capita rate (deaths per 100 agents over the 180-day interval)
    per_cap = (counts_mat / np.maximum(prev_pops[:, None], 1.0)) * 100
    for ci, c in enumerate(cats):
        ax_per.plot(years_arr, per_cap[:, ci], color=colors[ci], lw=1.7,
                     label=c)
    ax_per.set_ylabel('deaths per 100 agents\nper 180-day interval')
    ax_per.legend(loc='upper left', fontsize=9)
    ax_per.grid(True, alpha=0.3, linewidth=0.5)
    ax_per.set_title('per-capita mortality by cause', fontsize=10)

    # Bottom: composition fractions
    totals = counts_mat.sum(axis=1, keepdims=True)
    fracs = counts_mat / np.maximum(totals, 1)
    ax_pct.stackplot(years_arr, fracs.T, labels=cats, colors=colors,
                      alpha=0.92, edgecolor='white', linewidth=0.3)
    ax_pct.set_xlabel('year')
    ax_pct.set_ylabel('fraction of inferred deaths')
    ax_pct.set_ylim(0, 1)
    ax_pct.set_title('composition (stacked)', fontsize=10)
    ax_pct.grid(True, alpha=0.3, linewidth=0.5)
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  wrote {out_path}')


def plot_forage_firing_over_time(out_path, sample_n=120):
    """Behavioral counterpart to plot_forage_over_time. Replays each sampled
    agent's 24-hour chain walk against their dumped state and counts how
    many times hunt / fish / gather ACTUALLY FIRE per agent per day. This
    is the activity-level view — sensitive to chain-walk-trajectory shifts
    that happen *before* graph composition catches up via subgraph
    adoption. Sees behavioral regime shifts in the same hour they occur.

    State is held frozen during the replay (no hunger/cache evolution), so
    counts are 'what each agent would fire at each time-of-day given their
    current state'. Time- and age-gated forage gates are accurate; the
    state-dependent components (e.g. forage gating that includes pregnancy)
    are at the snapshot value."""
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    years = []
    hunt_per = []; fish_per = []; gath_per = []
    rng = np.random.default_rng(0)
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        agents = d.get('agents', [])
        if not agents:
            continue
        if sample_n and sample_n < len(agents):
            sample = [agents[i] for i in rng.choice(len(agents), sample_n, replace=False)]
        else:
            sample = agents
        h = f_ = g = 0; n_rep = 0
        for a in sample:
            c = _replay_agent_day(a)
            if c is None: continue
            h  += int(c[:, ACTION_ORDER.index('agent_hunt')].sum())
            f_ += int(c[:, ACTION_ORDER.index('agent_fish')].sum())
            g  += int(c[:, ACTION_ORDER.index('agent_gather')].sum())
            n_rep += 1
        if n_rep == 0:
            continue
        years.append(d.get('day', 0) / 365.0)
        hunt_per.append(h / n_rep)
        fish_per.append(f_ / n_rep)
        gath_per.append(g / n_rep)
    if not years:
        return
    H = np.array(hunt_per); F = np.array(fish_per); G = np.array(gath_per)
    tot = H + F + G
    safe = np.maximum(tot, 1e-9)

    fig, (ax_abs, ax_pct) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                          gridspec_kw={'hspace': 0.12})
    ax_abs.plot(years, G, color='#2a9d8f', lw=2.0, label='gather fires')
    ax_abs.plot(years, F, color='#577590', lw=1.6, label='fish fires')
    ax_abs.plot(years, H, color='#e76f51', lw=1.6, label='hunt fires')
    ax_abs.plot(years, tot, color='black', lw=0.8, linestyle=':',
                 label='total forage fires', alpha=0.5)
    ax_abs.set_ylabel('action fires / agent / day\n(replayed 24-hr walk)')
    ax_abs.set_title(
        f'Forage *firing rate* over time — {len(years)} dumps, '
        f'day {int(years[-1]*365)}\n'
        f'replayed 24-hr chain walk; behaviorally what agents are doing, '
        f'not what their graphs contain',
        fontsize=11)
    ax_abs.legend(fontsize=9, loc='upper right')
    ax_abs.grid(True, alpha=0.3, linewidth=0.5)

    ax_pct.stackplot(years, [G/safe, F/safe, H/safe],
                      labels=['gather', 'fish', 'hunt'],
                      colors=['#2a9d8f', '#577590', '#e76f51'],
                      alpha=0.92, edgecolor='white', linewidth=0.3)
    ax_pct.set_xlabel('year')
    ax_pct.set_ylabel('fraction of forage-fire mass')
    ax_pct.set_ylim(0, 1)
    ax_pct.legend(fontsize=9, loc='center right')
    ax_pct.grid(True, alpha=0.3, linewidth=0.5)
    ax_pct.set_title('composition (stacked fractions)', fontsize=10)
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


def plot_mating_sex_composition(out_path):
    """For every weekly dump, classify each mating_pairs entry by the sex of
    its two participants (MM, MF, FF) and weight by the accumulated value
    (mating-matrix entry, decay-weighted recency). Plot the per-capita rate
    of each class over time. Heterosexual MF is the only reproductive class;
    MM and FF capture pure social pair-bonds (homosocial mating events
    update the mating_matrix without producing offspring).

    mating_pairs is upper-triangle only — each unique pair counted once."""
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    years, mm, mf, ff, total = [], [], [], [], []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        agents = d.get('agents', [])
        if not agents:
            continue
        n = len(agents)
        sex = np.array([a['state']['is_female'] for a in agents], dtype=bool)
        s_mm = s_mf = s_ff = 0.0
        for i, j, v in d.get('mating_pairs', []):
            if i >= n or j >= n:
                continue
            fi, fj = sex[i], sex[j]
            if not fi and not fj:
                s_mm += v
            elif fi and fj:
                s_ff += v
            else:
                s_mf += v
        years.append(d.get('day', 0) / 365.0)
        mm.append(s_mm / n)
        mf.append(s_mf / n)
        ff.append(s_ff / n)
        total.append((s_mm + s_mf + s_ff) / n)
    if not years:
        return
    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                                          gridspec_kw={'hspace': 0.12})
    # Top: absolute per-capita rates
    ax_abs.plot(years, mf, color='#9b6f47', lw=2.0, label='M-F (reproductive)')
    ax_abs.plot(years, mm, color='#1b3a6b', lw=1.4, label='M-M (homosocial)')
    ax_abs.plot(years, ff, color='#e76f51', lw=1.4, label='F-F (homosocial)')
    ax_abs.plot(years, total, color='gray', lw=1.0, linestyle=':',
                 label='total')
    ax_abs.set_ylabel('mating-matrix mass per agent\n(decay-weighted)')
    ax_abs.set_title(
        f'Mating-pair sex composition over time — '
        f'{len(years)} dumps, day {int(years[-1]*365)}\n'
        f'mating_matrix entries classified by participants\' sex, '
        f'summed and divided by pop',
        fontsize=11)
    ax_abs.legend(fontsize=9, loc='upper right')
    ax_abs.grid(True, alpha=0.3, linewidth=0.5)

    # Bottom: composition fractions
    mm_a = np.array(mm); mf_a = np.array(mf); ff_a = np.array(ff)
    tot_a = mm_a + mf_a + ff_a
    safe = np.maximum(tot_a, 1e-9)
    ax_rel.stackplot(years, [mf_a/safe, mm_a/safe, ff_a/safe],
                      labels=['M-F', 'M-M', 'F-F'],
                      colors=['#9b6f47', '#1b3a6b', '#e76f51'], alpha=0.92,
                      edgecolor='white', linewidth=0.4)
    ax_rel.set_xlabel('year')
    ax_rel.set_ylabel('fraction of mating-pair mass')
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(fontsize=9, loc='center right')
    ax_rel.grid(True, alpha=0.3, linewidth=0.5)
    ax_rel.set_title('composition (stacked fractions)', fontsize=10)

    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# 10. Population + mean hunger trajectory (across all dumps)
# ---------------------------------------------------------------------------

def plot_population_trajectory(out_path):
    """Pop on the left y-axis, mean hunger on the right. Reads every dump."""
    paths = sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
    years, pops, hungers, fertFs = [], [], [], []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        ag = d['agents']
        if not ag:
            continue
        s = [a['state'] for a in ag]
        years.append(d['day'] / 365.0)
        pops.append(len(ag))
        hungers.append(float(np.mean([x['hunger'] for x in s])))
        age = np.array([x['age'] for x in s])
        fem = np.array([x['is_female'] for x in s])
        meno = np.array([x.get('is_menopausal', False) for x in s])
        fertFs.append(int(((age >= 16) & fem & ~meno).sum()))
    if not years:
        print("  (no dumps to plot)")
        return
    years = np.array(years); pops = np.array(pops)
    hungers = np.array(hungers); fertFs = np.array(fertFs)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(years, pops, color='#3a6ea5', lw=2.0, label='population')
    ax1.fill_between(years, pops, alpha=0.18, color='#3a6ea5')
    ax1.plot(years, fertFs, color='#e94560', lw=1.4, ls='--',
             label='fertile females')
    ax1.set_xlabel('sim year')
    ax1.set_ylabel('count', color='#3a6ea5')
    ax1.tick_params(axis='y', labelcolor='#3a6ea5')
    ax1.set_ylim(0, max(pops.max() * 1.05, 10))
    ax1.spines['top'].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(years, hungers, color='#f5b942', lw=1.4, label='mean hunger')
    ax2.set_ylabel('mean hunger (kcal accumulated)', color='#b88a1a')
    ax2.tick_params(axis='y', labelcolor='#b88a1a')
    ax2.spines['top'].set_visible(False)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9, framealpha=0.95)

    ax1.set_title(f"Population, fertile females, and mean hunger over time — "
                   f"{len(years)} weekly dumps (yr {years.min():.1f}–{years.max():.1f})",
                   fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('week', nargs='?', help='e.g. "week_18" — defaults to latest')
    p.add_argument('-o', '--out', default=None,
                    help='output dir (default: viz/<week>)')
    p.add_argument('--threshold', type=float, default=0.6,
                    help='clan single-link cos threshold (default 0.6)')
    p.add_argument('--gif', action='store_true',
                    help='build viz/grid.gif from all daily dumps and exit')
    p.add_argument('--fps', type=int, default=8, help='gif fps (default 8)')
    p.add_argument('--sample-every', type=int, default=1,
                    help='use every Nth dump as a frame (default 1)')
    args = p.parse_args()
    if args.gif:
        os.makedirs('viz', exist_ok=True)
        make_grid_animation('viz/grid.gif', fps=args.fps,
                             sample_every=args.sample_every)
        make_culture_animation('viz/culture.gif', fps=args.fps,
                                sample_every=args.sample_every)
        return

    if args.week:
        path = next((p for p in sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=_week_num)
                     if args.week in p), None)
        if not path:
            raise SystemExit(f"no dump matching {args.week}")
    else:
        path = latest_dump()

    print(f"loading {path}")
    a = load_arrays(path)
    week = os.path.basename(path).replace('.json', '')
    out_dir = args.out or f'viz/{week}'
    os.makedirs(out_dir, exist_ok=True)

    plot_family_clusters(a, f'{out_dir}/family_clusters.png',
                          threshold=args.threshold)
    plot_cohort_cohesion(a, f'{out_dir}/cohort_cohesion.png')
    plot_population_pyramid(a, f'{out_dir}/population_pyramid.png')
    plot_resource_distributions(a, f'{out_dir}/resource_distributions.png')
    plot_weight_heatmap(a, f'{out_dir}/weight_heatmap.png')
    # action-by-sex needs the raw dump JSON for gift_pairs / mating_pairs
    with open(path) as f:
        dump = json.load(f)
    plot_action_by_sex(a, dump, f'{out_dir}/action_by_sex.png')
    plot_camp_density(a, dump, f'{out_dir}/camp_density.png')
    plot_time_of_day_actions(dump, f'{out_dir}/time_of_day_actions.png')
    plot_time_of_day_by_camp(dump, f'{out_dir}/time_of_day_by_camp.png')
    plot_camp_encounter_distribution(dump, f'{out_dir}/camp_encounters.png')
    plot_ties_by_sex_age(dump, f'{out_dir}/ties_by_sex_age.png')
    # pop trajectory spans all dumps; save it once at the top of the viz dir
    plot_population_trajectory('viz/population_trajectory.png')
    plot_mating_sex_composition('viz/mating_sex_composition.png')
    plot_forage_over_time('viz/forage_over_time.png')
    plot_forage_firing_over_time('viz/forage_firing_over_time.png')
    plot_death_causes_inferred('viz/death_causes_inferred.png')
    print(f"done. yr {a['year']:.1f}, pop {a['N']}")


if __name__ == '__main__':
    main()
