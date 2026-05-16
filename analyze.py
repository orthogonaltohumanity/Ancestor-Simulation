"""General analysis script for anthrosim dumps.

Usage:
  python analyze.py                    # summary + latest detail
  python analyze.py week_05            # focus on a specific dump
  python analyze.py --traj             # population trajectory only
  python analyze.py --agent 42         # focus on a single agent in latest dump
  python analyze.py --culture          # cohort weights + cohesion
  python analyze.py --graphs           # graph evolution stats
  python analyze.py --death            # diagnose likely death modes from state
  python analyze.py --top              # most-liked / most-hated agents
"""
import json
import glob
import sys
import collections
import numpy as np

DUMP_DIR = '/mnt/anthrosim/dumps'
FN = ('net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
      'age', 'is_pregnant', 'is_menopausal',
      'matings', 'gifts', 'family', 'camp$')

# ------------------- loading helpers -------------------

def list_dumps():
    # sort by week number, not lexicographic, so week_100 doesn't slot
    # between week_10 and week_11.
    def week_num(p):
        try: return int(p.rsplit('week_', 1)[1].split('.')[0])
        except Exception: return 0
    return sorted(glob.glob(f'{DUMP_DIR}/week_*.json'), key=week_num)

def load(path):
    with open(path) as f:
        return json.load(f)

def load_arrays(d):
    """Vectorize agent state into numpy arrays for analysis."""
    ag = d['agents']
    if not ag:
        return None
    s = [a['state'] for a in ag]
    return {
        'agents': ag,
        'states': s,
        'N': len(ag),
        'day': d['day'],
        'year': d['day'] / 365,
        'age': np.array([x['age'] for x in s]),
        'fem': np.array([x['is_female'] for x in s]),
        'meno': np.array([x['is_menopausal'] for x in s]),
        'preg': np.array([x['is_pregnant'] for x in s]),
        'sleep': np.array([x['sleep'] for x in s]),
        'in_camp': np.array([x['is_in_camp'] for x in s]),
        'hunger': np.array([x['hunger'] for x in s]),
        'cache': np.array([x['cache'] for x in s]),
        'tired': np.array([x['tired'] for x in s]),
        'ndf': np.array([x['net_debt_flow'] for x in s]),
        'weights': np.array([x['weights'] for x in s], dtype=np.float32),
        'Q': np.array([x['Q'] for x in s], dtype=np.float32),
        'adopt_rate': np.array([x.get('adopt_rate', 0) for x in s]),
        'mut_rate': np.array([x.get('mut_rate', 0) for x in s]),
        'fslp': np.array([x.get('forced_sleep_hours', 0) for x in s]),
        'unwatched': np.array([x.get('unwatched_hours', 0) for x in s]),
        'n_nodes': np.array([len(a['graph']['nodes']) for a in ag]),
    }

# ------------------- analyses -------------------

def trajectory():
    paths = list_dumps()
    if not paths:
        print("no dumps yet"); return
    print(f"{'file':<14} {'yr':>5} {'pop':>5} {'kids<13':>7} {'preg':>4} {'fertF':>5} "
          f"{'oc':>4} {'mh':>6} {'mc':>5} {'w_giv':>6} {'adp':>5} {'mut':>9} {'nodes':>5}")
    for p in paths:
        try:
            d = load(p)
        except json.JSONDecodeError:
            print(f"{p.split('/')[-1]:<14} (mid-write — skip)"); continue
        a = load_arrays(d)
        if a is None:
            print(f"{p.split('/')[-1]:<14} extinct"); continue
        fert_f = int(((a['age'] >= 16) & a['fem'] & ~a['meno']).sum())
        print(f"{p.split('/')[-1]:<14} {a['year']:>5.1f} {a['N']:>5} "
              f"{int((a['age']<13).sum()):>7} {int(a['preg'].sum()):>4} {fert_f:>5} "
              f"{int((~a['in_camp']).sum()):>4} {a['hunger'].mean():>6.0f} {a['cache'].mean():>5.0f} "
              f"{a['weights'][:,0].mean():>+6.2f} {a['adopt_rate'].mean():>5.2f} "
              f"{a['mut_rate'].mean():>9.2e} {a['n_nodes'].mean():>5.0f}")

def demographics(a):
    print(f"\n=== Demographics (pop {a['N']}, year {a['year']:.1f}) ===")
    print(f"age:  mean={a['age'].mean():.1f}  median={np.median(a['age']):.1f}  max={a['age'].max():.1f}")
    print(f"females {int(a['fem'].sum())}  meno {int(a['meno'].sum())}  preg {int(a['preg'].sum())}  "
          f"fertF {int(((a['age']>=16) & a['fem'] & ~a['meno']).sum())}")
    print(f"in camp {int(a['in_camp'].sum())}/{a['N']}  forced sleep {int((a['fslp']>0).sum())}")
    print(f"hunger mean={a['hunger'].mean():.0f}  cache mean={a['cache'].mean():.0f}  "
          f"ndf mean={a['ndf'].mean():+.0f}")

    print("\nAge histogram:")
    hist, edges = np.histogram(a['age'], bins=[0,6,13,16,25,40,60,120])
    bar_scale = max(1, max(hist)/40)
    for i, c in enumerate(hist):
        bar = '#' * int(c / bar_scale)
        print(f"  [{edges[i]:>3.0f}-{edges[i+1]:>3.0f}): {bar} {c}")

def culture(a):
    print(f"\n=== Cohort weights ===")
    print(f"  {'cohort':<14} {'n':>4} {'F':>3}  " + ' '.join(f[:6].rjust(6) for f in FN))
    cohorts = [("kids 0-6", 0, 6), ("forager 6-13", 6, 13),
               ("teens 13-16", 13, 16), ("young 16-30", 16, 30),
               ("middle 30-50", 30, 50), ("elders 50+", 50, 999)]
    cohort_means = []
    for label, lo, hi in cohorts:
        m = (a['age'] >= lo) & (a['age'] < hi)
        if not m.any(): continue
        nF = int((m & a['fem']).sum())
        means = a['weights'][m].mean(axis=0)
        cohort_means.append((label, means))
        print(f"  {label:<14} {int(m.sum()):>4} {nF:>3}  "
              + ' '.join(f"{v:+.2f}".rjust(6) for v in means))

    # cross-cohort cosine similarity
    if len(cohort_means) >= 2:
        print("\n=== Cross-cohort cultural distance (cosine sim of cohort means) ===")
        labels = [m[0].split()[0][:8] for m in cohort_means]
        M = np.array([m[1] for m in cohort_means])
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        cross = Mn @ Mn.T
        print(f"  {'':<10} " + ' '.join(l.rjust(8) for l in labels))
        for i, l in enumerate(labels):
            row = ' '.join(f"{cross[i,j]:+.2f}".rjust(8) for j in range(len(labels)))
            print(f"  {l:<10} {row}")

    # population-wide cohesion
    wn = a['weights'] / (np.linalg.norm(a['weights'], axis=1, keepdims=True) + 1e-9)
    upper = (wn @ wn.T)[np.triu_indices(a['N'], k=1)]
    print(f"\nPop-wide cohesion (cosine sim): mean={upper.mean():+.3f}  std={upper.std():.3f}")

def graphs_stat(a):
    print(f"\n=== Graph evolution ===")
    n = a['n_nodes']
    print(f"n_nodes: mean={n.mean():.1f}  median={np.median(n):.0f}  "
          f"min={n.min()}  max={n.max()}")
    print(f"agents at canonical 33: {int((n == 33).sum())}")
    print(f"agents > 100: {int((n > 100).sum())}")
    print(f"agents at cap (>= 300): {int((n >= 300).sum())}")

    ta = collections.Counter()
    chunks = collections.Counter()
    for ag_data in a['agents']:
        for nd in ag_data['graph']['nodes']:
            ta[nd['true_action']] += 1
            if nd['seq']:
                for c in nd['seq']['chunks']:
                    v_ = round(c['value'], 1) if isinstance(c['value'], float) else c['value']
                    chunks[(c['var'], c['op'], v_)] += 1
    print(f"\nTop true_actions:")
    for act, c in ta.most_common(8):
        print(f"  {c:>5}× {act}")
    print(f"\nDistinct chunk signatures: {len(chunks)}  (canonical=90 at burn-in)")
    print(f"Top 6 chunks:")
    for (var, op, val), c in chunks.most_common(6):
        print(f"  {c:>6}× {var:>14} {op or '?':>3} {val}")

def adaptive_rates(a):
    print(f"\n=== Adaptive rates ===")
    print(f"adopt: mean={a['adopt_rate'].mean():.3f}  >0.5: {int((a['adopt_rate']>0.5).sum())}  "
          f"max={a['adopt_rate'].max():.3f}")
    print(f"mut:   mean={a['mut_rate'].mean():.2e}  max={a['mut_rate'].max():.2e}  "
          f">10x floor: {int((a['mut_rate']>3.8e-4).sum())}")

def death_diagnose(a):
    print(f"\n=== Death-mode diagnostic (state-based inference) ===")
    print(f"Population stress signals at this snapshot:")
    print(f"  starving (hunger > 60k):     {int((a['hunger']>60000).sum())}")
    print(f"  near death (hunger > 100k):  {int((a['hunger']>100000).sum())}  (death @ 120k)")
    print(f"  outside camp:                {int((~a['in_camp']).sum())}")
    print(f"  forced sleep right now:      {int((a['fslp']>0).sum())}")
    print(f"  forced sleep + outside:      {int((a['fslp']>0) & ~a['in_camp']).sum() if False else int(((a['fslp']>0) & ~a['in_camp']).sum())}  (active exposure risk)")
    is_kid = a['age'] < 13
    print(f"  kids unwatched > limit:      {int((is_kid & (a['unwatched']>3)).sum())}")
    print(f"  ages > 100:                  {int((a['age']>100).sum())}  (lifespan cap 120)")

def social_score_matrix(a):
    """Compute popularity = mean over observers of social_score(observer, target)."""
    # Mirror ref.FEATURE_NORM
    FN_NORM = np.array([1/10000, 1/72000, 1, 1, 1/10000, 1/60, 1, 1], dtype=np.float32)
    F = np.zeros((a['N'], 8), dtype=np.float32)
    F[:,0] = a['ndf']    * FN_NORM[0]
    F[:,1] = a['hunger'] * FN_NORM[1]
    F[:,2] = a['tired']  * FN_NORM[2]
    F[:,3] = np.where(a['fem'],  1.0, -1.0) * FN_NORM[3]
    F[:,4] = a['cache']  * FN_NORM[4]
    F[:,5] = a['age']    * FN_NORM[5]
    F[:,6] = np.where(a['preg'], 1.0, -1.0) * FN_NORM[6]
    F[:,7] = np.where(a['meno'], 1.0, -1.0) * FN_NORM[7]
    linear = a['weights'] @ F.T
    quad = np.einsum('nfg,mf,mg->nm', a['Q'], F, F)
    return (linear + quad).mean(axis=0)

def top_agents(a, k=3):
    pop = social_score_matrix(a)
    print(f"\n=== Most liked / most hated (peer-mean social score) ===")
    print(f"Distribution: min={pop.min():+.3f}  median={np.median(pop):+.3f}  max={pop.max():+.3f}")
    def show(idx, label):
        s = a['states'][idx]
        w = a['weights'][idx]
        sex = 'F' if s['is_female'] else 'M'
        flags = []
        if s['is_pregnant']: flags.append('preg')
        if s['is_menopausal']: flags.append('meno')
        if s['sleep']: flags.append('asleep')
        if not s['is_in_camp']: flags.append('OUT')
        f_str = ' | '.join(flags) if flags else ''
        print(f"  [{label}] #{idx} {sex} age {s['age']:.1f} {f_str}")
        print(f"    pop={pop[idx]:+.3f}  ndf={s['net_debt_flow']:+,.0f}  "
              f"cache={s['cache']:.0f}  hunger={s['hunger']:.0f}")
        order = np.argsort(-np.abs(w))[:3]
        for j in order:
            sign = '♥' if w[j] > 0 else '✗'
            print(f"    {sign} {FN[j]:>14}  {w[j]:+.2f}")
    print("\nMost liked:")
    for r, i in enumerate(np.argsort(-pop)[:k]):
        show(int(i), f"#{r+1}")
    print("\nMost hated:")
    for r, i in enumerate(np.argsort(pop)[:k]):
        show(int(i), f"#{r+1}")

def focus_agent(a, idx):
    print(f"\n=== Agent #{idx} ===")
    s = a['states'][idx]
    w = a['weights'][idx]
    g = a['agents'][idx]['graph']
    sex = 'F' if s['is_female'] else 'M'
    print(f"{sex}, age {s['age']:.1f}")
    print(f"hunger={s['hunger']:.0f}  cache={s['cache']:.0f}  tired={s['tired']:.2f}  "
          f"ndf={s['net_debt_flow']:+,.0f}")
    print(f"is_in_camp={s['is_in_camp']}  sleep={s['sleep']}  "
          f"preg={s['is_pregnant']}  meno={s['is_menopausal']}")
    print(f"adopt_rate={s.get('adopt_rate',0):.3f}  mut_rate={s.get('mut_rate',0):.2e}")
    print(f"\nLinear weights (sorted by |magnitude|):")
    for j in np.argsort(-np.abs(w)):
        sign = '♥' if w[j] > 0 else '✗'
        print(f"  {sign} {FN[j]:>16}  {w[j]:+.3f}")
    print(f"\nGraph: {len(g['nodes'])} nodes")
    ta = collections.Counter()
    for n in g['nodes']:
        ta[n['true_action']] += 1
    print(f"Top actions in their graph:")
    for act, c in ta.most_common(6):
        print(f"  {c}× {act}")

# ------------------- main dispatch -------------------

def main():
    args = sys.argv[1:]
    if not args:
        # default: trajectory + latest detail
        trajectory()
        paths = list_dumps()
        if not paths: return
        for p in reversed(paths):
            try:
                d = load(p); break
            except json.JSONDecodeError:
                continue
        else:
            return
        a = load_arrays(d)
        if a is None:
            print(f"\n{paths[-1].split('/')[-1]} is extinct"); return
        demographics(a)
        culture(a)
        graphs_stat(a)
        adaptive_rates(a)
        return

    flag = args[0]
    if flag == '--traj':
        trajectory(); return

    paths = list_dumps()
    if not paths: print("no dumps"); return

    # specific dump?
    if flag.startswith('week_'):
        target = next((p for p in paths if flag in p), None)
        if not target:
            print(f"no dump matching {flag}"); return
        a = load_arrays(load(target))
        demographics(a); culture(a); graphs_stat(a); adaptive_rates(a)
        return

    # latest dump for the rest
    a = None
    for p in reversed(paths):
        try:
            a = load_arrays(load(p)); break
        except json.JSONDecodeError:
            continue
    if a is None:
        print("no readable dump"); return
    print(f"\n=== Latest dump: year {a['year']:.1f} pop {a['N']} ===")

    if flag == '--culture':    culture(a)
    elif flag == '--graphs':   graphs_stat(a)
    elif flag == '--death':    death_diagnose(a)
    elif flag == '--top':      top_agents(a)
    elif flag == '--agent':
        idx = int(args[1])
        focus_agent(a, idx)
    else:
        print(f"unknown flag {flag}; see top of file for usage")

if __name__ == '__main__':
    main()
