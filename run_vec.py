"""Vectorized simulation driver — counterpart to main.py's `for t in range(T)` loop.

Object graphs remain the source of truth for graph structure (so we can reuse
ref.mutate, ref.assimilate_node, ref.clone_graph). Flat arrays are derived;
re-flattened per agent on any graph change. Scalar agent state lives in flat
arrays only.
"""
import json
import math
import os
import random
import time
import argparse
import numpy as np
from tqdm import tqdm

import main as ref
import main_vec as v


def _to_host_state(state):
    """Best-effort: return a dict of numpy views of `state` regardless of
    whether it's CPU- or GPU-backed. Used at JSON serialization points."""
    return {k: v.to_host(arr) for k, arr in state.items()}


def serialize_agent_from_flat(i, state, weights, Q, agents=None):
    """Build a state dict matching ref.serialize_agent's format from flat arrays."""
    return {
        'id': int(agents[i].id) if agents is not None else int(i),
        'hunger': float(state['hunger'][i]),
        'cache':  float(state['cache'][i]),
        'tired':  float(state['tired'][i]),
        'is_female': bool(state['is_female'][i]),
        'is_in_camp': bool(state['is_in_camp'][i]),
        'sleep': bool(state['sleep'][i]),
        'consecutive_sleep': int(state['consecutive_sleep'][i]),
        'net_debt_flow': float(state['net_debt_flow'][i]),
        'weights': weights[i].tolist(),
        'Q': Q[i].tolist(),
        'age': float(state['age'][i]),
        'is_pregnant': bool(state['is_pregnant'][i]),
        'pregnancy_hours': int(state['pregnancy_hours'][i]),
        'is_menopausal': bool(state['is_menopausal'][i]),
        # vec-only fields (extra info; ignored by code that targets ref's format)
        'mut_rate': float(state['mut_rate'][i]),
        'adopt_rate': float(state['adopt_rate'][i]),
        'unwatched_hours': int(state['unwatched_hours'][i]),
        'camp_id': int(state['camp_id'][i]),
        'x': int(state['x'][i]),
        'y': int(state['y'][i]),
    }


def dump_population(path, day, state, weights, Q, roots,
                     mating_matrix=None, gift_matrix=None, crests=None,
                     camp_caches=None, resources=None, agents=None,
                     camp_positions=None):
    """Write a JSON dump compatible with ref.dump_population's format.
    mating_matrix saved sparsely (upper-triangle, threshold > 0.01).
    gift_matrix is directional — saved as full (i, j, amount) triples for
    every nonzero entry above threshold."""
    N = len(roots)
    # Download GPU-backed state to host for JSON serialization. Cheap because
    # dumps happen every 360 sim days (~3 times per run).
    state = _to_host_state(state)
    weights = v.to_host(weights)
    Q = v.to_host(Q)
    if mating_matrix is not None:
        mating_matrix = v.to_host(mating_matrix)
    if gift_matrix is not None:
        gift_matrix = v.to_host(gift_matrix)
    if crests is not None:
        crests = v.to_host(crests)
    payload = {
        'day': day,
        'n_agents': N,
        'agents': [
            {'state': serialize_agent_from_flat(i, state, weights, Q, agents=agents),
             'graph': ref.serialize_graph(roots[i])}
            for i in range(N)
        ],
    }
    if mating_matrix is not None:
        upper = np.triu(mating_matrix, k=1)
        ii, jj = np.nonzero(upper > 0.01)
        payload['mating_pairs'] = [[int(i), int(j), float(upper[i, j])]
                                    for i, j in zip(ii, jj)]
    if gift_matrix is not None:
        # directional: keep both [i,j] and [j,i] when nonzero (no symmetry)
        ii, jj = np.nonzero(gift_matrix > 0.01)
        payload['gift_pairs'] = [[int(i), int(j), float(gift_matrix[i, j])]
                                  for i, j in zip(ii, jj)]
    if crests is not None:
        # save crests per agent (parallel to agents list)
        for i, ag in enumerate(payload['agents']):
            ag['state']['crest'] = [float(x) for x in crests[i]]
    if camp_caches is not None:
        if camp_positions is not None:
            payload['camps'] = [{'id': int(c), 'cache': float(camp_caches[c]),
                                  'x': int(camp_positions.get(c, (0, 0))[0]),
                                  'y': int(camp_positions.get(c, (0, 0))[1])}
                                 for c in camp_caches.keys()]
        else:
            payload['camps'] = [{'id': int(c), 'cache': float(v)}
                                 for c, v in camp_caches.items()]
    if resources is not None:
        if isinstance(resources, dict):
            payload['resources'] = {k: float(v) for k, v in resources.items()}
        else:
            # numpy (W, H, 3) array — full grid for per-cell analysis
            payload['resources_grid'] = np.asarray(resources).tolist()
    with open(path, 'w') as f:
        json.dump(payload, f)


N_AGENTS = ref.N_AGENTS
SIM_DAYS = ref.SIM_DAYS
DUMP_DIR = ref.DUMP_DIR
DUMP_EVERY_DAYS = ref.DUMP_EVERY_DAYS
BURN_IN_ASEX_CYCLES = ref.BURN_IN_ASEX_CYCLES
BURN_IN_TEST_HOURS = ref.BURN_IN_TEST_HOURS
BURN_IN_HUNGER_LIMIT = ref.BURN_IN_HUNGER_LIMIT
BURN_IN_MIN_CAMP_FRAC = ref.BURN_IN_MIN_CAMP_FRAC
BURN_IN_SEXUAL = ref.BURN_IN_SEXUAL
BURN_IN_CACHE_DIR = ref.BURN_IN_CACHE_DIR


def _burnin_cache_path():
    """Cache filename keyed by all parameters that affect the asexual burn-in result."""
    return os.path.join(
        BURN_IN_CACHE_DIR,
        f'asex_N{ref.N_AGENTS}'
        f'_C{BURN_IN_ASEX_CYCLES}'
        f'_H{BURN_IN_TEST_HOURS}'
        f'_HL{int(BURN_IN_HUNGER_LIMIT)}'
        f'_CF{int(BURN_IN_MIN_CAMP_FRAC*100)}'
        f'.pkl'
    )


def _save_burnin_cache(path, agents, roots):
    """Pickle (agents, roots) so the burn-in can be reused next run."""
    import pickle
    os.makedirs(BURN_IN_CACHE_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'agents': agents, 'roots': roots}, f)


def _load_burnin_cache(path):
    """Load pickled (agents, roots) from a previous run's burn-in."""
    import pickle
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['agents'], d['roots']


def _clone_agent_into(dst_idx, src_idx, agents, roots):
    """Replace agents[dst_idx] / roots[dst_idx] with a clone of src_idx's
    weights, Q, graph. Sex is sampled fresh (not inherited) — preserves
    50/50 distribution against drift-to-fixation in small populations."""
    src = agents[src_idx]
    new_ag = ref.agent()   # is_female randomized via random.random() < 0.5 in __init__
    new_ag.weights = list(src.weights)
    new_ag.Q = [list(row) for row in src.Q]
    agents[dst_idx] = new_ag
    roots[dst_idx] = ref.clone_graph(roots[src_idx])


def _solo_step_persistent(test_ag, root, hour, cur_node):
    """One hour of solo walking with persistent program counter.
    Mirrors main_vec.step_all semantics for a single agent."""
    n = cur_node
    visited = 0
    visit_cost = ref.NODE_VISIT_HUNGER * ref.metabolic_scale(test_ag.age)
    while n is not None and visited < ref.MAX_ACTION:
        visited += 1
        test_ag.hunger += visit_cost
        cond = ref.eval_seq(n.seq, test_ag, hour)
        action = n.true_action if cond else n.false_action
        if action is not None:
            action(test_ag)
        if action is not ref.agent_sleep:
            test_ag.sleep = False
            test_ag.consecutive_sleep = 0
        n = n.true_node if cond else n.false_node
    return n if n is not None else root


def _starvation_test_72h(root, test_ag, hours=BURN_IN_TEST_HOURS,
                          hunger_limit=BURN_IN_HUNGER_LIMIT,
                          min_camp_frac=BURN_IN_MIN_CAMP_FRAC):
    """Run a solo walk for `hours` hours. PASSES iff:
       - hunger stays at or below `hunger_limit` for the entire test, AND
       - at least `min_camp_frac` of hours are spent in camp.
    Returns peak hunger if both pass; on failure returns hunger_limit + 1
    (so the caller's `> hunger_limit` cull check fires).
    Hunger fail-fast: returns immediately on the first overflow hour.
    Camp-fraction check is evaluated at the end of the test."""
    test_ag.hunger = 0.0
    test_ag.cache = ref.CACHE_LIMIT
    test_ag.tired = 0.0
    test_ag.age = 18.0
    test_ag.is_in_camp = True
    test_ag.sleep = False
    test_ag.consecutive_sleep = 0
    test_ag.is_pregnant = False
    test_ag._is_menopausal = False
    test_ag._forced_sleep_hours = 0
    test_ag._talk_request = False
    test_ag._listen_request = False
    test_ag._mate_request = False
    test_ag._is_watching = False
    test_ag.net_debt_flow = 0.0
    test_ag.season = 0.5

    cur_node = root
    metab = ref.NODE_VISIT_HUNGER * ref.MAX_ACTION * ref.metabolic_scale(test_ag.age)
    peak_hunger = 0.0
    in_camp_hours = 0
    for t in range(hours):
        hour = t % 24
        # passive drift
        test_ag.tired += ref.TIRED_PER_HOUR
        test_ag.cache *= ref.CACHE_DECAY
        test_ag.season = ref.season(t)
        # walk (skip if in forced sleep)
        if test_ag._forced_sleep_hours == 0:
            cur_node = _solo_step_persistent(test_ag, root, hour, cur_node)
        # forced sleep trigger + override
        if test_ag._forced_sleep_hours == 0 and test_ag.tired >= ref.TIRED_DEATH:
            test_ag._forced_sleep_hours = 12
        if test_ag._forced_sleep_hours > 0:
            test_ag.sleep = True
            test_ag.consecutive_sleep += 1
            test_ag.tired = 0.0
            test_ag.hunger += metab
            test_ag._forced_sleep_hours -= 1
        # track in-camp time for sociality requirement
        if test_ag.is_in_camp:
            in_camp_hours += 1
        # fail-fast on hunger
        if test_ag.hunger > peak_hunger:
            peak_hunger = test_ag.hunger
        if test_ag.hunger > hunger_limit:
            return test_ag.hunger   # immediate failure

    # post-loop: must have spent enough time in camp
    if in_camp_hours / hours < min_camp_frac:
        return hunger_limit + 1.0   # signal failure to caller

    return peak_hunger


def _burn_in_asex_cycle(agents, roots, rng_py, test_ag):
    """One asexual cycle: for each agent, mutate once, run a 72h solo-walk
    test, and replace with a clone of a random other agent if the agent's
    final hunger exceeds BURN_IN_HUNGER_LIMIT."""
    N = len(agents)
    n_culled = 0
    for i in range(N):
        ref.mutate(roots[i], agents[i])
        final_hunger = _starvation_test_72h(roots[i], test_ag)
        if final_hunger > BURN_IN_HUNGER_LIMIT:
            donor = rng_py.randrange(N)
            if donor == i:
                donor = (donor + 1) % N
            _clone_agent_into(i, donor, agents, roots)
            n_culled += 1
    return n_culled


def _burn_in_sex(agents, roots, n_iterations, rng_py, progress=False):
    """Phase 2 burn-in: ranking-based sexual reproduction. Each iteration:
       1. pick i uniformly at random
       2. score all opposite-sex candidates by i's social_score (i's weights+Q)
       3. softmax-pick j; both i and j 'die' and are replaced by 2 offspring
    Offspring genotype = (mom + dad) / 2 averaged weights+Q, plus a few
    extra mutations to mix. Graph is a clone of either parent (50/50).
    Population size stays constant.
    """
    import math as _math
    N = len(agents)
    if N < 2:
        return

    iterator = tqdm(range(n_iterations), desc="phase 2 (sex)", unit="evt") if progress else range(n_iterations)
    for it in iterator:
        i = rng_py.randint(0, N - 1)
        ag_i = agents[i]
        # Opposite-sex candidates (sexual reproduction = M-F only)
        opposite = [k for k in range(N) if k != i and agents[k].is_female != ag_i.is_female]
        if not opposite:
            continue

        # softmax over i's ranking of all opposite candidates
        if it == 0:
            # first iteration: random pair (no learned ranking yet)
            j = opposite[rng_py.randint(0, len(opposite) - 1)]
        else:
            scores = [ref.social_score(ag_i, agents[k]) for k in opposite]
            mx = max(scores)
            exps = [_math.exp(s - mx) for s in scores]
            tot = sum(exps)
            r = rng_py.random() * tot
            cum = 0.0
            pick = 0
            for k, e in enumerate(exps):
                cum += e
                if r <= cum:
                    pick = k
                    break
            j = opposite[pick]

        mom_idx = i if agents[i].is_female else j
        dad_idx = j if agents[i].is_female else i
        mom = agents[mom_idx]; dad = agents[dad_idx]

        # Make 2 offspring (one will replace mom-slot, other dad-slot)
        new_pair = []
        for _ in range(2):
            ch = ref.agent()
            # genetic averaging
            ch.weights = [(mw + dw) / 2.0 for mw, dw in zip(mom.weights, dad.weights)]
            ch.Q = [[(mom.Q[a][b] + dad.Q[a][b]) / 2.0
                     for b in range(ref.N_FEATURES)]
                    for a in range(ref.N_FEATURES)]
            # crest: slerp midpoint
            ch.crest = ref.slerp_crests(mom.crest, dad.crest, 0.5)
            # graph: 50/50 clone from mom or dad
            parent_root = roots[mom_idx] if rng_py.random() < 0.5 else roots[dad_idx]
            ch_root = ref.clone_graph(parent_root)
            # a few mutations to add diversity
            for _ in range(3):
                ref.mutate(ch_root, ch)
            new_pair.append((ch, ch_root))

        # Replace parents
        agents[i], roots[i] = new_pair[0]
        agents[j], roots[j] = new_pair[1]


def init_population(N, seed=0):
    rng_py = random.Random(seed)
    # The "device" rng for the hot loop — numpy or cupy depending on USE_GPU.
    # Returned as `np_rng` by historical name; callers don't care which it is
    # as long as it implements .random()/.integers() identically.
    np_rng = v.xp.random.default_rng(seed)

    # python-side agent objects (state-bearing for births/snapshots only;
    # most fields are kept in the flat state dict instead).
    agents = [ref.agent() for _ in range(N)]
    roots = [ref.clone_graph(ref.canonical_root) for _ in range(N)]

    # Phase 1 burn-in: asexual mutation + per-agent 72h starvation test
    cache_path = _burnin_cache_path()
    if os.path.exists(cache_path):
        print(f"burn-in phase 1: loading cached burn-in from {cache_path}", flush=True)
        loaded_agents, loaded_roots = _load_burnin_cache(cache_path)
        if len(loaded_agents) == N:
            agents[:] = loaded_agents
            roots[:] = loaded_roots
            print(f"  loaded {len(agents)} burnt-in agents", flush=True)
        else:
            print(f"  cache size mismatch (got {len(loaded_agents)}, want {N}); regenerating",
                  flush=True)
            cache_path = None  # force regeneration
    if not os.path.exists(cache_path or '/__no_cache__'):
        print(f"burn-in phase 1 (asexual): {BURN_IN_ASEX_CYCLES} cycles "
              f"× (mutate + {BURN_IN_TEST_HOURS}h solo test, cull if hunger > {BURN_IN_HUNGER_LIMIT}, "
              f"cull if in_camp_frac < {BURN_IN_MIN_CAMP_FRAC})", flush=True)
        t0 = time.time()
        test_ag = ref.agent()  # reusable test agent (state is reset each test)
        total_culled = 0
        pbar = tqdm(range(BURN_IN_ASEX_CYCLES), desc="phase 1 (asex)", unit="cyc")
        for c in pbar:
            total_culled += _burn_in_asex_cycle(agents, roots, rng_py, test_ag)
            if c % 10 == 0:
                pbar.set_postfix(culled=total_culled)
        pbar.close()
        print(f"  phase 1 done in {time.time()-t0:.1f}s "
              f"(cumulative culls: {total_culled})", flush=True)
        # save the burnt-in population for reuse
        save_path = _burnin_cache_path()
        _save_burnin_cache(save_path, agents, roots)
        print(f"  saved burn-in cache → {save_path}", flush=True)

    # Phase 2 burn-in: ranking-based sexual reproduction (replaces parents)
    if BURN_IN_SEXUAL > 0:
        print(f"burn-in phase 2 (sexual): {BURN_IN_SEXUAL} pair-replacement events", flush=True)
        t0 = time.time()
        _burn_in_sex(agents, roots, BURN_IN_SEXUAL, rng_py, progress=True)
        print(f"  phase 2 done in {time.time()-t0:.1f}s", flush=True)

    # Now assign per-agent starting state: 2M + 2F per grid cell, ages uniform
    # in [20, 40], nobody starts pregnant. (done AFTER burn-in so sexual-phase
    # offspring inherit fresh sim-start ages)
    sex_template = [False, False, True, True]   # 2 males, 2 females per cell
    for idx, a in enumerate(agents):
        cell = idx // 4
        cx, cy = cell % ref.GRID_W, cell // ref.GRID_W
        a.x, a.y = ref.find_passable_cell(cx, cy)
        a.is_female = sex_template[idx % 4]
        a.cache = ref.CACHE_LIMIT
        a.age = rng_py.uniform(20.0, 40.0)
        a.is_in_camp = True
        a.camp_id = 0

    # Build flat state from agent attributes
    state = v.state_from_agents(agents)
    state['pregnancy_hours'] = np.array(
        [a._pregnancy_hours for a in agents], dtype=np.int32)
    state['pregnancy_target'] = np.array(
        [a._pregnancy_target for a in agents], dtype=np.int32)

    # Weights / Q from agent objects
    weights = np.array([a.weights for a in agents], dtype=np.float32)
    Q = np.array([a.Q for a in agents], dtype=np.float32)

    # Pairwise mating-history matrix — symmetric, decays each hour
    mating_matrix = np.zeros((N, N), dtype=np.float32)
    # Pairwise gift-history matrix — directional (gift_matrix[i,j] = i has given to j)
    gift_matrix = np.zeros((N, N), dtype=np.float32)

    # Family crests — (N, FAMILY_CREST_DIM) unit vectors. Initial agents get
    # random unit vectors; children inherit via slerp(mom, dad, 0.5).
    crests = np.array([a.crest for a in agents], dtype=np.float32)

    # Flatten graphs
    flat_list = [v.flatten_graph(r) for r in roots]
    graphs = v.stack_graphs(flat_list)
    n_nodes = np.array([g['n_nodes'] for g in flat_list], dtype=np.int32)

    # Cultural memory: every currently-occupied node slot starts at full
    # freshness. Empty slots (index >= n_nodes) stay at 0 — never visited,
    # never alive, so GC ignores them.
    for i in range(len(agents)):
        state['node_freshness'][i, :n_nodes[i]] = int(ref.FRESHNESS_MAX)

    # If GPU is enabled, upload all persistent state to device memory now.
    # Object graphs (agents, roots) stay on CPU — they're touched only at
    # phase boundaries (mutation, births, communication splicing) where we
    # explicitly sync the modified row back to GPU.
    if v.USE_GPU:
        for k in state:
            state[k] = v.to_device(state[k])
        for k in graphs:
            graphs[k] = v.to_device(graphs[k])
        weights = v.to_device(weights)
        Q = v.to_device(Q)
        crests = v.to_device(crests)
        mating_matrix = v.to_device(mating_matrix)
        gift_matrix = v.to_device(gift_matrix)
        n_nodes = v.to_device(n_nodes)

    return (agents, roots, state, weights, Q, graphs, n_nodes,
            mating_matrix, gift_matrix, crests, np_rng)


def menopause_step(s, rng):
    fem = s['is_female'] & ~s['is_menopausal'] & (s['age'] >= ref.MENOPAUSE_AGE)
    p = ref.MENOPAUSE_BASE_PER_HOUR * v.xp.exp(
        ref.MENOPAUSE_BETA * (s['age'] - ref.MENOPAUSE_AGE))
    trigger = fem & (rng.random(s['age'].shape[0]) < p)
    s['is_menopausal'] |= trigger


def passive_drift(s):
    s['hunger'] += np.float32(ref.BASE_HUNGER_PER_HOUR) * v.metabolic_scale_vec(s['age'])
    s['tired'] += np.float32(ref.TIRED_PER_HOUR)
    s['cache'] *= np.float32(ref.CACHE_DECAY)
    s['net_debt_flow'] *= np.float32(ref.NDF_DECAY)
    s['age']   += np.float32(ref.AGE_PER_HOUR)


def mutation_step(roots, agents, graphs, n_nodes, weights, Q, s, rng):
    """Pick mutating agents, mutate object graph, sync weights/Q + re-flatten.

    Re-flattening reorders nodes by DFS; we preserve cur by remapping via
    python-node identity (mutations never delete nodes — they modify content
    or duplicate — so the prior cur node still exists post-reflatten).
    """
    N = len(roots)
    if N == 0:
        return
    fire = rng.random(N) < s['mut_rate']
    # decide which agents fire on CPU — bring just the boolean mask back
    fire_host = v.to_host(fire) if v.USE_GPU else fire
    idx = np.flatnonzero(fire_host)
    for i in idx:
        ag = agents[i]
        i = int(i)
        # sync weights/Q into agent object — download just this agent's row
        w_row = v.to_host(weights[i])
        Q_row = v.to_host(Q[i])
        ag.weights = list(w_row)
        ag.Q = [list(row) for row in Q_row]
        # save cur's python node for re-anchoring
        old_a_nodes = ref.all_nodes(roots[i])
        old_cur = int(s['cur'][i])
        cur_pynode = old_a_nodes[old_cur] if old_cur < len(old_a_nodes) else None
        ref.mutate(roots[i], ag)
        # upload mutated weights back
        weights[i] = v.to_device(np.asarray(ag.weights, dtype=np.float32))
        Q[i] = v.to_device(np.asarray(ag.Q, dtype=np.float32))
        nn = v.reflatten_row(graphs, i, roots[i])
        n_nodes[i] = nn
        new_a = ref.all_nodes(roots[i])
        # Carry freshness through the reflatten: mutate's duplicate mode
        # adds a node, shifting DFS order. Freshness needs to follow node
        # identity so the un-shifted nodes don't suddenly look "stale".
        v.reindex_freshness(s, i, old_a_nodes, new_a)
        # remap cur via node identity (fall back to 0 if node was removed)
        if cur_pynode is not None:
            new_cur = 0
            for k, n in enumerate(new_a):
                if n is cur_pynode:
                    new_cur = k
                    break
            s['cur'][i] = new_cur
        else:
            s['cur'][i] = 0


def snapshot_dad_for_pregnancies(agents, roots, weights, Q, crests, dad_pairs):
    """For each successful (mom, dad) pair, snapshot dad's genotype onto mom.

    Snapshots are stored as HOST data (numpy lists / arrays) because births_step
    reads them later on CPU. Downloads dad rows from GPU when needed.
    """
    for mom_i, dad_i in dad_pairs:
        m = agents[mom_i]
        m._dad_graph_snapshot = ref.clone_graph(roots[dad_i])
        m._dad_weights_snapshot = list(v.to_host(weights[dad_i]))
        m._dad_Q_snapshot = [list(row) for row in v.to_host(Q[dad_i])]
        m._dad_crest_snapshot = v.to_host(crests[dad_i]).astype(np.float32).copy()


def births_step(agents, roots, state, weights, Q, graphs, n_nodes,
                mating_matrix, gift_matrix, crests, rng):
    """Advance pregnancies; produce new agents (returned as parallel lists).

    GPU note: per-mom loop runs on CPU (Python object work dominates). We
    download the relevant state columns at the top, do the work, and upload
    new chunks via xp.concatenate at the end.
    """
    # Tick pregnancy_hours for pregnant agents (device-native increment)
    fpreg = state['is_female'] & state['is_pregnant']
    state['pregnancy_hours'][...] = state['pregnancy_hours'] + fpreg.astype(state['pregnancy_hours'].dtype)
    # Tick down postpartum_hours (floored at 0) so the birth-window mask
    # naturally lifts BIRTH_WINDOW_HOURS after the most recent birth.
    state['postpartum_hours'][...] = v.xp.maximum(0, state['postpartum_hours'] - 1)
    # Bring host views of relevant columns
    is_female_h = v.to_host(state['is_female'])
    is_pregnant_h = v.to_host(state['is_pregnant'])
    pregnancy_hours_h = v.to_host(state['pregnancy_hours'])
    pregnancy_target_h = v.to_host(state['pregnancy_target'])
    camp_id_h = v.to_host(state['camp_id'])
    moms = np.flatnonzero(is_female_h & is_pregnant_h)
    if moms.size == 0:
        return 0
    due = moms[pregnancy_hours_h[moms] >= pregnancy_target_h[moms]]
    if due.size == 0:
        return 0

    new_agents = []; new_roots = []; new_flats = []
    new_weights = []; new_Qs = []; new_crests = []
    new_state_rows = []
    new_camp_ids = []   # parallel to new_agents: each baby inherits mom's camp_id
    new_xs = []; new_ys = []   # baby's spatial position = mom's
    x_h = v.to_host(state['x']) if v.USE_GPU else np.asarray(state['x'])
    y_h = v.to_host(state['y']) if v.USE_GPU else np.asarray(state['y'])

    for mom_i in due:
        m = agents[mom_i]
        n_babies = 1
        while rng.random() < math.exp(-ref.TWIN_BETA):
            n_babies += 1
        for _ in range(n_babies):
            child = ref.agent()
            child.age = 0.0
            child.is_in_camp = True
            child.cache = 0.0
            child.is_female = (rng.random() < 0.5)
            # 50/50 mom or dad graph clone
            if m._dad_graph_snapshot is not None and rng.random() < 0.5:
                child_root = ref.clone_graph(m._dad_graph_snapshot)
            else:
                child_root = ref.clone_graph(roots[mom_i])
            # genotype averaging — download mom rows on demand
            mw = v.to_host(weights[mom_i])
            dw = (np.array(m._dad_weights_snapshot, dtype=np.float32)
                  if m._dad_weights_snapshot is not None else mw)
            cw = (mw + dw) / 2.0
            mQ = v.to_host(Q[mom_i])
            dQ = (np.array(m._dad_Q_snapshot, dtype=np.float32)
                  if m._dad_Q_snapshot is not None else mQ)
            cQ = (mQ + dQ) / 2.0

            # family crest: slerp(mom, dad, 0.5) on the unit sphere
            mom_crest = v.to_host(crests[mom_i])
            dad_crest = (m._dad_crest_snapshot
                         if getattr(m, '_dad_crest_snapshot', None) is not None
                         else mom_crest)
            child_crest = ref.slerp_crests(mom_crest, dad_crest, 0.5)
            child.crest = child_crest

            new_agents.append(child)
            new_roots.append(child_root)
            new_flats.append(v.flatten_graph(child_root))
            new_weights.append(cw)
            new_Qs.append(cQ)
            new_crests.append(child_crest)
            new_state_rows.append(child)
            new_camp_ids.append(int(camp_id_h[int(mom_i)]))
            new_xs.append(int(x_h[int(mom_i)]))
            new_ys.append(int(y_h[int(mom_i)]))

        # reset mom; enter post-partum window (newborn is now in camp and
        # mom is barred from leaving for BIRTH_WINDOW_HOURS)
        state['is_pregnant'][mom_i] = False
        state['pregnancy_hours'][mom_i] = 0
        state['postpartum_hours'][mom_i] = int(ref.BIRTH_WINDOW_HOURS)
        m._dad_graph_snapshot = None
        m._dad_weights_snapshot = None
        m._dad_Q_snapshot = None
        m._dad_crest_snapshot = None

    if not new_agents:
        return 0
    # extend python lists
    agents.extend(new_agents)
    roots.extend(new_roots)
    # build new chunks as numpy; upload via to_device for device-aware concat
    new_flat_stack = v.stack_graphs(new_flats)   # numpy chunks
    for k in graphs:
        graphs[k] = v.xp.concatenate([graphs[k], v.to_device(new_flat_stack[k])])
    new_n_nodes = np.array([g['n_nodes'] for g in new_flats], dtype=np.int32)
    n_nodes_out = v.xp.concatenate([n_nodes, v.to_device(new_n_nodes)])
    new_w = np.stack(new_weights).astype(np.float32)
    new_q = np.stack(new_Qs).astype(np.float32)
    new_c = np.stack(new_crests).astype(np.float32)
    weights_out = v.xp.concatenate([weights, v.to_device(new_w)])
    Q_out = v.xp.concatenate([Q, v.to_device(new_q)])
    crests_out = v.xp.concatenate([crests, v.to_device(new_c)])

    # extend state arrays — children start at default
    M = len(new_agents)
    # Build new-state chunks as HOST numpy (per-element writes are cheap on
    # CPU; uploading to GPU once at the end is fast). Switch xp to numpy
    # temporarily via direct allocation.
    new_state_host = {}
    # mirror empty_state defaults but allocate as numpy
    new_state_host['hunger']             = np.zeros(M, np.float32)
    new_state_host['cache']              = np.zeros(M, np.float32)
    new_state_host['tired']              = np.zeros(M, np.float32)
    new_state_host['age']                = np.zeros(M, np.float32)
    new_state_host['is_female']          = np.zeros(M, np.bool_)
    new_state_host['is_in_camp']         = np.ones(M, np.bool_)
    new_state_host['sleep']              = np.zeros(M, np.bool_)
    new_state_host['consecutive_sleep']  = np.zeros(M, np.int32)
    new_state_host['is_pregnant']        = np.zeros(M, np.bool_)
    new_state_host['is_menopausal']      = np.zeros(M, np.bool_)
    new_state_host['net_debt_flow']      = np.zeros(M, np.float32)
    new_state_host['pregnancy_hours']    = np.zeros(M, np.int32)
    new_state_host['pregnancy_target']   = np.full(M, ref.PREGNANCY_HOURS, np.int32)
    new_state_host['postpartum_hours']   = np.zeros(M, np.int32)
    new_state_host['unwatched_hours']    = np.zeros(M, np.int32)
    new_state_host['mut_rate']           = np.full(M, ref.MUTATION_RATE_PER_HOUR, np.float32)
    new_state_host['adopt_rate']         = np.full(M, ref.LISTEN_ADOPT_P, np.float32)
    new_state_host['cur']                = np.zeros(M, np.int16)
    new_state_host['forced_sleep_hours'] = np.zeros(M, np.int16)
    new_state_host['talk_request']       = np.zeros(M, np.bool_)
    new_state_host['listen_request']     = np.zeros(M, np.bool_)
    new_state_host['mate_request']       = np.zeros(M, np.bool_)
    new_state_host['gift_request']       = np.zeros(M, np.bool_)
    new_state_host['make_camp_request']  = np.zeros(M, np.bool_)
    new_state_host['join_camp_request']  = np.zeros(M, np.bool_)
    new_state_host['is_watching']        = np.zeros(M, np.bool_)
    new_state_host['camp_id']            = np.array(new_camp_ids, dtype=np.int32)
    new_state_host['x']                  = np.array(new_xs, dtype=np.int32)
    new_state_host['y']                  = np.array(new_ys, dtype=np.int32)
    new_state_host['move_cooldown_hours'] = np.zeros(M, dtype=np.int8)
    # Newborn node_freshness: every slot that has a node starts at full.
    nf = np.zeros((M, state['node_freshness'].shape[1]), dtype=np.int16)
    for i in range(M):
        nf[i, :int(new_n_nodes[i])] = int(ref.FRESHNESS_MAX)
    new_state_host['node_freshness']     = nf
    for i, ch in enumerate(new_agents):
        new_state_host['is_female'][i] = ch.is_female

    for k in state:
        new_chunk = (new_state_host[k] if k in new_state_host
                     else np.zeros(M, dtype=v.to_host(state[k]).dtype))
        state[k] = v.xp.concatenate([state[k], v.to_device(new_chunk)])

    # extend mating_matrix and gift_matrix with zero rows + columns for newborns
    M_old = int(mating_matrix.shape[0])
    mat_out = v.xp.zeros((M_old + M, M_old + M), dtype=v.xp.float32)
    mat_out[:M_old, :M_old] = mating_matrix
    gift_out = v.xp.zeros((M_old + M, M_old + M), dtype=v.xp.float32)
    gift_out[:M_old, :M_old] = gift_matrix

    return len(new_agents), n_nodes_out, weights_out, Q_out, mat_out, gift_out, crests_out


def compact_dead(agents, roots, state, weights, Q, graphs, n_nodes,
                 mating_matrix, gift_matrix, crests, dead_mask):
    """Drop dead rows everywhere."""
    if not bool(dead_mask.any()):
        return weights, Q, n_nodes, mating_matrix, gift_matrix, crests
    keep = ~dead_mask
    # keep_idx needs to be CPU for indexing Python lists (agents, roots)
    keep_idx_host = v.to_host(keep) if v.USE_GPU else keep
    keep_idx = np.flatnonzero(keep_idx_host)
    # python lists
    agents[:] = [agents[i] for i in keep_idx]
    roots[:] = [roots[i] for i in keep_idx]
    # flat dicts
    for k in state:
        state[k] = state[k][keep].copy()
    for k in graphs:
        graphs[k] = graphs[k][keep].copy()
    # pairwise matrices: drop rows AND columns
    mating_matrix = mating_matrix[keep][:, keep].copy()
    gift_matrix = gift_matrix[keep][:, keep].copy()
    crests = crests[keep].copy()
    weights = weights[keep].copy()
    Q = Q[keep].copy()
    n_nodes = n_nodes[keep].copy()
    return weights, Q, n_nodes, mating_matrix, gift_matrix, crests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', action='store_true',
                    help='Use CuPy / GPU for hot tensor state (requires CUDA + nvidia-smi)')
    ap.add_argument('-N', type=int, default=None,
                    help='Override N_AGENTS (population size)')
    ap.add_argument('--days', type=int, default=None,
                    help='Run for this many sim-days then exit (omit for indefinite)')
    ap.add_argument('--no-burnin', action='store_true',
                    help='Skip the burn-in phase even if BURN_IN_ASEX_CYCLES > 0')
    ap.add_argument('--dump-every', type=int, default=None,
                    help='Override DUMP_EVERY_DAYS (e.g. 1 for daily-dump gif rendering)')
    args, _ = ap.parse_known_args()
    if args.dump_every is not None:
        global DUMP_EVERY_DAYS
        DUMP_EVERY_DAYS = int(args.dump_every)
        print(f'[dump cadence] every {DUMP_EVERY_DAYS} day(s)', flush=True)
    if args.gpu:
        v.set_device(True)
        print('[device] GPU (CuPy) enabled', flush=True)
    if args.N:
        global N_AGENTS
        N_AGENTS = args.N
        print(f'[N_AGENTS] overridden to {N_AGENTS}', flush=True)
    if args.no_burnin:
        ref.BURN_IN_ASEX_CYCLES = 0
        global BURN_IN_ASEX_CYCLES
        BURN_IN_ASEX_CYCLES = 0
        print('[burn-in] skipped via --no-burnin', flush=True)
    (agents, roots, state, weights, Q, graphs, n_nodes,
     mating_matrix, gift_matrix, crests, rng) = init_population(N_AGENTS)
    # Multi-camp dynamics: agents partitioned by camp_id; each camp has its
    # own larder. Founders all start as unaffiliated wanderers (camp_id=-1)
    # and must form camps via the three-stage migration: leave_camp →
    # make_camp / join_camp. The birth-window safety net (force-settle for
    # pregnant moms near term) means newborns won't be born in the wild
    # during bootstrap. The wanderer sentinel (-1) is never a real larder.
    N = state['hunger'].shape[0]
    # Homogeneous founder placement: one M+F pair per passable cell, both
    # 20 years old, sharing one camp per cell. Camp IDs match cell index.
    passable_cells = list(zip(*np.where(ref.TERRAIN_MASK)))
    n_cells = len(passable_cells)
    '''
    if N != 2 * n_cells:
        raise ValueError(f'N_AGENTS ({N}) must equal 2 × passable cells ({n_cells})')
    '''
    camp_caches = {}
    camp_positions = {}
    for k in range(n_cells):
        cx, cy = passable_cells[k]
        i_m, i_f = 2 * k, 2 * k + 1
        for i, female in ((i_m, False), (i_f, True)):
            state['camp_id'][i] = k
            state['is_in_camp'][i] = True
            state['x'][i] = cx
            state['y'][i] = cy
            state['is_female'][i] = female
            state['age'][i] = 20.0
            state['is_pregnant'][i] = False
            state['pregnancy_hours'][i] = 0
        camp_caches[k] = 0.0
        camp_positions[k] = (int(cx), int(cy))
    next_camp_id = n_cells

    # Per-cell resource pools, shape (GRID_W, GRID_H, 3). Axis-2 indexes
    # [0=hunt, 1=fish, 2=gather]. Per-cell K = global K / num_cells so total
    # carrying capacity is unchanged. Yields/probability scale by R/K of the
    # agent's current cell; extraction is local to the cell foraged.
    # Impassable terrain cells are zeroed (no growth, no yield — unreachable
    # anyway, but zeroing keeps the readout aggregates honest).
    resources = np.zeros((ref.GRID_W, ref.GRID_H, 3), dtype=np.float64)
    resources[..., 0] = ref.RESOURCE_K_HUNT_PER_CELL
    resources[..., 1] = ref.RESOURCE_K_FISH_PER_CELL
    resources[..., 2] = ref.RESOURCE_K_GATHER_PER_CELL
    impassable = ~ref.TERRAIN_MASK
    for _k in range(3):
        resources[..., _k][impassable] = 0.0
    os.makedirs(DUMP_DIR, exist_ok=True)
    # Persist the terrain mask once so viz / analysis can render it.
    with open(os.path.join(DUMP_DIR, 'terrain.json'), 'w') as _tf:
        json.dump({'mask': ref.TERRAIN_MASK.astype(int).tolist(),
                    'passable_frac': float(ref.TERRAIN_PASSABLE_FRAC),
                    'frequency': int(ref.TERRAIN_FREQUENCY),
                    'seed': int(ref.TERRAIN_SEED)}, _tf)

    # Canonical action profile — used to track drift in the readout.
    # Each agent has an action profile = histogram of true_action values
    # across their graph slots. The canonical's profile is the reference
    # (what every agent is born with). Population-level drift is the
    # complement of the mean cosine similarity to canonical.
    # `div` is the std of those per-agent similarities — high std = diverse
    # population, low std = clonal.
    _canon_actions = np.zeros(len(ref.ACTION_POOL) + 1, dtype=np.float64)
    for _n in ref.all_nodes(ref.canonical_root):
        a_id = v.ACTION_FN_TO_ID.get(_n.true_action, v.A_NONE)
        _canon_actions[int(a_id)] += 1
    _canon_norm = np.linalg.norm(_canon_actions)

    def _drift_stats():
        """Return (mean_canon_sim, std_canon_sim) over current population.
        mean_canon_sim near 1.0 = clonal-to-canonical; near 0 = diverged.
        std = how heterogeneous the divergence is."""
        N = state['hunger'].shape[0]
        if N == 0:
            return 1.0, 0.0
        n_act = _canon_actions.size
        true_a = v.to_host(graphs['true_action']) if v.USE_GPU else np.asarray(graphs['true_action'])
        nn_h = v.to_host(n_nodes) if v.USE_GPU else np.asarray(n_nodes)
        sims = np.empty(N, dtype=np.float64)
        for i in range(N):
            row = np.zeros(n_act, dtype=np.float64)
            valid = true_a[i, :int(nn_h[i])]
            for a_id in valid:
                if 0 <= a_id < n_act:
                    row[int(a_id)] += 1
            denom = np.linalg.norm(row) * _canon_norm
            sims[i] = float(np.dot(row, _canon_actions) / denom) if denom > 0 else 0.0
        return float(sims.mean()), float(sims.std())

    print(f"{'day':>4} {'pop':>5} {'F':>4} {'preg':>4} {'kid':>4} "
          f"{'birth':>5} {'death':>5} {'hng':>3} {'exp':>3} "
          f"{'haz':>3} {'ngl':>3} {'fslp':>4} {'mate':>4} {'adopt':>5} "
          f"{'mh':>5} {'mc':>5} {'camp$':>8} {'#cmp':>4} "
          f"{'R_h%':>4} {'R_f%':>4} {'R_g%':>4} "
          f"{'canSim':>6} {'divSt':>5} {'sec':>5}")

    deaths_today = births_today = matings_today = 0
    hng = exposed_today = haz = ngl = adopts_today = 0
    T = 24 * (args.days if args.days else SIM_DAYS)
    t_start = time.time()
    day_t0 = time.time()

    for t in range(T):
        hour = t % 24
        N = state['hunger'].shape[0]
        if N == 0:
            print(f"extinction at hour {t}")
            break

        if hour == 0:
            v.update_adaptive_rates(state, weights, Q, rng,
                                     mating_matrix=mating_matrix,
                                     gift_matrix=gift_matrix,
                                     crests=crests,
                                     camp_caches=camp_caches)

        passive_drift(state)
        menopause_step(state, rng)

        # decay pairwise history matrices once per hour
        mating_matrix *= np.float32(ref.MATING_DECAY)
        gift_matrix   *= np.float32(ref.GIFT_DECAY)

        # Season: a sin wave along the y axis whose phase sweeps through
        # time over SEASON_YEAR_DAYS. Per-agent value from each agent's row;
        # every latitude cycles through a full year (no fixed equator).
        season_val = ref.season_at_y(v.to_host(state['y']) if v.USE_GPU
                                       else np.asarray(state['y']), t)
        season_val = v.xp.asarray(season_val, dtype=v.xp.float32)
        camp_caches = v.step_all(graphs, state, hour, season_val, camp_caches, rng,
                                  n_nodes, resources=resources,
                                  camp_positions=camp_positions)
        # cultural memory: nodes visited this hour got refreshed inside
        # step_all; tick all freshness down by FRESHNESS_DECAY here so
        # unvisited slots slowly age.
        v.decay_freshness(state)
        # Move cooldown ticks down once per hour. Agents whose last move set
        # cooldown=2 will be able to move again after 2 hours of decrement.
        state['move_cooldown_hours'] = v.xp.maximum(
            0, state['move_cooldown_hours'] - 1)
        # Camp larder decay (food rots even while guarded — CACHE_DECAY has
        # ~1.5 day half-life). Total loss when the camp empties is handled
        # separately by cleanup_empty_camps (no one to guard → animals
        # take it).
        for _cid in list(camp_caches):
            camp_caches[_cid] *= float(ref.CACHE_DECAY)
        # resource pools regenerate via logistic growth (no extinction)
        v.resource_growth_step(resources)

        # forced sleep: agents whose tired hits TIRED_DEATH pass out for 12h
        v.forced_sleep_step(state)

        # birth-window safety net: any wandering mom in late pregnancy or
        # the first week post-partum is force-settled (join_camp_request
        # set; join_camp_phase resolves it, falling back to make if no
        # camps exist). Runs BEFORE join_camp_phase so the request is in
        # place by the time it executes.
        v.birth_window_force_settle_phase(state)

        # resolve camp-management requests (set by agent_join_camp / agent_make_camp).
        # join runs FIRST: it absorbs wanderers into existing camps, falling
        # back to make when none exist. make_camp_phase then catches any
        # still-wandering agents who only fired make (or who fired join but
        # there were no real camps yet and they aren't covered by the
        # fallback — defensive). Together this favors clustering.
        next_camp_id = v.join_camp_phase(state, weights, Q, rng,
                          mating_matrix=mating_matrix,
                          gift_matrix=gift_matrix, crests=crests,
                          camp_caches=camp_caches,
                          next_camp_id=next_camp_id,
                          camp_positions=camp_positions)
        next_camp_id = v.make_camp_phase(state, camp_caches, next_camp_id,
                                           camp_positions=camp_positions)

        nt, nl, na = v.communication_phase(
            state, graphs, n_nodes, weights, Q, rng,
            roots=roots, mating_matrix=mating_matrix, gift_matrix=gift_matrix,
            crests=crests, camp_caches=camp_caches)
        adopts_today += na

        n_mat, dad_pairs, mating_events = v.mating_phase(
            state, weights, Q, rng, mating_matrix=mating_matrix,
            gift_matrix=gift_matrix, crests=crests, camp_caches=camp_caches)
        matings_today += n_mat
        # apply mating events to the symmetric mating matrix
        for a, b in mating_events:
            mating_matrix[a, b] += 1.0
            mating_matrix[b, a] += 1.0
        snapshot_dad_for_pregnancies(agents, roots, weights, Q, crests, dad_pairs)

        # gift phase — runs after mating, before watching/mutation/births
        n_gift, gift_events = v.gift_phase(
            state, weights, Q, rng,
            mating_matrix=mating_matrix, gift_matrix=gift_matrix,
            crests=crests, camp_caches=camp_caches)
        # gift_matrix is directional: only the giver→recipient direction
        for giver, recip, amt in gift_events:
            gift_matrix[giver, recip] += np.float32(amt)

        camp_caches = v.watching_phase(state, camp_caches)

        # Daily cultural GC: prune nodes that haven't been visited recently.
        # Frees slots so subsequent subgraph adoptions have room.
        if (t % int(ref.GC_INTERVAL_HOURS)) == 0:
            v.gc_dead_nodes(state, graphs, roots, n_nodes)

        mutation_step(roots, agents, graphs, n_nodes, weights, Q, state, rng)

        out = births_step(agents, roots, state, weights, Q, graphs, n_nodes,
                          mating_matrix, gift_matrix, crests, rng)
        if isinstance(out, tuple):
            n_births, n_nodes, weights, Q, mating_matrix, gift_matrix, crests = out
            births_today += n_births

        # Recompute season from current y positions — births in this hour
        # changed N, so the hour-top season_val array is stale.
        death_season = ref.season_at_y(v.to_host(state['y']) if v.USE_GPU
                                         else np.asarray(state['y']), t)
        death_season = v.xp.asarray(death_season, dtype=v.xp.float32)
        dead, why = death_mask_with_counts(state, rng, season_val=death_season)
        weights, Q, n_nodes, mating_matrix, gift_matrix, crests = compact_dead(
            agents, roots, state, weights, Q, graphs, n_nodes,
            mating_matrix, gift_matrix, crests, dead)
        deaths_today += int(dead.sum())
        hng += why['hunger']; exposed_today += why['exposed']
        haz += why['hazard']; ngl += why['neglect']

        # disband any camp that just lost its last member (cache destroyed)
        v.cleanup_empty_camps(state, camp_caches, camp_positions=camp_positions)

        if hour == 23:
            day = t // 24
            n = state['hunger'].shape[0]
            if n == 0:
                break
            nf = int(state['is_female'].sum())
            npg = int(state['is_pregnant'].sum())
            nk = int((state['age'] < ref.WATCH_AGE_YEARS).sum())
            n_fslp = int((state['forced_sleep_hours'] > 0).sum())
            mh = float(state['hunger'].mean())
            mc = float(state['cache'].mean())
            sec = time.time() - day_t0
            mean_adopt = float(state['adopt_rate'].mean())
            total_camp_cache = sum(camp_caches.values())
            n_camps = len(camp_caches)
            # Per-cell resources array: report grid-mean R/K for each kind.
            # Average only over passable cells — impassable cells are
            # permanently zero and would skew the readout downward.
            _pmask = ref.TERRAIN_MASK
            r_h = 100.0 * float(resources[..., 0][_pmask].mean()) / ref.RESOURCE_K_HUNT_PER_CELL
            r_f = 100.0 * float(resources[..., 1][_pmask].mean()) / ref.RESOURCE_K_FISH_PER_CELL
            r_g = 100.0 * float(resources[..., 2][_pmask].mean()) / ref.RESOURCE_K_GATHER_PER_CELL
            can_sim, div_st = _drift_stats()
            print(f"{day:>4} {n:>5} {nf:>4} {npg:>4} {nk:>4} "
                  f"{births_today:>5} {deaths_today:>5} "
                  f"{hng:>3} {exposed_today:>3} {haz:>3} {ngl:>3} "
                  f"{n_fslp:>4} {matings_today:>4} {mean_adopt:>5.2f} "
                  f"{mh:>5.0f} {mc:>5.0f} "
                  f"{total_camp_cache:>8.0f} {n_camps:>4d} "
                  f"{r_h:>3.0f}% {r_f:>3.0f}% {r_g:>3.0f}% "
                  f"{can_sim:>6.3f} {div_st:>5.3f} {sec:>5.1f}",
                  flush=True)
            day_t0 = time.time()
            deaths_today = births_today = matings_today = 0
            hng = exposed_today = haz = ngl = adopts_today = 0

            if (day + 1) % DUMP_EVERY_DAYS == 0:
                week = (day + 1) // DUMP_EVERY_DAYS
                path = os.path.join(DUMP_DIR, f"week_{week:02d}.json")
                dump_population(path, day, state, weights, Q, roots,
                                 mating_matrix=mating_matrix,
                                 gift_matrix=gift_matrix,
                                 crests=crests,
                                 camp_caches=camp_caches,
                                 resources=resources,
                                 agents=agents,
                                 camp_positions=camp_positions)
                print(f"  -> dumped {path}", flush=True)

    print(f"total elapsed: {time.time()-t_start:.0f}s")


def death_mask_with_counts(s, rng, season_val=1.0):
    return v.death_mask(s, rng, season_val=season_val)


if __name__ == '__main__':
    main()
