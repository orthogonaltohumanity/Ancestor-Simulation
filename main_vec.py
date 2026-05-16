"""Vectorized prototype of main.py's per-hour walk.

Goal: replace the N×33 Python-level node walks in main.py:1178-1189 with
~33 numpy ops over an (N,) agent vector. Same canonical graph, same action
semantics, but the hot loop runs on flat int/float arrays.

Status: hot loop + action dispatch + CAMP reduction. Phases (mating,
watching, pregnancy, deaths) and the long sim loop are not yet wired up;
this module is intended to be imported by validate_vec.py.
"""
import numpy as np
import main as ref

# --- device abstraction: numpy on CPU, cupy on GPU ---
# `xp` is the active array module. Code below uses `xp.X` instead of `xp.X`
# so the same hot-path implementations run on CPU or GPU based on the flag.
# numpy is also still imported as `np` for cases where we explicitly need
# host arrays (boundary transfers, RNG for CPU-side per-agent work).
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

USE_GPU = False
xp = np

def set_device(use_gpu: bool):
    """Switch the array module. Call before init_population for a GPU run.

    When use_gpu=True, `xp` is rebound to cupy; persistent state allocated
    after this call lands on GPU memory. CPU↔GPU transfers across phase
    boundaries are explicit (.get() / cp.asarray) in run_vec.py.
    """
    global xp, USE_GPU
    if use_gpu:
        if not _HAS_CUPY:
            raise RuntimeError("CuPy is not installed. `pip install cupy-cuda12x`.")
        xp = cp
        USE_GPU = True
    else:
        xp = np
        USE_GPU = False


def to_host(a):
    """cupy array → numpy array, or pass-through if already numpy."""
    if _HAS_CUPY and isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)


def to_device(a, dtype=None):
    """numpy array → active xp array. dtype optional cast."""
    if dtype is not None:
        return xp.asarray(a, dtype=dtype)
    return xp.asarray(a)

# --- shape constants ---
MAX_NODES = 300                   # hard array cap; truncates if exceeded
ADOPTION_CAP = 200                # adoption stops adding when n_nodes hits this
                                  # (no REPLACE — full listeners just skip the adoption)
K_GRAFT = 10                      # subgraph adoption size: BFS up to K nodes from
                                  # the talker's exposed node; whole subgraph
                                  # transmitted as a cultural unit
MAX_SEQ_LEN = ref.MAX_SEQ_LEN     # 10
MAX_ACTION = ref.MAX_ACTION       # 33

# --- variable indexing for the (N, 9) state matrix ---
VAR_NAMES = ('hunger', 'cache', 'tired', 'time', 'age', 'season',
             'is_female', 'is_in_camp', 'sleep', 'rng',
             'x', 'y',
             'nearest_camp_n', 'nearest_camp_s',
             'nearest_camp_e', 'nearest_camp_w')
N_VARS = len(VAR_NAMES)
VAR_IDX = {n: i for i, n in enumerate(VAR_NAMES)}
(SV_HUNGER, SV_CACHE, SV_TIRED, SV_TIME, SV_AGE, SV_SEASON,
    SV_IS_FEMALE, SV_IS_IN_CAMP, SV_SLEEP, SV_RNG,
    SV_X, SV_Y, SV_NC_N, SV_NC_S, SV_NC_E, SV_NC_W) = range(N_VARS)

# --- chunk op codes ---
OP_LT, OP_GT, OP_EQ, OP_BOOL = 0, 1, 2, 3

# --- action codes: index into ref.ACTION_POOL ---
# ref.ACTION_POOL = [sleep, wake, eat, hunt, fish, gather, go_to_camp,
#                    leave_camp, talk, listen, deposit, withdraw,
#                    propose_mate, watch_children, idle, None]
(A_SLEEP, A_WAKE, A_EAT, A_HUNT, A_FISH, A_GATHER, A_GO_TO_CAMP,
    A_MOVE_N, A_MOVE_S, A_MOVE_E, A_MOVE_W,
    A_LEAVE_CAMP, A_TALK, A_LISTEN, A_DEPOSIT, A_WITHDRAW,
    A_PROPOSE_MATE, A_WATCH_CHILDREN, A_IDLE, A_GIFT,
    A_MAKE_CAMP, A_JOIN_CAMP, A_NONE) = range(23)
ACTION_FN_TO_ID = {fn: i for i, fn in enumerate(ref.ACTION_POOL)}

# --- 16-op truth-table lookup: TT[op_id, (a<<1)|b] -> 0/1 ---
# Cached per active xp module so we don't rebuild every call. set_device()
# swaps `xp` and we lazily build a matching TT on first use.
_TT_CACHE = {}
def _TT():
    mod_id = id(xp)
    if mod_id not in _TT_CACHE:
        _TT_CACHE[mod_id] = xp.array(
            [[(op_id >> k) & 1 for k in range(4)] for op_id in range(16)],
            dtype=xp.uint8)
    return _TT_CACHE[mod_id]
TT = _TT()  # initial seed for numpy-mode


# ---------------------------------------------------------------------------
# Graph flattening
# ---------------------------------------------------------------------------

def flatten_graph(root, max_nodes=MAX_NODES, max_seq=MAX_SEQ_LEN):
    """Flatten an object-graph into a dict of numpy arrays for one agent.

    If the object graph has more than max_nodes nodes (e.g., a mutation pushed
    past the cap), DFS-truncate to the first max_nodes. Edges to truncated
    nodes fall through to slot 0 (root) via the idx.get default.

    NB: this always allocates HOST (numpy) arrays — the body does many tiny
    per-element writes from Python objects which would be catastrophic on
    GPU. Callers upload the result to device when needed (cupy supports
    assignment from numpy directly: `gpu_array[idx] = numpy_array`).
    """
    nodes = ref.all_nodes(root)
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    idx = {id(n): i for i, n in enumerate(nodes)}

    seq_var      = np.full((max_nodes, max_seq), -1, dtype=np.int8)
    seq_op       = np.full((max_nodes, max_seq), -1, dtype=np.int8)
    seq_value    = np.zeros((max_nodes, max_seq), dtype=np.float32)
    seq_links    = np.zeros((max_nodes, max_seq - 1), dtype=np.int8)
    seq_len      = np.zeros(max_nodes, dtype=np.int8)
    true_action  = np.full(max_nodes, A_NONE, dtype=np.int8)
    false_action = np.full(max_nodes, A_NONE, dtype=np.int8)
    true_node    = np.zeros(max_nodes, dtype=np.int16)
    false_node   = np.zeros(max_nodes, dtype=np.int16)

    for i, n in enumerate(nodes):
        if n.seq is not None:
            chunks = n.seq.chunks[:max_seq]
            for k, c in enumerate(chunks):
                seq_var[i, k] = VAR_IDX[c.var]
                if c.var in ref.ORDERED_VARS:
                    seq_op[i, k] = {'<': OP_LT, '>': OP_GT, '==': OP_EQ}[c.op]
                    seq_value[i, k] = float(c.value if c.value is not None else 0.0)
                else:
                    seq_op[i, k] = OP_BOOL
                    if c.value is None:
                        seq_value[i, k] = -1.0   # sentinel: just bool(val)
                    else:
                        seq_value[i, k] = 1.0 if c.value else 0.0
            seq_len[i] = len(chunks)
            for k, lnk in enumerate(n.seq.chunk_links[:max_seq - 1]):
                seq_links[i, k] = lnk
        true_action[i]  = ACTION_FN_TO_ID[n.true_action]
        false_action[i] = ACTION_FN_TO_ID[n.false_action]
        true_node[i]  = idx.get(id(n.true_node), 0)  if n.true_node  is not None else 0
        false_node[i] = idx.get(id(n.false_node), 0) if n.false_node is not None else 0

    return dict(
        seq_var=seq_var, seq_op=seq_op, seq_value=seq_value,
        seq_links=seq_links, seq_len=seq_len,
        true_action=true_action, false_action=false_action,
        true_node=true_node, false_node=false_node,
        n_nodes=len(nodes),
    )


def stack_graphs(graph_dicts, max_nodes=MAX_NODES, max_seq=MAX_SEQ_LEN):
    """Stack N per-agent flat graphs into (N, M, ...) arrays.

    Always stacks as numpy (graph_dicts come from flatten_graph which is
    host-only). Callers upload to GPU once when initializing persistent state.
    """
    out = {
        'seq_var':      np.stack([g['seq_var']      for g in graph_dicts]),
        'seq_op':       np.stack([g['seq_op']       for g in graph_dicts]),
        'seq_value':    np.stack([g['seq_value']    for g in graph_dicts]),
        'seq_links':    np.stack([g['seq_links']    for g in graph_dicts]),
        'seq_len':      np.stack([g['seq_len']      for g in graph_dicts]),
        'true_action':  np.stack([g['true_action']  for g in graph_dicts]),
        'false_action': np.stack([g['false_action'] for g in graph_dicts]),
        'true_node':    np.stack([g['true_node']    for g in graph_dicts]),
        'false_node':   np.stack([g['false_node']   for g in graph_dicts]),
    }
    return out


# ---------------------------------------------------------------------------
# Agent state arrays
# ---------------------------------------------------------------------------

def empty_state(N):
    """Initialize per-agent state vectors. Mirrors agent.__init__ defaults."""
    return dict(
        hunger=xp.zeros(N, xp.float32),
        cache=xp.zeros(N, xp.float32),
        tired=xp.zeros(N, xp.float32),
        age=xp.zeros(N, xp.float32),
        is_female=xp.zeros(N, xp.bool_),
        is_in_camp=xp.ones(N, xp.bool_),
        sleep=xp.zeros(N, xp.bool_),
        consecutive_sleep=xp.zeros(N, xp.int32),
        is_pregnant=xp.zeros(N, xp.bool_),
        is_menopausal=xp.zeros(N, xp.bool_),
        net_debt_flow=xp.zeros(N, xp.float32),
        pregnancy_hours=xp.zeros(N, xp.int32),
        pregnancy_target=xp.full(N, ref.PREGNANCY_HOURS, xp.int32),
        # postpartum_hours: counts down from BIRTH_WINDOW_HOURS after each
        # birth so the new mom is held in camp during the first week of
        # her newborn's life (kid can't forage, can't survive in the wild).
        postpartum_hours=xp.zeros(N, xp.int32),
        unwatched_hours=xp.zeros(N, xp.int32),
        mut_rate=xp.full(N, ref.MUTATION_RATE_PER_HOUR, dtype=xp.float32),
        adopt_rate=xp.full(N, ref.LISTEN_ADOPT_P, dtype=xp.float32),
        # persistent program counter — agents pick up where they left off
        cur=xp.zeros(N, xp.int16),
        # forced-sleep timer: hours remaining when agent is passed out
        forced_sleep_hours=xp.zeros(N, xp.int16),
        # request flags
        talk_request=xp.zeros(N, xp.bool_),
        listen_request=xp.zeros(N, xp.bool_),
        mate_request=xp.zeros(N, xp.bool_),
        gift_request=xp.zeros(N, xp.bool_),
        make_camp_request=xp.zeros(N, xp.bool_),
        join_camp_request=xp.zeros(N, xp.bool_),
        is_watching=xp.zeros(N, xp.bool_),
        # camp affiliation (which dynamic camp the agent belongs to; default 0)
        camp_id=xp.zeros(N, xp.int32),
        # spatial grid position
        x=xp.zeros(N, xp.int32),
        y=xp.zeros(N, xp.int32),
        # cooldown after a move action — agents can only move every 2 hours.
        # Set to MOVE_COOLDOWN_HOURS on each successful move, ticked down by 1
        # per hour (clamped at 0). Move handlers only fire when this is 0.
        move_cooldown_hours=xp.zeros(N, xp.int8),
        # cultural memory: per-node freshness counter, decays each hour, refreshed
        # on every visit. When a slot hits 0 it gets pruned in gc_dead_nodes.
        node_freshness=xp.zeros((N, MAX_NODES), xp.int16),
    )


def state_from_agents(agents):
    """Build state dict from a list of ref.agent instances."""
    N = len(agents)
    s = empty_state(N)
    for i, a in enumerate(agents):
        s['hunger'][i] = a.hunger
        s['cache'][i] = a.cache
        s['tired'][i] = a.tired
        s['age'][i] = a.age
        s['is_female'][i] = a.is_female
        s['is_in_camp'][i] = a.is_in_camp
        s['sleep'][i] = a.sleep
        s['consecutive_sleep'][i] = a.consecutive_sleep
        s['is_pregnant'][i] = a.is_pregnant
        s['is_menopausal'][i] = a._is_menopausal
        s['net_debt_flow'][i] = a.net_debt_flow
        s['camp_id'][i] = getattr(a, 'camp_id', 0)
        s['x'][i] = getattr(a, 'x', 0)
        s['y'][i] = getattr(a, 'y', 0)
    return s


def state_to_agents(s, agents):
    """Write state arrays back into a list of ref.agent instances."""
    for i, a in enumerate(agents):
        a.hunger = float(s['hunger'][i])
        a.cache = float(s['cache'][i])
        a.tired = float(s['tired'][i])
        a.age = float(s['age'][i])
        a.is_female = bool(s['is_female'][i])
        a.is_in_camp = bool(s['is_in_camp'][i])
        a.sleep = bool(s['sleep'][i])
        a.consecutive_sleep = int(s['consecutive_sleep'][i])
        a.is_pregnant = bool(s['is_pregnant'][i])
        a._is_menopausal = bool(s['is_menopausal'][i])
        a.net_debt_flow = float(s['net_debt_flow'][i])
        a._talk_request = bool(s['talk_request'][i])
        a._listen_request = bool(s['listen_request'][i])
        a._mate_request = bool(s['mate_request'][i])
        a._is_watching = bool(s['is_watching'][i])


# ---------------------------------------------------------------------------
# Vectorized chunk evaluation
# ---------------------------------------------------------------------------

def _vision_distances(s, has_camp):
    """Compute (N,) per-agent nearest-camp distances in each cardinal direction,
    looking up to VISION_RANGE cells away on a TORUS (vision wraps around grid
    edges). Returns dict with keys 'n','s','e','w' → int16 arrays valued in
    [1, VISION_RANGE] for the nearest hit, or -1 if no camp visible within
    range in that direction. `has_camp` is a (GRID_W, GRID_H) bool array of
    cell-occupancy by any camp. Convention: smaller-y = north.

    Line of sight is OCCLUDED by impassable terrain: once the ray crosses an
    impassable cell, every cell beyond it in that direction is invisible —
    you can't see a camp through a mountain.

    Loop is over distance (sequential, since "nearest" is what we want).
    Per-step it's a single (N,) bool gather + masked update of the output."""
    ax = s['x']; ay = s['y']
    N = ax.shape[0]
    W = int(ref.GRID_W); H = int(ref.GRID_H)
    R = int(ref.VISION_RANGE)
    INT_T = xp.int16
    terrain = xp.asarray(ref.TERRAIN_MASK)   # (W, H) bool, True = passable
    out = {}
    # Build (dx, dy, key) tuples for the four cardinals.
    for key, dx, dy in (('n', 0, -1), ('s', 0, 1), ('e', 1, 0), ('w', -1, 0)):
        dist = xp.full(N, -1, dtype=INT_T)
        blocked = xp.zeros(N, dtype=bool)   # ray occluded by impassable terrain
        for d in range(1, R + 1):
            tx = (ax + dx * d) % W
            ty = (ay + dy * d) % H
            # a hit counts only if not already occluded by closer terrain
            hit = has_camp[tx, ty] & (dist < 0) & ~blocked
            dist = xp.where(hit, INT_T(d), dist)
            # an impassable cell at this step hides everything beyond it
            blocked = blocked | ~terrain[tx, ty]
        out[key] = dist
    return out


def _build_state_matrix(s, hour, season_val, N, rng=None, vision=None):
    """(N, N_VARS) float32 view of agent state, suitable for chunk eval.
    If rng is provided, SV_RNG column gets a fresh uniform [0, 1) draw per agent.
    `vision` (optional) is a dict {'n','s','e','w': (N,) int8} populated by
    `_vision_distances`. If absent, vision columns default to -1.

    SV_TIME is the agent's *local* hour given their column x — the world has
    24 timezones across its width. SV_SEASON is the agent's *local* season
    given their row y (sin band: summer at y≈H/4, winter at y≈3H/4)."""
    M = xp.empty((N, N_VARS), dtype=xp.float32)
    # Per-agent local hour from x (timezones across the torus).
    x_host = to_host(s['x']) if USE_GPU else np.asarray(s['x'])
    local_h_np = ref.local_hour(x_host, int(hour))
    local_h = xp.asarray(local_h_np, dtype=xp.float32)
    M[:, SV_HUNGER]     = s['hunger']
    M[:, SV_CACHE]      = s['cache']
    M[:, SV_TIRED]      = s['tired']
    M[:, SV_TIME]       = local_h
    M[:, SV_AGE]        = s['age']
    M[:, SV_SEASON]     = season_val
    M[:, SV_IS_FEMALE]  = s['is_female'].astype(xp.float32)
    M[:, SV_IS_IN_CAMP] = s['is_in_camp'].astype(xp.float32)
    M[:, SV_SLEEP]      = s['sleep'].astype(xp.float32)
    M[:, SV_RNG]        = rng.random(N, dtype=xp.float32) if rng is not None else 0.0
    M[:, SV_X]          = s['x'].astype(xp.float32)
    M[:, SV_Y]          = s['y'].astype(xp.float32)
    if vision is not None:
        M[:, SV_NC_N] = vision['n'].astype(xp.float32)
        M[:, SV_NC_S] = vision['s'].astype(xp.float32)
        M[:, SV_NC_E] = vision['e'].astype(xp.float32)
        M[:, SV_NC_W] = vision['w'].astype(xp.float32)
    else:
        M[:, SV_NC_N] = -1.0
        M[:, SV_NC_S] = -1.0
        M[:, SV_NC_E] = -1.0
        M[:, SV_NC_W] = -1.0
    return M


def _eval_seqs(seq_var, seq_op, seq_value, seq_links, seq_len, state_mat):
    """Evaluate one chunk_seq per agent. All inputs are (N, ...) for the
    *current node* of each agent (already gathered)."""
    N, S = seq_var.shape
    ar = xp.arange(N)

    # gather state value per chunk slot. var=-1 (unused) → just read column 0
    # since result is masked out by seq_len.
    safe_var = xp.where(seq_var >= 0, seq_var, 0).astype(xp.intp)
    val = state_mat[ar[:, None], safe_var]   # (N, S) float32

    # chunk-level boolean result
    cr_lt = (seq_op == OP_LT) & (val <  seq_value)
    cr_gt = (seq_op == OP_GT) & (val >  seq_value)
    cr_eq = (seq_op == OP_EQ) & (val == seq_value)
    val_b = val != 0.0
    target_b = seq_value > 0.5
    cr_bn = (seq_op == OP_BOOL) & (seq_value < 0) & val_b           # value is None: bool(val)
    cr_be = (seq_op == OP_BOOL) & (seq_value >= 0) & (val_b == target_b)
    cr = cr_lt | cr_gt | cr_eq | cr_bn | cr_be       # (N, S)

    # left-fold the sequence
    acc = cr[:, 0].copy()
    for k in range(S - 1):
        b = cr[:, k + 1]
        idx = (acc.astype(xp.uint8) << 1) | b.astype(xp.uint8)
        new = _TT()[seq_links[:, k].astype(xp.intp), idx].astype(bool)
        active = k < (seq_len - 1)
        acc = xp.where(active, new, acc)
    return acc                                       # (N,)


# ---------------------------------------------------------------------------
# Vectorized action dispatch
# ---------------------------------------------------------------------------

def birth_window_mask(s):
    """Bool (N,) mask: True for moms in the late-pregnancy or post-partum
    window where they MUST stay in camp (or be force-settled). Newborns
    can't forage and can't survive in the wild, so the mom is locked to a
    camp from BIRTH_WINDOW_HOURS before her due date through
    BIRTH_WINDOW_HOURS after birth."""
    near_due = (s['is_pregnant']
                & (s['pregnancy_hours'] >= s['pregnancy_target'] - ref.BIRTH_WINDOW_HOURS))
    post_partum = s['postpartum_hours'] > 0
    return near_due | post_partum


def _apply_actions(action, s, camp_caches, rng, walking=None, season_val=1.0,
                   resources=None, birth_window=None, hour=0):
    """Apply one visit-step's actions across N agents.

    `walking` is an optional (N,) bool mask; when provided, all action effects
    are gated by it so that non-walking agents (e.g. forced-sleep) are no-ops.

    `camp_caches` is a dict {camp_id: float kcal} — mutated in place. Each
    agent's deposit/withdraw targets their own camp_id's larder.

    `resources` (optional) is a dict {'hunt'/'fish'/'gather': float kcal}.
    When provided, forage yields are scaled by (R/K) per resource type and
    extracted kcal is subtracted from R (floored at MIN_RESOURCE_FRAC*K).
    """
    N = action.shape[0]
    if walking is None:
        walking = xp.ones(N, dtype=bool)
    age = s['age']
    sleep = s['sleep']
    cs = s['consecutive_sleep']
    is_in_camp = s['is_in_camp']
    is_pregnant = s['is_pregnant']

    # pregnant females can still forage during the first PREGNANCY_FORAGE_FRACTION of pregnancy
    can_preg_forage = (~is_pregnant) | (s['pregnancy_hours'] < ref.PREGNANCY_FORAGE_FRACTION * s['pregnancy_target'])
    forage_base = walking & (~is_in_camp) & can_preg_forage
    hunt_ok   = forage_base & (age >= ref.HUNT_AGE_YEARS)
    fish_ok   = forage_base & (age >= ref.FISH_AGE_YEARS)
    gather_ok = forage_base & (age >= ref.GATHER_AGE_YEARS)
    # generic agent_forage (mutation-pool only) keeps the unified threshold
    forage_ok = forage_base & (age >= ref.FORAGE_AGE_YEARS)

    # --- non-sleep visit resets sleep streak (mirrors step() in main.py) ---
    not_sleep = walking & (action != A_SLEEP)
    s['sleep'] = xp.where(not_sleep, False, sleep)
    s['consecutive_sleep'] = xp.where(not_sleep, 0, cs)

    # --- agent_sleep ---
    # Sleeping in the wild is the dominant cause of death in this sim
    # (death_mask: exposed = sleep & ~is_in_camp). So when a wanderer
    # fires sleep, we redirect: set join_camp_request instead and skip
    # setting sleep=True. The current hour's join_camp_phase will scoop
    # them into a camp (or fall back to make if none exist); next hour
    # they're affiliated and sleep normally.
    m = walking & (action == A_SLEEP)
    m_wanderer = m & (s['camp_id'] < 0)
    s['join_camp_request'] |= m_wanderer
    m_aff = m & (s['camp_id'] >= 0)
    new_cs = xp.where(s['sleep'], s['consecutive_sleep'] + 1, 1)
    s['consecutive_sleep'] = xp.where(m_aff, new_cs, s['consecutive_sleep'])
    s['sleep'] = xp.where(m_aff, True, s['sleep'])
    s['is_in_camp'] = xp.where(m_aff, True, s['is_in_camp'])
    recovered = m_aff & (s['consecutive_sleep'] > ref.SLEEP_WARMUP)
    s['tired'] = xp.where(recovered, xp.maximum(0.0, s['tired'] - ref.SLEEP_RECOVERY), s['tired'])

    # --- agent_eat ---
    m = walking & (action == A_EAT) & (s['cache'] > 0)
    amt = xp.minimum(ref.EAT_RATE, s['cache'])
    s['cache']  = xp.where(m, s['cache'] - amt, s['cache'])
    s['hunger'] = xp.where(m, xp.maximum(0.0, s['hunger'] - amt), s['hunger'])

    # Per-cell resource access. `resources` is a numpy (W, H, 3) array where
    # axis-2 indexes [0=hunt, 1=fish, 2=gather]. R/K for each agent is taken
    # from the agent's current cell; extraction is scatter-subtracted back
    # into the same cell. Each forage event mutates only the agent's cell.
    if resources is not None:
        ax = to_host(s['x']) if USE_GPU else np.asarray(s['x'])
        ay = to_host(s['y']) if USE_GPU else np.asarray(s['y'])
        K_per = np.array([ref.RESOURCE_K_HUNT_PER_CELL,
                           ref.RESOURCE_K_FISH_PER_CELL,
                           ref.RESOURCE_K_GATHER_PER_CELL], dtype=np.float64)
        floor_per = K_per * ref.MIN_RESOURCE_FRAC
        # per-agent (R/K) for each kind, dtype float32 vector aligned with N
        rk_hunt   = (resources[ax, ay, 0] / K_per[0]).astype(np.float32)
        rk_fish   = (resources[ax, ay, 1] / K_per[1]).astype(np.float32)
        rk_gather = (resources[ax, ay, 2] / K_per[2]).astype(np.float32)
        rk_hunt   = xp.asarray(rk_hunt)
        rk_fish   = xp.asarray(rk_fish)
        rk_gather = xp.asarray(rk_gather)
    else:
        rk_hunt = rk_fish = rk_gather = xp.ones(N, dtype=xp.float32)

    def _scatter_extract(kind_idx, extracted_xp):
        """Scatter-subtract per-agent extracted amounts back into the (W, H)
        pool, clipped at the MIN_RESOURCE_FRAC floor."""
        if resources is None:
            return
        ext = to_host(extracted_xp) if USE_GPU else np.asarray(extracted_xp)
        # Use np.subtract.at for the scatter (handles multiple agents/same cell)
        np.subtract.at(resources[..., kind_idx], (ax, ay), ext)
        # Clip floor in place — but only on passable cells; impassable cells
        # must stay at zero (no phantom regrowth in unreachable terrain).
        floor_val = K_per[kind_idx] * ref.MIN_RESOURCE_FRAC
        np.maximum(resources[..., kind_idx], floor_val, out=resources[..., kind_idx])
        resources[..., kind_idx][~ref.TERRAIN_MASK] = 0.0

    # --- agent_hunt: success prob scaled by per-cell R/K; yield seasonal only ---
    # season_val may be scalar OR per-agent (N,) — geographical seasons make
    # it position-dependent. Arithmetic broadcasts either way. Time-of-day
    # factor is per-agent (local hour depends on x → timezone).
    x_host = to_host(s['x']) if USE_GPU else np.asarray(s['x'])
    local_h_np = ref.local_hour(x_host, int(hour))
    hunt_tod   = xp.asarray(ref.hunt_time_factor(local_h_np),   dtype=xp.float32)
    fish_tod   = xp.asarray(ref.fish_time_factor(local_h_np),   dtype=xp.float32)
    gather_tod = xp.asarray(ref.gather_time_factor(local_h_np), dtype=xp.float32)
    hunt_factor = (ref.HUNT_WINTER_FACTOR
                   + (1.0 - ref.HUNT_WINTER_FACTOR) * season_val)
    # Time-of-day modulates hunt success probability (crepuscular).
    hunt_p_per = float(ref.HUNT_SUCCESS_P) * rk_hunt * hunt_tod   # (N,)
    m = (action == A_HUNT) & hunt_ok & (rng.random(N) < hunt_p_per)
    if m.any():
        old_cache = s['cache'].copy()
        new_cache = xp.minimum(ref.CACHE_LIMIT,
                                xp.maximum(s['cache'], ref.HUNT_YIELD * hunt_factor))
        s['cache'] = xp.where(m, new_cache, s['cache'])
        extracted = xp.where(m, xp.maximum(0, s['cache'] - old_cache), 0.0)
        _scatter_extract(0, extracted)

    # --- agent_fish: success rate constant, yield scales with per-cell R/K ---
    fish_seasonal = (ref.FISH_WINTER_FACTOR
                     + (1.0 - ref.FISH_WINTER_FACTOR) * season_val)
    # Fish yield modulated by midday peak (daytime visibility on water).
    fish_yield_per = ref.FISH_YIELD * fish_seasonal * rk_fish * fish_tod   # (N,)
    m = (action == A_FISH) & fish_ok & (rng.random(N) < ref.FISH_SUCCESS_P)
    if m.any():
        old_cache = s['cache'].copy()
        s['cache'] = xp.where(m, xp.minimum(ref.CACHE_LIMIT,
                                             s['cache'] + fish_yield_per),
                              s['cache'])
        extracted = xp.where(m, xp.maximum(0, s['cache'] - old_cache), 0.0)
        _scatter_extract(1, extracted)

    # --- agent_gather: success rate constant, yield scales with per-cell R/K ---
    gather_seasonal = (ref.GATHER_WINTER_FACTOR
                       + (1.0 - ref.GATHER_WINTER_FACTOR) * season_val)
    # Gather yield modulated by morning peak (h≈9 — cool daylight hours).
    gather_yield_per = ref.GATHER_YIELD * gather_seasonal * rk_gather * gather_tod   # (N,)
    m = (action == A_GATHER) & gather_ok & (rng.random(N) < ref.GATHER_SUCCESS_P)
    if m.any():
        old_cache = s['cache'].copy()
        s['cache'] = xp.where(m, xp.minimum(ref.CACHE_LIMIT,
                                             s['cache'] + gather_yield_per),
                              s['cache'])
        extracted = xp.where(m, xp.maximum(0, s['cache'] - old_cache), 0.0)
        _scatter_extract(2, extracted)

    # --- agent_go_to_camp ---
    # Only meaningful for affiliated agents; wanderers (camp_id=-1) must use
    # join_camp to gain affiliation, so go_to_camp is a no-op for them.
    m = walking & (action == A_GO_TO_CAMP) & (s['camp_id'] >= 0)
    s['is_in_camp'] = xp.where(m, True, s['is_in_camp'])

    # --- spatial move actions: one cell per visit on a TORUS (wraps).
    # HARD RESTRICTION: only wanderers (camp_id < 0) can move. Affiliated
    # agents are bound to their camp's cell — they must agent_leave_camp
    # (which drops camp_id to -1) before they can traverse the grid.
    # Also gated by MOVE_COOLDOWN_HOURS so a wanderer moves at most every
    # ~2 hours. Cooldown is decremented each hour in run_vec's main loop.
    W = ref.GRID_W; H = ref.GRID_H
    can_move = (walking
                & (s['camp_id'] < 0)
                & (s['move_cooldown_hours'] == 0))
    mN = can_move & (action == A_MOVE_N)
    mS = can_move & (action == A_MOVE_S)
    mE = can_move & (action == A_MOVE_E)
    mW = can_move & (action == A_MOVE_W)
    # Terrain gating: moves blocked when destination cell is impassable.
    # TERRAIN_MASK is shape (W, H), bool, True = passable. Indexed [x, y].
    terrain = xp.asarray(ref.TERRAIN_MASK)
    ny_N = (s['y'] - 1) % H
    ny_S = (s['y'] + 1) % H
    nx_E = (s['x'] + 1) % W
    nx_W = (s['x'] - 1) % W
    pass_N = terrain[s['x'], ny_N]
    pass_S = terrain[s['x'], ny_S]
    pass_E = terrain[nx_E, s['y']]
    pass_W = terrain[nx_W, s['y']]
    mN = mN & pass_N
    mS = mS & pass_S
    mE = mE & pass_E
    mW = mW & pass_W
    s['y'] = xp.where(mN, ny_N, s['y'])
    s['y'] = xp.where(mS, ny_S, s['y'])
    s['x'] = xp.where(mE, nx_E, s['x'])
    s['x'] = xp.where(mW, nx_W, s['x'])
    moved = mN | mS | mE | mW
    s['move_cooldown_hours'] = xp.where(moved,
                                          xp.int8(ref.MOVE_COOLDOWN_HOURS),
                                          s['move_cooldown_hours'])

    # --- agent_leave_camp ---
    # Leaving camp = becoming an unaffiliated wanderer: drops camp_id to -1
    # and is_in_camp to False in one stroke (strict invariant: is_in_camp
    # iff camp_id != -1). Only affiliated agents can leave.
    # Adults (>= WATCH_AGE_YEARS) can leave anytime, EXCEPT during the
    # birth-window (late pregnancy + post-partum) — newborns can't survive
    # outside a camp, so the mom is held until the kid is past peak vulnerability.
    # Kids (FORAGE_AGE_YEARS .. WATCH_AGE_YEARS) can leave only if at least
    # one adult is currently a wanderer (tag along to forage).
    adults_outside = ((s['age'] >= ref.WATCH_AGE_YEARS) & ~s['is_in_camp']).any()
    bw = birth_window if birth_window is not None else birth_window_mask(s)
    adult_leave = (walking & (action == A_LEAVE_CAMP)
                   & (s['age'] >= ref.WATCH_AGE_YEARS)
                   & (s['camp_id'] >= 0)
                   & ~bw)
    kid_leave = (walking & (action == A_LEAVE_CAMP)
                 & (s['age'] >= ref.FORAGE_AGE_YEARS)
                 & (s['age'] < ref.WATCH_AGE_YEARS)
                 & (s['camp_id'] >= 0)
                 & adults_outside)
    m = adult_leave | kid_leave
    s['is_in_camp'] = xp.where(m, False, s['is_in_camp'])
    s['camp_id']    = xp.where(m, -1,    s['camp_id'])

    # --- talk / listen / propose_mate / gift / watch_children: just set flags ---
    s['talk_request']   |= walking & (action == A_TALK)   & s['is_in_camp']
    s['listen_request'] |= walking & (action == A_LISTEN) & s['is_in_camp']
    # Pregnant females CAN mate — pregnancy doesn't preclude pair-bonding,
    # sexual signaling, or coalition-building, only conception. The
    # mating_phase still routes only non-pregnant females through the
    # reproductive path; matings between pregnant females and males just
    # update mating_matrix (social effect, no new pregnancy).
    mate_ok = walking & (action == A_PROPOSE_MATE) & (s['age'] >= ref.MATE_AGE_YEARS) & s['is_in_camp']
    s['mate_request']   |= mate_ok
    gift_ok = walking & (action == A_GIFT) & s['is_in_camp'] & (s['cache'] > 0)
    s['gift_request']   |= gift_ok
    s['is_watching']    |= walking & (action == A_WATCH_CHILDREN)
    # camp-management actions fire regardless of in_camp — foragers can defect
    s['make_camp_request'] |= walking & (action == A_MAKE_CAMP)
    s['join_camp_request'] |= walking & (action == A_JOIN_CAMP)

    # --- deposit (per-camp; each agent's deposit goes to their own camp_id) ---
    dep_m = walking & (action == A_DEPOSIT) & s['is_in_camp']
    dep_amt = xp.where(dep_m, xp.minimum(ref.DEPOSIT_AMOUNT, s['cache']), 0.0)
    s['cache'] -= dep_amt
    s['net_debt_flow'] += dep_amt
    if dep_m.any():
        # bin deposits by camp_id
        active = dep_m & (dep_amt > 0)
        if active.any():
            cids = s['camp_id'][active]
            amts = dep_amt[active]
            uniq, inv = xp.unique(cids, return_inverse=True)
            per_camp = xp.bincount(inv, weights=amts.astype(xp.float64))
            for cid, dx in zip(uniq, per_camp):
                cid = int(cid)
                camp_caches[cid] = camp_caches.get(cid, 0.0) + float(dx)

    # --- withdraw (pro-rata per camp; vectorized via bincount) ---
    w_m = walking & (action == A_WITHDRAW) & s['is_in_camp']
    if w_m.any():
        headroom = xp.maximum(0.0, ref.CACHE_LIMIT - s['cache'])
        want = xp.where(w_m, xp.minimum(ref.WITHDRAW_AMOUNT, headroom), 0.0)
        active = w_m & (want > 0)
        if active.any():
            w_idx = xp.flatnonzero(active)
            w_cids = s['camp_id'][w_idx]
            uniq, inv = xp.unique(w_cids, return_inverse=True)
            # per-camp total demand and available supply (O(C) lookups)
            per_camp_want = xp.bincount(inv, weights=want[w_idx].astype(xp.float64))
            per_camp_avail = xp.array([camp_caches.get(int(c), 0.0)
                                        for c in uniq], dtype=xp.float64)
            per_camp_ratio = xp.minimum(1.0, per_camp_avail / xp.maximum(per_camp_want, 1e-12))
            # expand back to per-agent ratio in a single vector op
            agent_ratio = per_camp_ratio[inv].astype(xp.float32)
            got_active = want[w_idx] * agent_ratio
            s['cache'][w_idx] += got_active
            s['net_debt_flow'][w_idx] -= got_active
            # subtract per-camp totals (O(C) dict updates only)
            got_per_camp = xp.bincount(inv, weights=got_active.astype(xp.float64))
            for i, c in enumerate(uniq):
                camp_caches[int(c)] = float(per_camp_avail[i] - got_per_camp[i])

    # --- agent_idle: hunger += IDLE_HUNGER (which is 0 in current code) ---
    if ref.IDLE_HUNGER:
        m = walking & (action == A_IDLE)
        s['hunger'] = xp.where(m, s['hunger'] + ref.IDLE_HUNGER, s['hunger'])

    # make_camp / join_camp action effects are deferred to dedicated phases
    # (see make_camp_phase / join_camp_phase below). _apply_actions only sets
    # the request flags above.

    # --- agent_forage (+FORAGE_HUNGER; not in canonical but in mutation pool) ---
    forage_id = ACTION_FN_TO_ID.get(getattr(ref, 'agent_forage', None))
    if forage_id is not None:
        m = walking & (action == forage_id) & forage_ok & (rng.random(N) < ref.FORAGE_SUCCESS_P)
        s['cache'] = xp.where(m, xp.minimum(ref.CACHE_LIMIT, s['cache'] + ref.FORAGE_YIELD), s['cache'])

    return camp_caches


# ---------------------------------------------------------------------------
# Vectorized step (one hour, all agents)
# ---------------------------------------------------------------------------

def metabolic_scale_vec(age):
    """Mirror ref.metabolic_scale: 0→METABOLIC_ADULT_AGE is 12.5%→100%, adult=100%."""
    ma = ref.METABOLIC_ADULT_AGE
    k = ref.KID_METAB_FRAC
    return xp.where(age >= ma, 1.0, k + (age / ma) * (1.0 - k)).astype(xp.float32)


def step_all(graphs, s, hour, season_val, camp_caches, rng, n_nodes_arr=None,
             resources=None, budget=MAX_ACTION, camp_positions=None):
    """Walk the graph for all N agents in parallel for one hour.

    graphs: dict of (N, M, ...) and (N, M) arrays from stack_graphs.
    s: dict of per-agent state arrays.
    hour: int 0..23.
    camp_caches: dict {camp_id: float} of per-camp larders (mutated in place).
    rng: xp.random.Generator.
    n_nodes_arr: (N,) int array; if provided, cur landing outside [0, n_nodes)
        is snapped back to slot 0 (safety against dead/empty fall-throughs).
    Returns updated camp_caches.
    """
    N = s['hunger'].shape[0]
    if N == 0:
        return camp_caches
    ar = xp.arange(N)

    # forced-sleep agents skip the walk entirely — they don't fire actions,
    # don't pay per-visit hunger, and their cur stays put. Regular sleeping
    # agents (action=A_SLEEP via their graph) still walk.
    walking = s['forced_sleep_hours'] == 0

    # persistent program counter — agents pick up where they left off
    cur = s['cur'].astype(xp.intp)
    if n_nodes_arr is not None:
        cur = xp.where(cur < n_nodes_arr, cur, 0)
    visit_cost = ref.NODE_VISIT_HUNGER * metabolic_scale_vec(s['age'])
    # Birth-window mask only changes between hours (pregnancy_hours ticks
    # once per hour in births_step, postpartum_hours ticks once per hour).
    # Compute once here instead of every visit inside _apply_actions.
    bw = birth_window_mask(s)

    # Vision: camp positions don't change inside the per-visit loop (make/
    # join/cleanup all run between hours), so build the (W,H) cell-occupancy
    # bool once per hour and use it to compute per-agent direction-distances.
    # The agent's own (x, y) DOES change per visit (via move actions), so
    # we'll recompute vision per visit but the cell-occupancy grid stays
    # fixed across the budget loop.
    W = int(ref.GRID_W); H = int(ref.GRID_H)
    has_camp = xp.zeros((W, H), dtype=bool)
    if camp_positions:
        for cid, pos in camp_positions.items():
            cx, cy = pos
            if 0 <= cx < W and 0 <= cy < H:
                has_camp[int(cx), int(cy)] = True

    for _ in range(budget):
        # increment hunger by visit cost (only walking agents)
        s['hunger'] = xp.where(walking, s['hunger'] + visit_cost, s['hunger'])

        # Cultural-memory refresh: every visit to a node resets its freshness
        # counter. Only walking agents make progress through their chain so
        # only their freshness updates here (forced-sleepers don't tick).
        if 'node_freshness' in s:
            s['node_freshness'][ar[walking], cur[walking]] = ref.FRESHNESS_MAX

        # rebuild state matrix each visit (action effects mutate state)
        vision = _vision_distances(s, has_camp)
        state_mat = _build_state_matrix(s, hour, season_val, N, rng=rng,
                                          vision=vision)

        # gather current node's seq across agents
        sv = graphs['seq_var'][ar, cur]            # (N, S)
        so = graphs['seq_op'][ar, cur]
        svl = graphs['seq_value'][ar, cur]
        sl  = graphs['seq_links'][ar, cur]
        sln = graphs['seq_len'][ar, cur]

        cond = _eval_seqs(sv, so, svl, sl, sln, state_mat)  # (N,) bool

        action = xp.where(cond,
                          graphs['true_action'][ar, cur],
                          graphs['false_action'][ar, cur])

        _apply_actions(action, s, camp_caches, rng, walking=walking,
                       season_val=season_val, resources=resources,
                       birth_window=bw, hour=hour)

        new_cur = xp.where(cond,
                           graphs['true_node'][ar, cur],
                           graphs['false_node'][ar, cur]).astype(xp.intp)
        # cur only advances for walking agents
        cur = xp.where(walking, new_cur, cur)
        # safety: out-of-bounds (empty / dead fall-through) → snap to slot 0
        if n_nodes_arr is not None:
            cur = xp.where(cur < n_nodes_arr, cur, 0)

    s['cur'] = cur.astype(xp.int16)
    return camp_caches


# ---------------------------------------------------------------------------
# Social ranking — vectorized
# ---------------------------------------------------------------------------

# Cached per active xp module like _TT
_FN_CACHE = {}
def _FN():
    mod_id = id(xp)
    if mod_id not in _FN_CACHE:
        _FN_CACHE[mod_id] = xp.array(ref.FEATURE_NORM, dtype=xp.float32)
    return _FN_CACHE[mod_id]
FN = _FN()  # initial seed


def features_vec(s):
    """(N, 8) base feature matrix — per-agent only. The 9th feature
    (matings_with_target) is observer-target pairwise and is added at
    score time via _expand_with_mating()."""
    fn = _FN()
    F = xp.empty((s['hunger'].shape[0], 8), dtype=xp.float32)
    F[:, 0] = s['net_debt_flow'] * fn[0]
    F[:, 1] = s['hunger']        * fn[1]
    F[:, 2] = s['tired']         * fn[2]
    F[:, 3] = xp.where(s['is_female'],    1.0, -1.0).astype(xp.float32) * fn[3]
    F[:, 4] = s['cache']         * fn[4]
    F[:, 5] = s['age']           * fn[5]
    F[:, 6] = xp.where(s['is_pregnant'],  1.0, -1.0).astype(xp.float32) * fn[6]
    F[:, 7] = xp.where(s['is_menopausal'],1.0, -1.0).astype(xp.float32) * fn[7]
    return F


def _expand_with_pairwise(F_base, mating_col, gift_col, family_col,
                          camp_cache_col, opinion_col):
    """Append the five pairwise feature columns to F_base.
    F_base: (..., 8); each col: (...) same leading dims; returns (..., 13)."""
    m = (mating_col * xp.float32(ref.MATING_NORM))[..., None]
    g = (gift_col * xp.float32(ref.GIFT_NORM))[..., None]
    fam = (family_col * xp.float32(ref.FAMILY_NORM))[..., None]
    cc = (camp_cache_col * xp.float32(ref.CAMP_CACHE_NORM))[..., None]
    op = (opinion_col * xp.float32(ref.OPINION_NORM))[..., None]
    return xp.concatenate([F_base, m, g, fam, cc, op], axis=-1)


def _agent_optimal_dirs(weights, Q, n_iters=5):
    """For each agent, the unit vector f* in feature space that maximizes
    their ranking score wᵀf + fᵀQf on the unit ball ||f||=1. Computed via
    projected (normalized) gradient ascent: start at w/||w||, iterate
    f ← (w + 2Qf) / ||w + 2Qf||. Converges to a local max in a few steps.
    Returns (N, F) unit vectors — these are the 'ideal target profiles'
    each agent's scorer maximally rewards.
    """
    f = weights / xp.maximum(xp.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    for _ in range(n_iters):
        grad = weights + 2.0 * xp.einsum('nij,nj->ni', Q, f)
        n = xp.linalg.norm(grad, axis=1, keepdims=True)
        f = grad / xp.maximum(n, 1e-12)
    return f.astype(xp.float32)


def _opinion_cosine_pairs(opt_dirs, obs_idx, peer_idx):
    """Cosine sim between observer's optimal direction and each peer's.
    opt_dirs: (N, F) unit vectors; obs_idx: (A,); peer_idx: (A, K) → (A, K)."""
    d_obs = opt_dirs[obs_idx]                              # (A, F)
    d_peer = opt_dirs[peer_idx]                            # (A, K, F)
    return xp.einsum('af,akf->ak', d_obs, d_peer).astype(xp.float32)


def _opinion_cosine_flat(opt_dirs, observer, pool):
    """Flat 1D version: one observer scalar idx vs a pool array."""
    d_obs = opt_dirs[observer]                             # (F,)
    d_peer = opt_dirs[pool]                                # (P, F)
    return (d_peer @ d_obs).astype(xp.float32)


# legacy alias kept for any external callers (not used internally anymore)
def _expand_with_mating(F_base, mating_col):
    zeros = xp.zeros_like(mating_col, dtype=xp.float32)
    return _expand_with_pairwise(F_base, mating_col, zeros, zeros, zeros, zeros)


def social_score_self(weights, Q, F_base):
    """Score each observer against self. Pairwise mating + gift + family-sim +
    camp-cache features are 0 — self-pairwise comparisons aren't meaningful
    and including them distorts the self-rank delta."""
    N = F_base.shape[0]
    zeros = xp.zeros(N, dtype=xp.float32)
    F = _expand_with_pairwise(F_base, zeros, zeros, zeros, zeros, zeros)
    linear = (weights * F).sum(axis=1)
    quad = xp.einsum('nfg,nf,ng->n', Q, F, F)
    return linear + quad


def _bidir_pair_value(matrix, obs_idx, peer_idx):
    """Sum of bidirectional flow between observer and peer in a directional
    matrix: matrix[obs, peer] + matrix[peer, obs]. Used for both gift_matrix
    (directional) and mating_matrix (already symmetric, but harmless to sum)."""
    if matrix is None or obs_idx is None:
        return xp.zeros(peer_idx.shape, dtype=xp.float32)
    a = matrix[obs_idx[:, None], peer_idx]
    b = matrix[peer_idx, obs_idx[:, None]]
    return (a + b).astype(xp.float32)


def social_score_pairs(weights, Q, F_base, peer_idx,
                       mating_matrix=None, gift_matrix=None, obs_idx=None,
                       crests=None, camp_id=None, camp_caches=None):
    """Score observers against K peers. peer_idx (A,K), obs_idx (A,) — actual
    observer indices for matrix lookups. crests (N, D) gives per-agent unit-vec
    family crests; family-similarity column = dot(obs_crest, peer_crest).
    camp_id (N,) + camp_caches dict yields target_camp_cache feature value."""
    f_peer_base = F_base[peer_idx]                               # (A, K, 8)
    if mating_matrix is not None and obs_idx is not None:
        mating_col = mating_matrix[obs_idx[:, None], peer_idx].astype(xp.float32)
    else:
        mating_col = xp.zeros(peer_idx.shape, dtype=xp.float32)
    if gift_matrix is not None and obs_idx is not None:
        gift_col = _bidir_pair_value(gift_matrix, obs_idx, peer_idx)
    else:
        gift_col = xp.zeros(peer_idx.shape, dtype=xp.float32)
    if crests is not None and obs_idx is not None:
        family_col = xp.einsum('ad,akd->ak',
                               crests[obs_idx], crests[peer_idx]).astype(xp.float32)
    else:
        family_col = xp.zeros(peer_idx.shape, dtype=xp.float32)
    if camp_id is not None and camp_caches is not None:
        # look up each peer's camp_cache via their camp_id
        camp_cache_col = _camp_cache_lookup(camp_id[peer_idx], camp_caches)
    else:
        camp_cache_col = xp.zeros(peer_idx.shape, dtype=xp.float32)
    if obs_idx is not None:
        opt_dirs = _agent_optimal_dirs(weights, Q)
        opinion_col = _opinion_cosine_pairs(opt_dirs, obs_idx, peer_idx)
    else:
        opinion_col = xp.zeros(peer_idx.shape, dtype=xp.float32)
    f_peer = _expand_with_pairwise(f_peer_base, mating_col, gift_col,
                                    family_col, camp_cache_col, opinion_col)
    w_obs = weights[obs_idx] if obs_idx is not None else weights
    Q_obs = Q[obs_idx] if obs_idx is not None else Q
    linear = xp.einsum('af,akf->ak', w_obs, f_peer)
    quad   = xp.einsum('afg,akf,akg->ak', Q_obs, f_peer, f_peer)
    return linear + quad


def _camp_cache_lookup(cids_arr, camp_caches):
    """Map an array of camp_ids to a same-shape array of camp_cache values.
    Missing camps default to 0.0."""
    cids_arr = xp.asarray(cids_arr)
    flat = cids_arr.ravel()
    out = xp.zeros(flat.shape, dtype=xp.float32)
    for i, cid in enumerate(flat):
        out[i] = float(camp_caches.get(int(cid), 0.0))
    return out.reshape(cids_arr.shape)


def adaptive_rates(self_scores, peer_means):
    delta = xp.clip(self_scores - peer_means, -50.0, 50.0)
    sig = 1.0 / (1.0 + xp.exp(-delta))
    log_rate = (1.0 - sig) * ref._LOG_HIGH_MUT + sig * ref._LOG_LOW_MUT
    mut = xp.exp(log_rate).astype(xp.float32)
    adopt = ((1.0 - sig) * ref.ADAPTIVE_ADOPT_HIGH +
             sig * ref.ADAPTIVE_ADOPT_LOW).astype(xp.float32)
    return mut, adopt


def update_adaptive_rates(s, weights, Q, rng, mating_matrix=None,
                           gift_matrix=None, crests=None, camp_caches=None,
                           K=ref.K_SAMPLE):
    """Each agent's mut/adopt rates are calibrated against a peer SAMPLE.

    Peers are drawn from the agent's OWN CAMP — your relevant social context
    is the people you actually interact with daily. Singletons (camp with no
    other members) fall back to sampling from the global population.
    """
    N = s['hunger'].shape[0]
    if N <= 1:
        return
    F = features_vec(s)
    self_scores = social_score_self(weights, Q, F)
    K = min(K, N)

    # Per-agent peer pool: same-camp members (excluding self) where possible,
    # global fallback for singletons.
    camp_id = s['camp_id']
    # Group agents by camp once
    camp_members = {}
    for i, c in enumerate(camp_id):
        camp_members.setdefault(int(c), []).append(i)
    camp_members = {c: xp.array(v, dtype=xp.intp) for c, v in camp_members.items()}
    all_idx = xp.arange(N)

    peer_idx = xp.empty((N, K), dtype=xp.intp)
    for i in range(N):
        same_camp = camp_members[int(camp_id[i])]
        same_camp = same_camp[same_camp != i]
        if same_camp.size == 0:
            # singleton: fall back to global (still excluding self)
            pool = all_idx[all_idx != i]
        else:
            pool = same_camp
        # sample with replacement so we always fill K slots, even for tiny camps
        peer_idx[i] = pool[rng.integers(0, pool.size, size=K)]

    obs_idx = all_idx
    peer_scores = social_score_pairs(weights, Q, F, peer_idx,
                                      mating_matrix=mating_matrix,
                                      gift_matrix=gift_matrix,
                                      obs_idx=obs_idx,
                                      crests=crests,
                                      camp_id=s.get('camp_id'),
                                      camp_caches=camp_caches)
    peer_means = peer_scores.mean(axis=1)
    mut, adopt = adaptive_rates(self_scores, peer_means)
    s['mut_rate']   = mut
    s['adopt_rate'] = adopt


# ---------------------------------------------------------------------------
# Watching phase — vectorized
# ---------------------------------------------------------------------------

def watching_phase(s, camp_caches=None):
    """Process the watching phase per-camp. Watchers feed kids from their own
    personal cache (camp larder is irrelevant); each camp's watchers only feed
    that camp's kids. `camp_caches` is accepted for signature compatibility
    but unused — watchers donate from `s['cache']` directly. Returns it
    unchanged."""
    fa = ref.WATCH_AGE_YEARS
    is_kid = s['age'] < fa
    is_w_all = s['is_watching']
    in_camp_w_all  = is_w_all & s['is_in_camp']
    out_camp_w_all = is_w_all & ~s['is_in_camp']
    in_camp_k_all  = is_kid & s['is_in_camp']
    out_camp_k_all = is_kid & ~s['is_in_camp']

    # Vectorized per-camp feed: aggregate watcher-pool and kid-demand per camp
    # via bincount, settle each camp's ratio, then scatter per-agent.
    camp_ids = s['camp_id']
    N = camp_ids.shape[0]

    def _feed_pool_vec(watchers_mask, kids_mask):
        if not watchers_mask.any() or not kids_mask.any():
            return
        # camps that have ANY watcher or kid (intersection: only need camps with both)
        w_cids = camp_ids[watchers_mask]
        k_cids = camp_ids[kids_mask]
        uniq_cids, c_inv = xp.unique(xp.concatenate([w_cids, k_cids]),
                                       return_inverse=True)
        n_c = uniq_cids.size
        # map per-agent to camp-index for fast bincount per camp
        # build per-agent → camp-index for watchers and kids separately
        w_to_c = xp.searchsorted(uniq_cids, w_cids)
        k_to_c = xp.searchsorted(uniq_cids, k_cids)
        # per-camp watcher cache pool
        watcher_cache = s['cache'][watchers_mask].astype(xp.float64)
        pool_per_camp = xp.bincount(w_to_c, weights=watcher_cache, minlength=n_c)
        # per-camp kid demand (capped at each kid's hunger)
        kid_hunger = s['hunger'][kids_mask].astype(xp.float64)
        kid_demand = xp.minimum(float(ref.WATCH_FEED), kid_hunger)
        demand_per_camp = xp.bincount(k_to_c, weights=kid_demand, minlength=n_c)
        # available per camp
        avail_per_camp = xp.minimum(demand_per_camp, pool_per_camp)
        # safe ratios with division-by-zero guards
        donate_frac_per_camp = xp.where(pool_per_camp > 0,
                                          avail_per_camp / xp.maximum(pool_per_camp, 1e-12),
                                          0.0)
        supply_ratio_per_camp = xp.where(demand_per_camp > 0,
                                           avail_per_camp / xp.maximum(demand_per_camp, 1e-12),
                                           0.0)
        # apply: scale watcher caches down
        w_donate = donate_frac_per_camp[w_to_c].astype(xp.float32)
        s['cache'][watchers_mask] = (s['cache'][watchers_mask] * (1.0 - w_donate))
        # apply: reduce kid hunger
        k_ratio = supply_ratio_per_camp[k_to_c].astype(xp.float32)
        k_feed = kid_demand.astype(xp.float32) * k_ratio
        s['hunger'][kids_mask] = xp.maximum(0.0, s['hunger'][kids_mask] - k_feed)

    _feed_pool_vec(in_camp_w_all,  in_camp_k_all)
    _feed_pool_vec(out_camp_w_all, out_camp_k_all)

    # unwatched-hours bookkeeping — vectorized via per-camp "has watcher?" lookup
    # build per-camp watcher presence flags (boolean) using bincount of watcher mask
    is_sleep = s['sleep']
    # in-camp kids: zero unwatched if there's any in-camp watcher in their camp
    if in_camp_w_all.any() or in_camp_k_all.any():
        # presence: bool array indexed by camp_id (using dict-like inverse)
        kid_cids = camp_ids[in_camp_k_all]
        if kid_cids.size:
            # set of camps with at least one in-camp watcher
            has_w = xp.isin(kid_cids, camp_ids[in_camp_w_all])
            kid_idx = xp.flatnonzero(in_camp_k_all)
            # zero unwatched for kids in watched camps OR sleeping
            zero_mask = has_w | is_sleep[kid_idx]
            inc_mask = ~zero_mask
            s['unwatched_hours'][kid_idx[zero_mask]] = 0
            s['unwatched_hours'][kid_idx[inc_mask]] += 1
    if out_camp_w_all.any() or out_camp_k_all.any():
        kid_cids = camp_ids[out_camp_k_all]
        if kid_cids.size:
            has_w = xp.isin(kid_cids, camp_ids[out_camp_w_all])
            kid_idx = xp.flatnonzero(out_camp_k_all)
            zero_mask = has_w | is_sleep[kid_idx]
            inc_mask = ~zero_mask
            s['unwatched_hours'][kid_idx[zero_mask]] = 0
            s['unwatched_hours'][kid_idx[inc_mask]] += 1

    s['is_watching'][:] = False
    return camp_caches


# ---------------------------------------------------------------------------
# Communication phase — vectorized assimilation via flat-row copy
# ---------------------------------------------------------------------------

def communication_phase(s, graphs, n_nodes_arr, weights, Q, rng,
                        roots=None, mating_matrix=None, gift_matrix=None,
                        crests=None, camp_caches=None, K=ref.K_SAMPLE):
    """Talkers expose a random node from their graph; listeners (per their
    adopt_rate) softmax-pick a talker by social_score and modify their own
    graph: APPEND a new node spliced into one rewired edge if room exists,
    else REPLACE a random node's content.

    If `roots` (list of object graphs) is provided, the same change is mirrored
    into the object graph so subsequent mutation re-flattens preserve adoptions.

    Returns (n_talks, n_listens, n_adopts).
    """
    talker_mask = s['is_in_camp'] & s['talk_request']
    listener_mask = s['is_in_camp'] & s['listen_request']
    s['talk_request'][:] = False
    s['listen_request'][:] = False

    talker_idx = xp.flatnonzero(talker_mask)
    listener_idx = xp.flatnonzero(listener_mask)
    n_talks = int(talker_idx.size)
    n_listens = int(listener_idx.size)
    if n_talks == 0 or n_listens == 0:
        return n_talks, n_listens, 0

    # which node each talker exposes
    talker_node = (rng.random(n_talks) * n_nodes_arr[talker_idx]).astype(xp.int32)

    # adopt-rate gate per listener
    adopt = s['adopt_rate'][listener_idx]
    rolls = rng.random(n_listens)
    adopters = listener_idx[rolls < adopt]
    if adopters.size == 0:
        return n_talks, n_listens, 0

    # K-sample talkers per adopter
    K = min(K, n_talks)
    chosen_t_local = rng.integers(0, n_talks, size=(adopters.size, K))   # idx into talker_idx
    chosen_t_global = talker_idx[chosen_t_local]                         # (A, K)

    # softmax-weight talkers by social_score(adopter, talker) including
    # pairwise mating-history and gift-history features
    F_base = features_vec(s)
    f_peer_base = F_base[chosen_t_global]                                # (A, K, 8)
    if mating_matrix is not None:
        mating_col = mating_matrix[adopters[:, None], chosen_t_global].astype(xp.float32)
    else:
        mating_col = xp.zeros(chosen_t_global.shape, dtype=xp.float32)
    if gift_matrix is not None:
        gift_col = _bidir_pair_value(gift_matrix, adopters, chosen_t_global)
    else:
        gift_col = xp.zeros(chosen_t_global.shape, dtype=xp.float32)
    if crests is not None:
        family_col = xp.einsum('ad,akd->ak',
                               crests[adopters], crests[chosen_t_global]).astype(xp.float32)
    else:
        family_col = xp.zeros(chosen_t_global.shape, dtype=xp.float32)
    if camp_caches is not None and 'camp_id' in s:
        camp_cache_col = _camp_cache_lookup(s['camp_id'][chosen_t_global], camp_caches)
    else:
        camp_cache_col = xp.zeros(chosen_t_global.shape, dtype=xp.float32)
    opt_dirs = _agent_optimal_dirs(weights, Q)
    opinion_col = _opinion_cosine_pairs(opt_dirs, adopters, chosen_t_global)
    f_peer = _expand_with_pairwise(f_peer_base, mating_col, gift_col,
                                    family_col, camp_cache_col, opinion_col)
    w_obs = weights[adopters]
    Q_obs = Q[adopters]
    linear = xp.einsum('af,akf->ak', w_obs, f_peer)
    quad   = xp.einsum('afg,akf,akg->ak', Q_obs, f_peer, f_peer)
    scores = linear + quad
    # Same-camp filter: mask out talkers in a different camp than the adopter.
    # Listeners only adopt from agents in the same camp.
    if 'camp_id' in s:
        adopter_camp = s['camp_id'][adopters][:, None]               # (A, 1)
        talker_camp = s['camp_id'][chosen_t_global]                  # (A, K)
        same_camp = (adopter_camp == talker_camp)
        scores = xp.where(same_camp, scores, -xp.float32(1e9))
        # adopters with NO same-camp talker in their K-sample have all-blocked rows
        any_eligible = same_camp.any(axis=1)
    else:
        any_eligible = xp.ones(adopters.size, dtype=bool)
    scores -= scores.max(axis=1, keepdims=True)
    p = xp.exp(scores); psum = p.sum(axis=1, keepdims=True)
    p = xp.where(psum > 0, p / psum, 0)
    cum = xp.cumsum(p, axis=1)
    r = rng.random((adopters.size, 1))
    pick = (r < cum).argmax(axis=1)                                      # (A,)
    # drop adopters who had no same-camp talker
    if not any_eligible.all():
        adopters = adopters[any_eligible]
        chosen_t_global = chosen_t_global[any_eligible]
        chosen_t_local = chosen_t_local[any_eligible]
        pick = pick[any_eligible]
    if adopters.size == 0:
        return n_talks, n_listens, 0
    src_agents = chosen_t_global[xp.arange(adopters.size), pick]
    src_nodes  = talker_node[chosen_t_local[xp.arange(adopters.size), pick]]

    # --- memetic weight interpolation ---
    # Each listener pulls (weights, Q) toward the chosen talker by
    #   α = mut_rate × adopt_rate × WEIGHT_TEACH_SCALE,
    # so agents who are individually open to mutation and to peer adoption
    # learn ranking-function preferences faster. Talker arrays are unchanged
    # (knowledge flows one way per event). Vectorized arrays are the source
    # of truth for weights/Q (object-graph .weights is only read during the
    # NEXT mutation_step, which downloads from these arrays first).
    alpha = (s['mut_rate'][adopters] * s['adopt_rate'][adopters]
             * ref.WEIGHT_TEACH_SCALE).astype(weights.dtype)
    delta_w = alpha[:, None] * (weights[src_agents] - weights[adopters])
    weights[adopters] += delta_w
    delta_Q = alpha[:, None, None] * (Q[src_agents] - Q[adopters])
    Q[adopters] += delta_Q

    # Subgraph adoption: BFS up to K_GRAFT nodes from the talker's exposed
    # node, deep-clone, splice the whole subgraph into one of the listener's
    # edges. Listeners with insufficient headroom (n_nodes + K > MAX_NODES)
    # just skip the adoption — no REPLACE. Cultural memory decay
    # (gc_dead_nodes) is responsible for reclaiming slots over time.

    # Pre-snapshot every talker's python node list (see note in old code).
    talker_cache = {}
    if roots is not None:
        for t in set(int(x) for x in src_agents):
            talker_cache[t] = ref.all_nodes(roots[t])

    n_adopted = 0
    if roots is not None:
        for li in range(adopters.size):
            adopter = int(adopters[li])
            src_a   = int(src_agents[li])
            src_n   = int(src_nodes[li])

            a_nodes = ref.all_nodes(roots[adopter])
            src_list = talker_cache.get(src_a) or []
            if not a_nodes or not src_list or src_n >= len(src_list):
                continue

            # --- BFS K nodes from src_n in talker's graph ---
            entry = src_list[src_n]
            bfs_order = [entry]
            bfs_set = {id(entry)}
            qi = 0
            while qi < len(bfs_order) and len(bfs_order) < K_GRAFT:
                cur_n = bfs_order[qi]; qi += 1
                for nxt in (cur_n.true_node, cur_n.false_node):
                    if nxt is None: continue
                    if id(nxt) in bfs_set: continue
                    bfs_set.add(id(nxt))
                    bfs_order.append(nxt)
                    if len(bfs_order) >= K_GRAFT: break

            # --- skip if no headroom (cap is hard; we don't replace) ---
            K_actual = len(bfs_order)
            if int(n_nodes_arr[adopter]) + K_actual > MAX_NODES:
                continue

            # --- pick splice point in listener BEFORE we clone, since
            # cloned subgraph's "exit" edges will rewire here ---
            existing_idx = int(rng.integers(0, len(a_nodes)))
            target_node = a_nodes[existing_idx]
            edge_is_true = (int(rng.integers(0, 2)) == 0)
            orig_target_node = target_node.true_node if edge_is_true else target_node.false_node

            # save cur python node for re-anchor
            old_cur = int(s['cur'][adopter])
            cur_pynode = a_nodes[old_cur] if old_cur < len(a_nodes) else None

            # --- deep-clone the BFS subgraph into the listener's graph ---
            # Map talker node id → freshly-built listener node
            clone_map = {}
            for src_n_obj in bfs_order:
                new_n = ref.node()
                if src_n_obj.seq is not None:
                    new_n.seq = ref.clone_seq(src_n_obj.seq)
                new_n.true_action  = src_n_obj.true_action
                new_n.false_action = src_n_obj.false_action
                new_n.true_end     = src_n_obj.true_end
                new_n.false_end    = src_n_obj.false_end
                new_n.lineage      = src_n_obj.lineage   # carry memetic lineage
                # edges set in second pass once all clones exist
                clone_map[id(src_n_obj)] = new_n
            for src_n_obj in bfs_order:
                new_n = clone_map[id(src_n_obj)]
                # internal edges (target in BFS set) → wired to the new clone.
                # external/exit edges (target outside the BFS set) → routed
                # to orig_target_node so the subgraph plugs into the
                # listener's chain past the splice point. Self-loops on
                # entry that pointed at None pass through unchanged.
                for side in ('true_node', 'false_node'):
                    tgt = getattr(src_n_obj, side)
                    if tgt is None:
                        setattr(new_n, side, None)
                    elif id(tgt) in clone_map:
                        setattr(new_n, side, clone_map[id(tgt)])
                    else:
                        setattr(new_n, side, orig_target_node)

            # --- splice subgraph entry into target_node's chosen edge ---
            subgraph_entry = clone_map[id(entry)]
            if edge_is_true:
                target_node.true_node = subgraph_entry
            else:
                target_node.false_node = subgraph_entry

            # --- reflatten + re-anchor freshness + cur ---
            n_nodes_arr[adopter] = reflatten_row(graphs, adopter, roots[adopter])
            new_nodes = ref.all_nodes(roots[adopter])
            reindex_freshness(s, adopter, a_nodes, new_nodes)
            if cur_pynode is not None:
                for i, n in enumerate(new_nodes):
                    if n is cur_pynode:
                        s['cur'][adopter] = i
                        break
            n_adopted += 1

    return n_talks, n_listens, n_adopted


# ---------------------------------------------------------------------------
# Mating phase — python-side over a small subset
# ---------------------------------------------------------------------------

def _softmax_pick(scores, rng):
    mx = scores.max()
    e = xp.exp(scores - mx)
    cum = xp.cumsum(e / e.sum())
    return int(min(xp.searchsorted(cum, rng.random()), len(scores) - 1))


def _score_pool_pairs(pool, F_base, weights, Q,
                        mating_matrix=None, gift_matrix=None, crests=None,
                        camp_id=None, camp_caches=None):
    """Compute the full (P, P) score matrix for all (observer, target) pairs
    in `pool`. score[i, j] = how observer pool[i] ranks target pool[j].

    Used for per-camp batched scoring in mating_phase / gift_phase — avoids
    the Python loop over P observers each doing P-element scoring."""
    P = int(pool.size)
    if P == 0:
        return xp.zeros((0, 0), dtype=xp.float32)
    # intrinsic target features — same across observers, broadcast
    f_tgt_base = F_base[pool]                                       # (P, 8)
    f_intrinsic = xp.broadcast_to(f_tgt_base[None, :, :], (P, P, 8))
    # pairwise feature columns (P_obs, P_tgt)
    if mating_matrix is not None:
        m_col = mating_matrix[pool[:, None], pool[None, :]].astype(xp.float32)
    else:
        m_col = xp.zeros((P, P), dtype=xp.float32)
    if gift_matrix is not None:
        g_col = (gift_matrix[pool[:, None], pool[None, :]] +
                  gift_matrix[pool[None, :], pool[:, None]]).astype(xp.float32)
    else:
        g_col = xp.zeros((P, P), dtype=xp.float32)
    if crests is not None:
        # cosine matrix: (P, D) @ (D, P)
        fam_col = (crests[pool] @ crests[pool].T).astype(xp.float32)
    else:
        fam_col = xp.zeros((P, P), dtype=xp.float32)
    if camp_id is not None and camp_caches is not None:
        # target_camp_cache depends only on target → broadcast over observers
        cc_per_target = xp.array(
            [float(camp_caches.get(int(camp_id[int(p)]), 0.0)) for p in pool],
            dtype=xp.float32)
        cc_col = xp.broadcast_to(cc_per_target[None, :], (P, P))
    else:
        cc_col = xp.zeros((P, P), dtype=xp.float32)
    # opinion cosine matrix: each agent's optimal-direction unit vector
    # vs each other's; (P, F) @ (F, P) since both are unit norm.
    d_pool = _agent_optimal_dirs(weights, Q)[pool]                    # (P, F)
    op_col = (d_pool @ d_pool.T).astype(xp.float32)                   # (P, P)
    m_col = m_col * xp.float32(ref.MATING_NORM)
    g_col = g_col * xp.float32(ref.GIFT_NORM)
    fam_col = fam_col * xp.float32(ref.FAMILY_NORM)
    cc_col = cc_col * xp.float32(ref.CAMP_CACHE_NORM)
    op_col = op_col * xp.float32(ref.OPINION_NORM)
    pairwise = xp.stack([m_col, g_col, fam_col, cc_col, op_col], axis=-1)   # (P, P, 5)
    f = xp.concatenate([f_intrinsic, pairwise], axis=-1)            # (P, P, 13)
    w_obs = weights[pool]                                            # (P, 13)
    Q_obs = Q[pool]                                                  # (P, 13, 13)
    linear = xp.einsum('of,otf->ot', w_obs, f)                       # (P_obs, P_tgt)
    quadratic = xp.einsum('ofg,otf,otg->ot', Q_obs, f, f)
    return (linear + quadratic).astype(xp.float32)


def _score_against_pool(observer, pool, F_base, weights, Q,
                          mating_matrix, gift_matrix=None, crests=None,
                          camp_id=None, camp_caches=None):
    """Score one observer against an array of candidate indices."""
    if pool.size == 0:
        return xp.zeros(0, dtype=xp.float32)
    f_pool_base = F_base[pool]                                          # (P, 8)
    if mating_matrix is not None:
        m_col = mating_matrix[observer, pool].astype(xp.float32)
    else:
        m_col = xp.zeros(pool.size, dtype=xp.float32)
    if gift_matrix is not None:
        g_col = (gift_matrix[observer, pool] + gift_matrix[pool, observer]).astype(xp.float32)
    else:
        g_col = xp.zeros(pool.size, dtype=xp.float32)
    if crests is not None:
        f_col = (crests[pool] @ crests[observer]).astype(xp.float32)
    else:
        f_col = xp.zeros(pool.size, dtype=xp.float32)
    if camp_id is not None and camp_caches is not None:
        cc_col = _camp_cache_lookup(camp_id[pool], camp_caches)
    else:
        cc_col = xp.zeros(pool.size, dtype=xp.float32)
    opt_dirs = _agent_optimal_dirs(weights, Q)
    op_col = _opinion_cosine_flat(opt_dirs, observer, pool)
    f_pool = _expand_with_pairwise(f_pool_base, m_col, g_col, f_col, cc_col, op_col)
    lin = (weights[observer] * f_pool).sum(1)
    qd  = xp.einsum('fg,kf,kg->k', Q[observer], f_pool, f_pool)
    return lin + qd


def mating_phase(s, weights, Q, rng, mating_matrix=None, gift_matrix=None,
                 crests=None, camp_caches=None, K=ref.K_SAMPLE):
    """Returns (n_resolved, dad_pairs, mating_events).

    Same-sex pairs are allowed; only opposite-sex pairs with a fertile mom
    can produce pregnancy. Candidate selection uses TOP-K by the observer's
    own ranking function (no longer random sampling).

    Incest gate is applied as a PRE-FILTER on the candidate pool, not as a
    post-pick rejection — the softmax is taken over the available (non-kin)
    set, so a proposer whose closest kin would have ranked highest still
    gets to mate with the best-ranked non-kin candidate.
    """
    # Only proposers physically present in their camp can mate.
    proposers = xp.flatnonzero(s['mate_request'] & s['is_in_camp'])
    s['mate_request'][:] = False
    if proposers.size <= 1:
        return 0, [], []

    is_f = s['is_female']; is_p = s['is_pregnant']; is_meno = s['is_menopausal']
    F_base = features_vec(s)
    n_resolved = 0
    dad_pairs = []
    mating_events = []
    camp_id = s.get('camp_id') if isinstance(s, dict) else None

    # Group proposers by CELL (x, y) rather than camp_id — co-location, not
    # affiliation, defines the mating pool. Different camps sharing a cell
    # can interbreed; same-camp members at different cells cannot.
    if 'x' in s and 'y' in s:
        proposer_cells = s['x'][proposers].astype(xp.int64) * 100000 + s['y'][proposers].astype(xp.int64)
    else:
        proposer_cells = (camp_id[proposers] if camp_id is not None
                          else xp.zeros(proposers.size, dtype=xp.int64))

    for cell_key in xp.unique(proposer_cells):
        camp_pool = proposers[proposer_cells == cell_key]
        P = camp_pool.size
        if P <= 1:
            continue
        # Full (P, P) score matrix in one batched call — score[i, j] = i's
        # ranking of j as observer→target.
        score_mat = _score_pool_pairs(camp_pool, F_base, weights, Q,
                                        mating_matrix=mating_matrix,
                                        gift_matrix=gift_matrix,
                                        crests=crests, camp_id=camp_id,
                                        camp_caches=camp_caches)
        # Mask self-pairings: nobody ranks themselves as a target.
        xp.fill_diagonal(score_mat, -xp.inf)
        # Kin gate: build the (P, P) cos-similarity matrix and mask kin out.
        if crests is not None:
            kin_mat = (crests[camp_pool] @ crests[camp_pool].T) > ref.FAMILY_INCEST_THRESHOLD
            score_mat = xp.where(kin_mat, -xp.float32(xp.inf), score_mat)

        # Per-proposer (row): top-K + softmax-pick.
        Keff = min(int(K), P - 1)
        # argpartition with -inf rows is fine; we'll filter rows with no
        # finite scores after.
        topk_idx = xp.argpartition(-score_mat, Keff - 1, axis=1)[:, :Keff]   # (P, K)
        row_idx = xp.arange(P)[:, None]
        topk_scores = score_mat[row_idx, topk_idx]                            # (P, K)
        # Rows with no valid (non -inf) entries → row has no available mate
        row_max = topk_scores.max(axis=1)                                     # (P,)
        valid_proposer = xp.isfinite(row_max)
        if not valid_proposer.any():
            continue
        # Stable softmax row-wise (set -inf entries to a number that gives 0)
        shifted = topk_scores - row_max[:, None]
        shifted = xp.where(xp.isneginf(topk_scores), -1e30, shifted)
        e = xp.exp(shifted)
        e_sum = e.sum(axis=1, keepdims=True)
        # Guard against div-by-zero on all-blocked rows
        e_sum = xp.where(e_sum > 0, e_sum, 1.0)
        probs = e / e_sum
        cum = xp.cumsum(probs, axis=1)
        rolls = rng.random((P, 1))
        picks = (rolls < cum).argmax(axis=1)                                  # (P,)
        chosen_local = topk_idx[xp.arange(P), picks]                          # (P,) — local idx into camp_pool

        # For each proposer p with valid_proposer[i]=True, the chosen partner
        # is camp_pool[chosen_local[i]]. Now compute the accept side: chosen's
        # accept probability for p, derived from chosen's row of score_mat.
        for i in range(P):
            if not valid_proposer[i]:
                continue
            ci_local = int(chosen_local[i])
            # chosen's full ranking of all proposers (this camp)
            chosen_row = score_mat[ci_local].copy()
            chosen_row[ci_local] = -xp.inf   # chosen doesn't pair with self
            # take chosen's top-K
            Kc = min(int(K), P - 1)
            chosen_topk = xp.argpartition(-chosen_row, Kc - 1)[:Kc]
            chosen_top_scores = chosen_row[chosen_topk]
            # ensure p (= camp_pool[i]) is in chosen's top-K — force-include
            # to match original semantics
            if i not in chosen_topk:
                # is p still eligible from chosen's perspective?
                if xp.isfinite(chosen_row[i]):
                    chosen_topk = xp.append(chosen_topk, i)
                    chosen_top_scores = xp.append(chosen_top_scores, chosen_row[i])
                else:
                    continue
            # softmax accept probability for chosen → p
            ct_max = chosen_top_scores.max()
            if not xp.isfinite(ct_max):
                continue
            ct_shift = chosen_top_scores - ct_max
            ct_shift = xp.where(xp.isneginf(chosen_top_scores), -1e30, ct_shift)
            ce = xp.exp(ct_shift); ce_sum = ce.sum()
            if ce_sum <= 0:
                continue
            p_pos_in_topk = xp.where(chosen_topk == i)[0]
            if p_pos_in_topk.size == 0:
                continue
            accept_prob = float(ce[p_pos_in_topk[0]] / ce_sum)
            n_resolved += 1
            if rng.random() >= accept_prob:
                continue

            p_g = int(camp_pool[i])
            chosen_g = int(camp_pool[ci_local])
            mating_events.append((p_g, chosen_g))

            # Pregnancy only when one is female (fertile) and one is male
            if is_f[p_g] != is_f[chosen_g]:
                mom = p_g if is_f[p_g] else chosen_g
                dad = chosen_g if is_f[p_g] else p_g
                if is_p[mom] or is_meno[mom]:
                    continue
                if rng.random() < ref.PREGNANCY_BIRTH_P:
                    s['is_pregnant'][mom] = True
                    s['pregnancy_hours'][mom] = 0
                    s['pregnancy_target'][mom] = int(rng.integers(
                        ref.PREGNANCY_HOURS - ref.PREGNANCY_HOURS_RANGE,
                        ref.PREGNANCY_HOURS + ref.PREGNANCY_HOURS_RANGE + 1))
                    dad_pairs.append((int(mom), int(dad)))

    return n_resolved, dad_pairs, mating_events


# ---------------------------------------------------------------------------
# Gift phase — like communication, ranking-driven, in-camp transfers from
# giver's cache to a recipient's cache. Returns the list of (giver, recipient,
# amount) tuples for the run_vec driver to apply to the gift_matrix.
# ---------------------------------------------------------------------------

def gift_phase(s, weights, Q, rng, mating_matrix=None, gift_matrix=None,
               crests=None, camp_caches=None, K=ref.K_SAMPLE):
    """Gift transfers: each in-camp giver picks a top-K receiver via softmax
    of their ranking and moves GIFT_AMOUNT (capped at giver's cache) into the
    receiver's cache. Returns list of (giver, receiver, amount).

    Batched per-camp: each camp's in-camp pool is scored against itself once
    via _score_pool_pairs, then the giver rows are processed for picks.
    """
    givers = xp.flatnonzero(s['gift_request'] & s['is_in_camp'] & (s['cache'] > 0))
    s['gift_request'][:] = False
    if givers.size == 0:
        return 0, []
    N = s['hunger'].shape[0]
    if N <= 1:
        return 0, []
    F_base = features_vec(s)
    camp_id = s.get('camp_id')
    events = []
    n_done = 0

    # group givers by camp
    giver_camps = (camp_id[givers] if camp_id is not None
                    else xp.zeros(givers.size, dtype=xp.int32))

    for cid in xp.unique(giver_camps):
        camp_givers = givers[giver_camps == cid]
        # full in-camp pool for this camp (potential recipients)
        if camp_id is not None:
            camp_in = xp.flatnonzero(s['is_in_camp'] & (camp_id == cid))
        else:
            camp_in = xp.flatnonzero(s['is_in_camp'])
        if camp_in.size < 2:
            continue
        # score the in-camp pool against itself (P, P)
        score_mat = _score_pool_pairs(camp_in, F_base, weights, Q,
                                        mating_matrix=mating_matrix,
                                        gift_matrix=gift_matrix,
                                        crests=crests, camp_id=camp_id,
                                        camp_caches=camp_caches)
        # find each giver's row index within camp_in
        # (camp_givers are members of camp_in by construction)
        g_local = xp.searchsorted(camp_in, camp_givers)
        # mask self in each giver's row
        P = camp_in.size
        # softmax-pick a recipient per giver
        for i, g_idx in enumerate(g_local):
            row = score_mat[g_idx].copy()
            row[g_idx] = -xp.inf
            Keff = min(int(K), P - 1)
            if Keff <= 0:
                continue
            topk = xp.argpartition(-row, Keff - 1)[:Keff]
            sp = row[topk]
            mx = sp.max()
            if not xp.isfinite(mx):
                continue
            e = xp.exp(sp - mx); e_sum = e.sum()
            if e_sum <= 0:
                continue
            r = float(rng.random()) * e_sum
            cum = 0.0; pick_local = 0
            for kk, ev in enumerate(e):
                cum += ev
                if r <= cum:
                    pick_local = kk; break
            recipient = int(camp_in[topk[pick_local]])
            g = int(camp_givers[i])
            amount = min(float(ref.GIFT_AMOUNT), float(s['cache'][g]),
                         max(0.0, float(ref.CACHE_LIMIT - s['cache'][recipient])))
            if amount <= 0:
                continue
            s['cache'][g] -= amount
            s['cache'][recipient] += amount
            events.append((g, recipient, amount))
            n_done += 1
    return n_done, events


# ---------------------------------------------------------------------------
# Camp phases — birth_settle / join_camp / make_camp / cleanup
# ---------------------------------------------------------------------------

def birth_window_force_settle_phase(s):
    """Any wanderer in the birth window (late pregnancy or post-partum) is
    forced to settle this hour. Sets join_camp_request so that the normal
    join_camp_phase scoops them into an existing camp — or, via the join→
    make fallback, founds a fresh one when no camps exist. This is the
    safety net that keeps newborns from being born in the wild.
    """
    wanderer = s['camp_id'] < 0
    bw = birth_window_mask(s)
    force = wanderer & bw
    if force.any():
        s['join_camp_request'] |= force


def make_camp_phase(s, camp_caches, next_camp_id, camp_positions=None):
    """For each agent that fired agent_make_camp:
      - if a camp already exists at the agent's (x, y), JOIN that camp
        instead of founding a new one (consolidation — prevents singleton
        camps from accumulating at a cell that already has a band).
      - else, allocate a fresh camp_id at the founder's position.
    Returns the updated next_camp_id counter."""
    # Filter to wanderers only — when join_camp_phase runs FIRST and grabs an
    # agent, their stale make_request must not also fire (would orphan them
    # from the camp they just joined).
    make_mask = s['make_camp_request'] & (s['camp_id'] < 0)
    s['make_camp_request'][:] = False
    idx = xp.flatnonzero(make_mask)
    # Build a fast (x,y) → existing_camp_id index from camp_positions.
    pos_index = {}
    if camp_positions is not None:
        for cid, pos in camp_positions.items():
            pos_index.setdefault(pos, cid)   # first-found at cell wins
    for i in idx:
        i_int = int(i)
        pos = (int(s['x'][i_int]), int(s['y'][i_int]))
        existing = pos_index.get(pos)
        if existing is not None:
            # Consolidation: join the camp already present at this cell.
            s['camp_id'][i_int] = int(existing)
            s['is_in_camp'][i_int] = True
            continue
        # No camp here yet — found a new one and register its position.
        new_id = int(next_camp_id)
        next_camp_id += 1
        s['camp_id'][i_int] = new_id
        s['is_in_camp'][i_int] = True
        camp_caches[new_id] = 0.0
        if camp_positions is not None:
            camp_positions[new_id] = pos
            pos_index[pos] = new_id      # subsequent founders this hour join it
    return next_camp_id


JOIN_CAMP_CANDIDATE_CAP = 12  # how many candidate camps to score per requester
                               # (random subsample when more exist; keeps per-
                               #  requester cost bounded regardless of camp count)

def join_camp_phase(s, weights, Q, rng, mating_matrix=None, gift_matrix=None,
                    crests=None, camp_caches=None, K=ref.K_SAMPLE,
                    next_camp_id=None, camp_positions=None):
    """For each agent that fired agent_join_camp, softmax-pick a real camp
    *located in the agent's current cell* by mean social_score over that
    camp's members and reassign their camp_id. When no co-located camp
    exists, the requester founds a new camp at their own position (fallback
    to make_camp so the 'wants to settle' intent is never wasted).
    Returns updated next_camp_id (None if not provided).

    `camp_positions` is the parallel dict {camp_id: (x, y)} maintained by
    make_camp_phase. Required for spatial filtering; if absent the function
    falls back to non-spatial join (legacy behavior)."""
    # Filter to wanderers only — agents who fired BOTH make and join in the
    # same hour get make'd by the earlier phase, so by the time we run their
    # camp_id >= 0 and the join_request is a stale leftover. Without this
    # filter join would re-roll them away from the camp they just founded.
    join_mask = s['join_camp_request'] & (s['camp_id'] < 0)
    s['join_camp_request'][:] = False
    requesters = xp.flatnonzero(join_mask)
    if requesters.size == 0:
        return next_camp_id
    all_camps = xp.unique(s['camp_id'])
    # Exclude the wanderer sentinel — wanderers can join real camps, but no
    # one can "join" the wandering pool (you become a wanderer via leave_camp).
    all_camps = all_camps[all_camps >= 0]

    def _found_new_camp(r):
        nonlocal next_camp_id
        new_id = int(next_camp_id)
        next_camp_id += 1
        s['camp_id'][r] = new_id
        s['is_in_camp'][r] = True
        camp_caches[new_id] = 0.0
        if camp_positions is not None:
            camp_positions[new_id] = (int(s['x'][r]), int(s['y'][r]))
        return new_id

    if all_camps.size == 0:
        # No camps exist yet — every join request becomes a make.
        if next_camp_id is None or camp_caches is None:
            return next_camp_id
        for r in requesters:
            _found_new_camp(int(r))
        return next_camp_id
    F_base = features_vec(s)
    # Precompute camp_id → members array once (O(N)), reused across requesters.
    camp_id_arr = s['camp_id']
    members_by_camp = {}
    for i in range(camp_id_arr.size):
        c = int(camp_id_arr[i])
        members_by_camp.setdefault(c, []).append(i)
    members_by_camp = {c: xp.array(v, dtype=xp.intp)
                        for c, v in members_by_camp.items()}
    for r in requesters:
        r = int(r)
        own_camp = int(s['camp_id'][r])
        # Spatial filter: only camps at the requester's current cell are
        # candidates. Without camp_positions we fall back to the
        # non-spatial behavior (all camps everywhere).
        if camp_positions is not None:
            rx, ry = int(s['x'][r]), int(s['y'][r])
            candidates = [c for c in all_camps.tolist()
                          if int(c) != own_camp
                          and camp_positions.get(int(c)) == (rx, ry)]
        else:
            candidates = [c for c in all_camps.tolist() if int(c) != own_camp]
        if not candidates:
            # No co-located camp — found a new one here (spatial fallback).
            if next_camp_id is not None and camp_caches is not None:
                _found_new_camp(r)
            continue
        # cap the candidate set to a small number of camps per requester —
        # keeps per-requester cost bounded as C grows. Random subsample.
        # NB: use permutation+slice (works on both numpy & cupy; .choice is
        # not implemented on cupy.random.Generator).
        cand_cap = min(K, JOIN_CAMP_CANDIDATE_CAP)
        if len(candidates) > cand_cap:
            # Efraimidis-Spirakis weighted sample-without-replacement, with
            # weight = camp_size^SIZE_WEIGHT. This bakes preferential
            # attachment into the candidate subsampling itself, so big
            # camps don't get dropped from the consideration set. Without
            # this, the cap would erase the population-level Matthew
            # effect — a 100-person camp would have the same prob of being
            # a candidate as a 1-person one.
            sizes = np.array(
                [members_by_camp[int(c)].size for c in candidates],
                dtype=np.float64)
            w = float(ref.JOIN_CAMP_SIZE_WEIGHT)
            weights_v = np.power(sizes, w) if w != 0.0 else np.ones_like(sizes)
            u = rng.random(len(candidates))
            u_h = to_host(u) if hasattr(u, 'get') or not isinstance(u, np.ndarray) else u
            u_h = np.asarray(u_h, dtype=np.float64)
            keys = -np.log(np.clip(u_h, 1e-12, 1.0)) / np.maximum(weights_v, 1e-12)
            order = np.argsort(keys)  # smallest key = highest weight
            picked = [int(order[i]) for i in range(cand_cap)]
            candidates = [candidates[i] for i in picked]
        camp_scores = []
        for c in candidates:
            members = members_by_camp.get(int(c))
            if members is None or members.size == 0:
                continue
            full_size = int(members.size)
            # cap scoring cost when camps are large; note we keep full_size
            # (pre-cap) for the preferential-attachment size bonus so the
            # subsample doesn't penalize big camps.
            if members.size > K:
                keys = rng.random(members.size)
                order = xp.argsort(keys)[:K]
                members = members[order]
            sp = _score_against_pool(r, members, F_base, weights, Q,
                                       mating_matrix, gift_matrix=gift_matrix,
                                       crests=crests,
                                       camp_id=s.get('camp_id'),
                                       camp_caches=camp_caches)
            # Preferential attachment is added later (after temperature
            # scaling) so the size weight isn't tied to the social temp.
            camp_scores.append((c, float(sp.mean()), full_size))
        if not camp_scores:
            continue
        # Score-and-sample on CPU-friendly path: small lists, easy with numpy.
        cids = [c for c, _, _ in camp_scores]
        scs = np.array([sc for _, sc, _ in camp_scores], dtype=np.float64)
        sizes_v = np.array([sz for _, _, sz in camp_scores], dtype=np.float64)
        # Social score gets temperature-scaled; the preferential-attachment
        # size term is added AFTER scaling so it's stable w.r.t. temp.
        # Result: P(c) ∝ exp(social_c / T) * size_c^SIZE_WEIGHT.
        scs = scs / max(float(ref.JOIN_CAMP_TEMPERATURE), 1e-6)
        scs = scs + float(ref.JOIN_CAMP_SIZE_WEIGHT) * np.log(np.maximum(sizes_v, 1.0))
        scs = scs - scs.max()
        probs = np.exp(scs); probs = probs / probs.sum()
        # weighted-pick via cumsum (numpy land; small array)
        u = float(rng.random())
        cum = np.cumsum(probs)
        pick = int(np.searchsorted(cum, u))
        if pick >= len(cids):
            pick = len(cids) - 1
        s['camp_id'][r] = int(cids[pick])
        s['is_in_camp'][r] = True
    return next_camp_id


def cleanup_empty_camps(s, camp_caches, camp_positions=None):
    """Disband any camp with 0 members; destroy its cache (and its position
    record if camp_positions is provided). Called after deaths (and after
    compact_dead so the state array is current). 'No guards → animals
    take the larder' — unguarded camps disappear entirely."""
    if not camp_caches:
        return
    occupied = set(int(c) for c in xp.unique(s['camp_id']))
    for cid in list(camp_caches.keys()):
        if cid not in occupied:
            del camp_caches[cid]
            if camp_positions is not None:
                camp_positions.pop(cid, None)


# ---------------------------------------------------------------------------
# Cultural memory: per-node freshness, decay, and GC.
# ---------------------------------------------------------------------------

def decay_freshness(s, decay=None):
    """Tick down node_freshness by `decay` per hour (clip at 0). Called once
    per hour after step_all. Nodes that hit 0 are eligible for GC."""
    if 'node_freshness' not in s:
        return
    d = int(ref.FRESHNESS_DECAY) if decay is None else int(decay)
    s['node_freshness'] = xp.maximum(0, s['node_freshness'] - d)


def reindex_freshness(s, agent_i, old_nodes, new_nodes):
    """After a reflatten changes the DFS slot order, carry per-node freshness
    through to the new layout. Looks up each new-DFS slot's Python node by
    identity in the old layout; never-seen-before nodes (e.g. freshly-grafted
    subgraph members) get FRESHNESS_MAX so they start fresh."""
    if 'node_freshness' not in s:
        return
    cap = s['node_freshness'].shape[1]
    old_id_to_freshness = {}
    for slot, n in enumerate(old_nodes):
        if slot >= cap:
            break
        old_id_to_freshness[id(n)] = int(s['node_freshness'][agent_i, slot])
    new_arr = np.zeros(cap, dtype=np.int16)
    for slot, n in enumerate(new_nodes):
        if slot >= cap:
            break
        new_arr[slot] = old_id_to_freshness.get(id(n), int(ref.FRESHNESS_MAX))
    s['node_freshness'][agent_i] = to_device(new_arr, dtype=xp.int16) if USE_GPU else xp.asarray(new_arr)


def gc_dead_nodes(s, graphs, roots, n_nodes_arr):
    """Per-agent garbage collection: nodes with freshness==0 (and not the
    root slot, which we keep structurally) have their incoming edges
    redirected to skip past them — making them unreachable in the next
    DFS-flatten. Reflatten each affected row; freshness and `cur` are
    re-anchored to Python-node identity, so an agent never loses its
    place. Run once per day from the main loop."""
    if 'node_freshness' not in s or roots is None:
        return
    N = s['node_freshness'].shape[0]
    fresh_h = to_host(s['node_freshness']) if USE_GPU else np.asarray(s['node_freshness'])
    n_nodes_h = to_host(n_nodes_arr) if USE_GPU else np.asarray(n_nodes_arr)
    cur_h = to_host(s['cur']) if USE_GPU else np.asarray(s['cur'])
    for i in range(N):
        n_alive = int(n_nodes_h[i])
        if n_alive <= 1:
            continue
        # find dead slots in (1..n_alive), keep slot 0 (root) alive always
        dead_slots = [d for d in range(1, n_alive) if fresh_h[i, d] == 0]
        if not dead_slots:
            continue
        old_nodes = ref.all_nodes(roots[i])
        if not old_nodes:
            continue
        # Build dead-set as python-node identities
        dead_ids = set()
        for d in dead_slots:
            if d < len(old_nodes):
                dead_ids.add(id(old_nodes[d]))
        if not dead_ids:
            continue
        # Edge surgery: for every node, if its true_node/false_node points to
        # a dead node, redirect past it (chain of skips terminates at slot 0
        # in the worst case, since root is never dead).
        def skip_dead(target, depth=0):
            seen = set()
            while target is not None and id(target) in dead_ids and depth < len(old_nodes):
                if id(target) in seen:
                    return roots[i]    # cycle guard: fall back to root
                seen.add(id(target))
                target = target.true_node if target.true_node is not None else roots[i]
                depth += 1
            return target if target is not None else roots[i]
        for n in old_nodes:
            if n.true_node is not None and id(n.true_node) in dead_ids:
                n.true_node = skip_dead(n.true_node)
            if n.false_node is not None and id(n.false_node) in dead_ids:
                n.false_node = skip_dead(n.false_node)
        # remember current cur python node for re-anchor
        old_cur = int(cur_h[i])
        cur_pynode = old_nodes[old_cur] if old_cur < len(old_nodes) else None
        # if cur landed on a dead node, snap to its skip target
        if cur_pynode is not None and id(cur_pynode) in dead_ids:
            cur_pynode = skip_dead(cur_pynode)
        # Re-flatten — dead nodes are now unreachable from root → dropped
        new_size = reflatten_row(graphs, i, roots[i])
        n_nodes_arr[i] = new_size
        new_nodes = ref.all_nodes(roots[i])
        reindex_freshness(s, i, old_nodes, new_nodes)
        # re-anchor cur
        if cur_pynode is not None:
            for ni, nn in enumerate(new_nodes):
                if nn is cur_pynode:
                    s['cur'][i] = ni
                    break


def resource_growth_step(resources):
    """Per-cell logistic growth on the (W, H, 3) numpy resource array:
       R <- clip(R + r*R*(1-R/K), MIN_FRAC*K, K)   per cell, per kind
    Mutates `resources` in place. Each kind has its own per-cell K (so the
    growth normalization is correct in cells where stocks are still high)."""
    if resources is None:
        return
    r = float(ref.LOGISTIC_GROWTH_R)
    floor_frac = float(ref.MIN_RESOURCE_FRAC)
    K_per = np.array([ref.RESOURCE_K_HUNT_PER_CELL,
                       ref.RESOURCE_K_FISH_PER_CELL,
                       ref.RESOURCE_K_GATHER_PER_CELL], dtype=np.float64)
    # Broadcast K across (W, H, 3)
    K_grid = K_per.reshape(1, 1, 3)
    R = resources
    R += r * R * (1.0 - R / K_grid)
    np.minimum(R, K_grid, out=R)
    np.maximum(R, K_grid * floor_frac, out=R)
    # Impassable terrain cells stay at zero — the floor clamp above would
    # otherwise re-grow them to MIN_RESOURCE_FRAC*K, leaking unreachable yield
    # into the global readout.
    impassable = ~ref.TERRAIN_MASK
    R[impassable, :] = 0.0


# ---------------------------------------------------------------------------
# Deaths — vectorized
# ---------------------------------------------------------------------------

def death_mask(s, rng, season_val=1.0):
    """Bool mask of agents that die this hour."""
    N = s['hunger'].shape[0]
    hazard = rng.random(N) < ref.HAZARD_PER_HOUR
    oldage = s['age'] > ref.MAX_AGE_YEARS
    hunger = s['hunger'] > ref.HUNGER_DEATH
    # per-hour death chance when sleeping outside camp; scales seasonally
    # (more dangerous in winter, summer at base risk)
    exposed_p = ref.SLEEP_EXPOSED_DEATH_P * (
        ref.SLEEP_EXPOSED_WINTER_FACTOR
        + (1.0 - ref.SLEEP_EXPOSED_WINTER_FACTOR) * season_val)
    exposed = s['sleep'] & ~s['is_in_camp'] & (rng.random(N) < exposed_p)
    is_kid = (s['age'] < ref.WATCH_AGE_YEARS) & ~s['sleep']
    # Tolerance for being unwatched grows linearly with age, from the
    # newborn baseline (3 hr in-camp / 1 hr out-of-camp) up to
    # CHILD_NEGLECT_TOLERANCE_MAX hours at age WATCH_AGE_YEARS. This
    # models increasing self-sufficiency as kids get older — a near-toddler
    # can survive a couple of unsupervised hours, while a near-WATCH-AGE
    # kid (8 in current config) is essentially self-sufficient for a day.
    frac = xp.minimum(s['age'], ref.WATCH_AGE_YEARS) / float(ref.WATCH_AGE_YEARS)
    limit_in  = ref.CHILD_IN_CAMP_LIMIT  + (ref.CHILD_NEGLECT_TOLERANCE_MAX - ref.CHILD_IN_CAMP_LIMIT)  * frac
    limit_out = ref.CHILD_OUT_CAMP_LIMIT + (ref.CHILD_NEGLECT_TOLERANCE_MAX - ref.CHILD_OUT_CAMP_LIMIT) * frac
    limit = xp.where(s['is_in_camp'], limit_in, limit_out)
    neglect = is_kid & (s['unwatched_hours'] >= limit)
    return hazard | oldage | hunger | exposed | neglect, dict(
        hazard=int(hazard.sum()),
        oldage=int(oldage.sum()),
        hunger=int((hunger & ~hazard).sum()),
        exposed=int((exposed & ~hazard & ~hunger).sum()),
        neglect=int((neglect & ~hazard & ~hunger & ~exposed & ~oldage).sum()),
    )


def forced_sleep_step(s):
    """Trigger forced sleep when tired hits threshold; override sleep state and
    decrement timer for agents already in forced sleep. Runs after step_all."""
    # Trigger: tired >= TIRED_DEATH and not already in forced sleep
    trigger = (s['tired'] >= ref.TIRED_DEATH) & (s['forced_sleep_hours'] == 0)
    s['forced_sleep_hours'] = xp.where(trigger, 12, s['forced_sleep_hours'])
    # Any wanderer who just got forced to pass out must be settled THIS
    # hour, otherwise they sleep=True out-of-camp and the death_mask
    # (run later this hour) kills them by exposure. Queue a join request;
    # the join phase later in the same hour will move them into a camp
    # before deaths are evaluated.
    s['join_camp_request'] |= trigger & (s['camp_id'] < 0)

    forced = s['forced_sleep_hours'] > 0
    if not forced.any():
        return
    # Override: forced agents are asleep regardless of graph behavior. They do
    # NOT get moved back to camp — they pass out where they are.
    s['sleep'] = xp.where(forced, True, s['sleep'])
    s['consecutive_sleep'] = xp.where(forced, s['consecutive_sleep'] + 1,
                                       s['consecutive_sleep'])
    # Forced sleep: tired hard-clamped to 0 each hour (guaranteed full recovery)
    s['tired'] = xp.where(forced, 0.0, s['tired'])
    # Forced agents skip the graph walk in step_all (so no per-visit hunger).
    # They still pay an hour of metabolism here — same total cost as a walking
    # agent would have racked up over 33 visits.
    metab = ref.NODE_VISIT_HUNGER * ref.MAX_ACTION * metabolic_scale_vec(s['age'])
    s['hunger'] = xp.where(forced, s['hunger'] + metab, s['hunger'])
    # Tick timer down
    s['forced_sleep_hours'] = xp.where(forced, s['forced_sleep_hours'] - 1,
                                        s['forced_sleep_hours'])


# ---------------------------------------------------------------------------
# Compaction & growth helpers
# ---------------------------------------------------------------------------

def slice_state(s, mask):
    """Return new state dict containing only agents where mask is True."""
    return {k: v[mask].copy() for k, v in s.items()}


def slice_graphs(graphs, mask):
    return {k: v[mask].copy() for k, v in graphs.items()}


def append_state(s, s2):
    return {k: xp.concatenate([v, s2[k]]) for k, v in s.items()}


def append_graphs(graphs, graphs2):
    return {k: xp.concatenate([v, graphs2[k]]) for k, v in graphs.items()}


def reflatten_row(graphs, agent_idx, root):
    """Re-flatten one agent's graph into the stacked arrays.

    flatten_graph always returns numpy; we explicitly upload to the device
    that `graphs` lives on before assigning to the row.
    """
    g = flatten_graph(root)
    for k in ('seq_var', 'seq_op', 'seq_value', 'seq_links',
              'seq_len', 'true_action', 'false_action',
              'true_node', 'false_node'):
        graphs[k][agent_idx] = to_device(g[k])
    return g['n_nodes']

