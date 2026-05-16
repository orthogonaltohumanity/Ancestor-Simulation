import json
import math
import os
import random
import numpy as np
from tqdm import tqdm
MAX_ACTION = 30   # exactly N nodes visited per hour-walk (= canonical size)
MAX_SEQ_LEN = 10  # max chunk sequence length
MAX_NODES = 10    # max nodes per agent's decision graph

# Cultural-memory parameters. Each node carries a freshness counter; visiting
# the node during a chain walk refreshes it to FRESHNESS_MAX; each hour it
# decays by FRESHNESS_DECAY. Once at 0, a GC pass (run every
# GC_INTERVAL_HOURS) splices the node out of the agent's graph. Models the
# fading-from-disuse property of real cultural practices.
FRESHNESS_MAX = 3*24                            # hours = 7 days
FRESHNESS_DECAY = 1                            # per hour
GC_INTERVAL_HOURS = 24                         # daily GC

MUTATION_RATE_PER_HOUR = 3.0 / 24.0            # default; overridden by adaptive per-agent rate
ADAPTIVE_MUT_HIGH = 1.0 / (24.0)    # 1 / 3 months for low-self-ranked agents
ADAPTIVE_MUT_LOW = 1.0 / (3 * 365.0 * 24.0)    # 1 / 3 years for high-self-ranked agents
LISTEN_ADOPT_P = 0.30                          # default; overridden by adaptive per-agent rate
ADAPTIVE_ADOPT_HIGH = 1.0                     # 30% for low-self-ranked agents
ADAPTIVE_ADOPT_LOW = 0.001                       # 0% for high-self-ranked agents

# --- social ranking / weighted listening ---
# Each agent has a linear social-ranking function scored over five features.
# Listeners softmax-sample which talker to attend to using their own weights.
SOCIAL_FEATURES = ('net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
                   'age', 'is_pregnant', 'is_menopausal',
                   'matings_with_target', 'gifts_with_target',
                   'family_similarity', 'target_camp_cache',
                   'opinion_similarity')
N_FEATURES = len(SOCIAL_FEATURES)
OPINION_NORM = 1.0   # cosine sim already in [-1, 1]
# matings_with_target and gifts_with_target are observer-target pairwise —
# each pair of agents has a decaying running count of their interactions.
# Computed at score time. See mating_matrix and gift_matrix in run_vec.py.
MATING_DECAY = 0.5 ** (1.0 / (30 * 24))   # half-life = 30 days
MATING_NORM  = 1.0 / 10.0                  # feature normalizer (typical pair count 0-10)
GIFT_AMOUNT = 1000                          # kcal moved per gift action
GIFT_DECAY = 0.5 ** (1.0 / (30 * 24))      # half-life = 7 days
GIFT_NORM = 1.0 / 1000.0                   # feature normalizer (typical pair flow 0-5000)
WEIGHT_INIT_STD = 0.5                 # initial weight ~ N(0, this)
WEIGHT_MUT_STD = 0.1                  # per-mutation Δweight ~ N(0, this)
# Memetic weight teaching: per listen-adopt event, the listener's
# (weights, Q) pull toward the chosen talker by α = mut_rate * adopt_rate
# * WEIGHT_TEACH_SCALE. Both rates are per-agent (adaptive), so curious /
# open agents culturally learn ranking-function preferences faster than
# rigid ones. Product of baselines ≈ 0.04, so SCALE=1.0 means ~4% pull
# per event at default rates.
WEIGHT_TEACH_SCALE = 1.0
P_MUT_WEIGHT = 0.20                   # 20% weight / 80% graph (split below)
# Among non-weight mutations (must sum to 1.0): chunk / seq / node / graph_dual.
# Targets at top-level: 58% chunk, 12% seq, 9% node, 1% dual (= 80% non-weight).
P_MUT_CHUNK_FRAC = 58.0 / 80.0        # 0.725
P_MUT_SEQ_FRAC   = 12.0 / 80.0        # 0.15
P_MUT_NODE_FRAC  =  9.0 / 80.0        # 0.1125
P_MUT_DUAL_FRAC  =  1.0 / 80.0        # 0.0125 — rare whole-graph dual macro-mutation
# Weight-mutation flavor split (must sum to 1.0):
WEIGHT_MUT_PERTURB_P = 0.70           # additive Gaussian
WEIGHT_MUT_FLIP_P    = 0.15           # negate the weight (×-1)
WEIGHT_MUT_RESET_P   = 0.15           # reset to 0
# --- family crests ---
FAMILY_CREST_DIM       = 256           # dim of unit-vector crest. Two random unit
                                        # vectors have cos ~ N(0, 1/√D) ≈ N(0, 0.063),
                                        # so unrelated agents land in roughly [-0.2, +0.2].
FAMILY_INCEST_THRESHOLD = 0.6          # mating disallowed if cos(crest_a, crest_b) > this.
                                        # blocks: siblings (1.0), parent-child (~0.71),
                                        # aunt/uncle-nephew/niece (~0.71 — see note).
                                        # allows: half-siblings (~0.50), first cousins
                                        # (~0.50), grandparent-grandkid (~0.50), and
                                        # all second cousins and beyond.
                                        # Note: under deterministic slerp, aunts share
                                        # their parent's crest exactly with moms, so
                                        # aunt-nephew gets blocked alongside parent-child.
FAMILY_NORM            = 1.0           # ranking-feature normalizer for cosine sim
SIM_DAYS = 365 * 2000                         # 200 years

# --- dynamics constants ---
BASE_HUNGER_PER_HOUR = 0.0                    # everything folded into per-visit cost
SLEEP_RECOVERY = 0.17                         # 6 recovery hrs × 0.17 ≈ 1.0 tired cleared per 8h cycle
EAT_RATE = 1000
FORAGE_YIELD = 2000          # given P=0.2 success, expected yield = 400/h
FORAGE_SUCCESS_P = 1.0
SLEEP_WARMUP = 2                              # hrs of consecutive sleep before recovery starts
TIRED_PER_HOUR = 1.0 / 24.0                   # passive tiredness drift
FORAGE_HUNGER = 0                             # foraging cost folded into node-visit cost
IDLE_HUNGER = 0                               # idle == None (cost is just NODE_VISIT_HUNGER)
NODE_VISIT_HUNGER = 3                         # 33×24×3 = ~2376/day total adult cost
KID_METAB_FRAC = 300.0 / 2400.0               # newborn: 12.5% of adult metabolism

def metabolic_scale(age):
    """0→METABOLIC_ADULT_AGE years: 12.5% → 100% linearly. Adult: 100%."""
    if age >= METABOLIC_ADULT_AGE:
        return 1.0
    return KID_METAB_FRAC + (age / METABOLIC_ADULT_AGE) * (1.0 - KID_METAB_FRAC)
CACHE_LIMIT = 12000                           # ~6 days' worth of kcal
CACHE_DECAY = 0.5 ** (1.0 / (30.0*24.0))             # half-life = 30 days = 720 h
NDF_DECAY = 0.5 ** (1.0 / (30 * 24))           # net_debt_flow half-life = 7 days
DEPOSIT_AMOUNT = 1000                          # kcal moved per deposit/withdraw action
WITHDRAW_AMOUNT = 1000
CAMP = {'cache': 0.0}                         # legacy single-camp store (main.py reference path only;
                                              # vec sim uses per-camp dict in run_vec.py)

# --- multi-camp dynamics ---
CAMP_INIT_CACHE = 0.0                         # founder camp starts empty; agents
                                              # build the larder via deposits
CAMP_CACHE_NORM = 1.0 / 3_000_000.0            # feature normalizer for target_camp_cache (scaled 3× to match 30-day cache half-life equilibrium)
                                              # (1M kcal larder → feature value 1.0)
JOIN_CAMP_TEMPERATURE = 1.0                   # softmax temp for agent_join_camp camp pick
JOIN_CAMP_SIZE_WEIGHT = 1.0                   # preferential-attachment coefficient: per-camp score gets += SIZE_WEIGHT*log(pop_size) before softmax, so P(pick) ∝ pop_size^SIZE_WEIGHT × exp(social_score). 1.0 = Barabási-Albert; 0.0 = pure social-score; >1 = "winner takes all"

# --- seasonal forage scaling ---
# yield_factor = WINTER_FACTOR + (1 - WINTER_FACTOR) * season,  season ∈ [0, 1]
# season=0 (winter) → WINTER_FACTOR; season=1 (peak summer) → 1.0
HUNT_WINTER_FACTOR = 1.2                      # hunting unaffected by season
FISH_WINTER_FACTOR = 0.5                      # fishing halves in winter
GATHER_WINTER_FACTOR = 0.3                    # gathering drops to 10% in winter

# --- per-action forage params (success prob + summer yield in kcal) ---
HUNT_SUCCESS_P   = 0.01      # rare jackpot
HUNT_YIELD       = CACHE_LIMIT  # fills cache (uses max(cache, CACHE_LIMIT*f) semantics)
FISH_SUCCESS_P   = 0.8
FISH_YIELD       = 8000
GATHER_SUCCESS_P = 1.0
GATHER_YIELD     = 22000

# --- resource pools (logistic-growth populations of game, fish, plants) ---
# Each forage type extracts from its own pool. Yields scale by (R/K): full
# yield at carrying capacity, MIN_FRAC * full yield at the floor.
# Growth: R += LOGISTIC_R * R * (1 - R/K) per hour. Resources never extinct:
# the pool is floored at MIN_RESOURCE_FRAC * K each step.
RESOURCE_K_HUNT   = 2_000_000_000_000.0     # carrying capacity (kcal-equivalent) of game
RESOURCE_K_FISH   = 2_000_000_000_000.0
RESOURCE_K_GATHER = 2_000_000_000_000.0

# Spatial grid (Sugarscape-style). Per-cell K = global / num cells, so total
# carrying capacity is preserved as the grid is subdivided.
GRID_W = 10
GRID_H = 10
MOVE_COOLDOWN_HOURS = 12    # min hours between successive moves for one agent (it takes time to walk between cells)
VISION_RANGE = 2            # max distance (in cells) at which nearest_camp_* sentinels report a hit; -1 = nothing visible at or within this range in that direction. Line of sight is blocked by impassable terrain.
RESOURCE_K_HUNT_PER_CELL   = RESOURCE_K_HUNT   / (GRID_W * GRID_H)
RESOURCE_K_FISH_PER_CELL   = RESOURCE_K_FISH   / (GRID_W * GRID_H)
RESOURCE_K_GATHER_PER_CELL = RESOURCE_K_GATHER / (GRID_W * GRID_H)
LOGISTIC_GROWTH_R = 1e-6           # per-hour intrinsic growth rate (logistic r)
MIN_RESOURCE_FRAC = 0.001            # pool can't fall below this fraction of K

# --- timezones & time-of-day foraging rhythms ---
# Local hour at column x is offset from the global tick by (x / W) of a full
# day. TIMEZONES_PER_WORLD sets that scale: 24 = one full day across the
# x-width (continuous on the torus), 12 = half-day across (so x=0 and x=W
# differ by 12 hours), etc. Per-activity peaks below are in local hours.
TIMEZONES_PER_WORLD   = 24           # local hours spanned across the world's x-width
TOD_FLOOR             = 0.15         # min factor at worst time of day (vs 1.0 at peak)
# HUNT: cos² rhythm. PERIOD sets how often peaks repeat — 12 = twin-peak
# crepuscular (dawn + dusk), 24 = single daily peak, 8 = three peaks per
# day, etc. PEAK_HOUR shifts the first peak; subsequent peaks fall every
# PERIOD hours after it (mod 24).
HUNT_TOD_PEAK_HOUR    = 6.0
HUNT_TOD_PERIOD       = 12.0         # hours between successive peaks (12 = crepuscular)
# FISH: same cos² family. PERIOD=24 → one peak/day at PEAK_HOUR.
FISH_TOD_PEAK_HOUR    = 12.0
FISH_TOD_PERIOD       = 24.0
# GATHER: Gaussian bell on toroidal hour distance — single peak per day,
# WIDTH controls how sharply it falls off (larger = longer foraging window).
GATHER_TOD_PEAK_HOUR  = 9.0
GATHER_TOD_WIDTH      = 4.5          # gaussian std-dev in hours

# --- terrain (mountains-with-valleys) ---
# A perlin-ish noise field thresholded into passable/impassable cells. Forces
# agents into habitability bubbles connected by narrow passes, mirroring
# Pleistocene refugia: isolation most of the time, occasional gene flow.
# TERRAIN_PASSABLE_FRAC sits in the percolation band (~0.4-0.6) — too high
# and the passable region is one big blob (no funneling); too low and it
# shatters into islands (each crashes from inbreeding).
TERRAIN_PASSABLE_FRAC = 1.0 
TERRAIN_FREQUENCY     = 20     # base octave grid size (low = few big mountains)
TERRAIN_OCTAVES       = 10
TERRAIN_SEED          = 3 


def _bilinear_upsample(coarse, W, H):
    """Toroidal bilinear upsample: coarse is shape (cw, ch), output (W, H).
    The coarse grid is treated as periodic — sampling wraps modulo (cw, ch)
    so the result is continuous across the x=0 and y=0 seams of the torus
    that agent movement already wraps around."""
    import numpy as _np
    cw, ch = coarse.shape
    # Sample positions span [0, cw) (NOT [0, cw-1]) so the seam at i=W lands
    # cleanly on coarse index 0 again, giving the wrap.
    xs = _np.arange(W, dtype=_np.float64) * (cw / W)
    ys = _np.arange(H, dtype=_np.float64) * (ch / H)
    x0 = _np.floor(xs).astype(_np.int64) % cw
    y0 = _np.floor(ys).astype(_np.int64) % ch
    x1 = (x0 + 1) % cw
    y1 = (y0 + 1) % ch
    fx = (xs - _np.floor(xs))[:, None]
    fy = (ys - _np.floor(ys))[None, :]
    a = coarse[_np.ix_(x0, y0)]
    b = coarse[_np.ix_(x1, y0)]
    c = coarse[_np.ix_(x0, y1)]
    d = coarse[_np.ix_(x1, y1)]
    return a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy


def generate_terrain_mask(W=GRID_W, H=GRID_H,
                           frequency=TERRAIN_FREQUENCY,
                           octaves=TERRAIN_OCTAVES,
                           passable_frac=TERRAIN_PASSABLE_FRAC,
                           seed=TERRAIN_SEED):
    """Summed-octaves value noise → bool mask (W, H), True = passable.
    Threshold is set by quantile so passable_frac is exact regardless of
    noise distribution. Indexed [x, y] to match the resources array."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    field = _np.zeros((W, H), dtype=_np.float64)
    amp = 1.0
    total = 0.0
    f = max(2, frequency)
    for _ in range(octaves):
        coarse = rng.random((f, f))
        field += amp * _bilinear_upsample(coarse, W, H)
        total += amp
        amp *= 0.5
        f *= 2
    field /= total
    thresh = _np.quantile(field, passable_frac)
    return field <= thresh


# Generated once at module import — deterministic from TERRAIN_SEED.
TERRAIN_MASK = generate_terrain_mask()


def find_passable_cell(x, y, mask=None):
    """Return (x', y') = the passable cell nearest to (x, y) by Chebyshev
    distance. Falls through to (x, y) if no passable cell exists (which can't
    happen for any sane mask)."""
    import numpy as _np
    m = mask if mask is not None else TERRAIN_MASK
    W, H = m.shape
    x = int(x) % W; y = int(y) % H
    if m[x, y]:
        return x, y
    for r in range(1, max(W, H)):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                xx, yy = (x + dx) % W, (y + dy) % H
                if m[xx, yy]:
                    return xx, yy
    return x, y


PASSABLE_CENTER = find_passable_cell(GRID_W // 2, GRID_H // 2)

# --- selection ---
N_AGENTS = GRID_W * GRID_H * 4  # two M+F pairs per cell
K_SAMPLE = 200    # listener / proposer samples K random candidates instead of all
BURN_IN_ASEX_CYCLES = 0      # phase 1: cycles of (mutate-once + 72h solo-walk test); cull failures. Set to 0 to skip — burn-in is currently disabled while we iterate on the spatial mechanics.
BURN_IN_TEST_HOURS = 144        # hours of solo simulation each agent runs each cycle
BURN_IN_HUNGER_LIMIT = 5000  # peak hunger cutoff during the solo test; a healthy canonical agent reaches ~2880 over 144h, so this gives ~3× headroom while still catching mutants whose eat path is broken (they trend toward HUNGER_DEATH=120000)
BURN_IN_MIN_CAMP_FRAC = 0.25  # min fraction of test hours spent in camp (else cull)
BURN_IN_SEXUAL = 0          # phase 2: ranking-based sexual reproduction events
BURN_IN_CACHE_DIR = 'burnin_cache'  # save/load burnt-in agents here for reuse
HUNGER_DEATH = 60 * 2000                        # 60 days of pure idle starvation
TIRED_DEATH = 1.0                                # tired threshold that triggers 12h forced sleep
SLEEP_EXPOSED_DEATH_P = 1.0/100.0                     # per-hour base death chance while sleeping outside camp
SLEEP_EXPOSED_WINTER_FACTOR = 10.0                # multiplier on base in winter; summer=1×, winter=factor×
# effective_risk = SLEEP_EXPOSED_DEATH_P × (WINTER_FACTOR + (1 - WINTER_FACTOR) * season)

# --- demography ---
HOURS_PER_YEAR = 365 * 24
AGE_PER_HOUR = 1.0 / HOURS_PER_YEAR            # age tracked in years
HUNT_AGE_YEARS = 13                             # age to start hunting
FISH_AGE_YEARS = 8                              # age to start fishing
GATHER_AGE_YEARS = 4                            # age to start gathering
FORAGE_AGE_YEARS = GATHER_AGE_YEARS             # earliest forage age (used by generic agent_forage and leave-camp gate)
WATCH_AGE_YEARS = 8                            # age to stop needing a watcher; can leave camp solo
METABOLIC_ADULT_AGE = 15                        # age to reach full adult metabolism
MATE_AGE_YEARS = 16
PREGNANCY_HOURS = 9 * 30 * 24                  # 9 months mean
PREGNANCY_HOURS_RANGE = 20 * 24                # ± stochastic spread (days × 24)
PREGNANCY_FORAGE_FRACTION = 1.0               # pregnant females can forage during first X of pregnancy
BIRTH_WINDOW_HOURS = 7 * 24                    # late-pregnancy + post-partum protection: mom is barred from leaving camp (and forced to settle if wandering) for this many hours before and after birth. Newborns are too young to forage and need a camp to survive.


HAZARD_PER_HOUR = 0.005 / HOURS_PER_YEAR       # 0.005 hazard per year
PREGNANCY_BIRTH_P = 0.75                       # P(pregnancy | successful mating)
TWIN_BETA = 4.0                                 # P(N+1|N babies) = exp(-beta) → twins ~1.8%, triplets ~0.03%
MENOPAUSE_AGE = 40                              # menopause hazard kicks in here
MENOPAUSE_BETA = 0.10                           # exponential rate parameter
MENOPAUSE_BASE_PER_HOUR = 1.0 / (365.0 * 24.0)  # baseline P at age 40 (mean ~1 yr)
WATCH_FEED = 500                               # hunger reduction per watch (free)
CHILD_IN_CAMP_LIMIT = 3                        # newborn baseline: die if unwatched ≥ 3 hr in camp
CHILD_OUT_CAMP_LIMIT = 1                       # newborn baseline: die if unwatched ≥ 1 hr out of camp
CHILD_NEGLECT_TOLERANCE_MAX = 24               # by WATCH_AGE_YEARS, kids can go 24 hr without a watcher (both in- and out-of-camp). Tolerance grows linearly with age from the newborn baseline up to this cap, modeling increasing self-sufficiency as kids age.
MAX_AGE_YEARS = 120                            # hard lifespan cap


DUMP_EVERY_DAYS = 90     # small-test default: one snapshot per simulated day, so the grid-animation viz can render every frame
DUMP_DIR = 'dumps'

# state variable names — these double as agent attribute names
STATE_VARS = ('hunger', 'cache', 'tired', 'time', 'season',
              'is_female', 'is_in_camp', 'sleep', 'rng',
              'x', 'y',
              'nearest_camp_n', 'nearest_camp_s',
              'nearest_camp_e', 'nearest_camp_w')
ORDERED_VARS = ('hunger', 'cache', 'tired', 'time', 'age', 'season', 'rng',
                'x', 'y',
                'nearest_camp_n', 'nearest_camp_s',
                'nearest_camp_e', 'nearest_camp_w')


SEASON_YEAR_DAYS = 365   # days for the summer/winter band to sweep one full cycle


def season_at_y(y, t):
    """Seasonal value in [0, 1] at row y and global hour-tick t.

    A single sine wave runs over the vertical (y) axis — one warm band,
    one cold band — and its PHASE advances through time, so the bands
    sweep along y over a SEASON_YEAR_DAYS cycle. Every latitude therefore
    passes through a full year of seasons; there is no fixed 'equator' or
    permanent summer/winter latitude. One full wavelength fits in [0, H),
    so it stays continuous across the y=0/y=H torus seam. Accepts scalar
    or array y; t is the scalar global hour tick."""
    import numpy as _np
    day = int(t) // 24
    phase = 2.0 * _np.pi * day / SEASON_YEAR_DAYS
    return 0.5 + 0.5 * _np.sin(2.0 * _np.pi * _np.asarray(y) / GRID_H - phase)


def season(t):
    """Legacy season shim for the burn-in solo test agent, which has no
    spatial position. Returns the season at the y=0 band for tick t — a
    representative latitude that still cycles through the full year."""
    return float(season_at_y(0, t))


def local_hour(x, global_t, W=None):
    """Per-agent local hour given the global hour-tick. The world spans
    TIMEZONES_PER_WORLD local hours across its x-width: an agent at column
    x is `TIMEZONES_PER_WORLD * x / W` hours ahead of x=0. The torus wraps
    cleanly at x=W (back to x=0). Vectorized over x; t is scalar."""
    import numpy as _np
    W_ = GRID_W if W is None else W
    offset = (TIMEZONES_PER_WORLD * _np.asarray(x) / W_).astype(_np.int32)
    return (int(global_t) + offset) % 24


def hunt_time_factor(h):
    """cos² rhythm. Peaks every HUNT_TOD_PERIOD hours starting at
    HUNT_TOD_PEAK_HOUR; troughs halfway between. Default (period=12)
    gives the classic dawn/dusk crepuscular pattern."""
    import numpy as _np
    h = _np.asarray(h, dtype=_np.float32)
    return TOD_FLOOR + (1.0 - TOD_FLOOR) * _np.cos(
        _np.pi * (h - HUNT_TOD_PEAK_HOUR) / HUNT_TOD_PERIOD) ** 2


def fish_time_factor(h):
    """cos² rhythm. Peaks every FISH_TOD_PERIOD hours starting at
    FISH_TOD_PEAK_HOUR. Default (period=24) = one peak/day at noon."""
    import numpy as _np
    h = _np.asarray(h, dtype=_np.float32)
    return TOD_FLOOR + (1.0 - TOD_FLOOR) * _np.cos(
        _np.pi * (h - FISH_TOD_PEAK_HOUR) / FISH_TOD_PERIOD) ** 2


def gather_time_factor(h):
    """Gaussian bell around GATHER_TOD_PEAK_HOUR with width GATHER_TOD_WIDTH
    (toroidal hour distance, so the tails wrap correctly)."""
    import numpy as _np
    h = _np.asarray(h, dtype=_np.float32)
    d = (h - GATHER_TOD_PEAK_HOUR + 12.0) % 24.0 - 12.0
    return TOD_FLOOR + (1.0 - TOD_FLOOR) * _np.exp(
        -(d * d) / (2.0 * GATHER_TOD_WIDTH * GATHER_TOD_WIDTH))


class chunk:
    def __init__(self):
        self.var = None      # which state variable to read
        self.value = None    # comparison value (number or bool)
        self.op = None       # '<', '>', '==' for ordered vars; None for bool


class chunk_seq:
    def __init__(self):
        self.chunks = []
        self.chunk_links = []   # list of ints in 0..15, length == len(chunks)-1


_NEXT_LINEAGE = 1   # 0 reserved for unstamped / canonical-seed nodes
def _new_lineage():
    global _NEXT_LINEAGE
    lid = _NEXT_LINEAGE
    _NEXT_LINEAGE += 1
    return lid


class node:
    def __init__(self):
        self.seq = None
        self.true_action = None
        self.false_action = None
        self.true_node = None
        self.false_node = None
        self.true_end = True
        self.false_end = True
        # Heritable variant ID. Bumped on any content-changing mutation;
        # preserved on clone and on assimilation (target inherits source's).
        self.lineage = 0


# --- family crest helpers ---
def random_unit_crest():
    """Random unit vector in R^FAMILY_CREST_DIM."""
    v = np.array([random.gauss(0.0, 1.0) for _ in range(FAMILY_CREST_DIM)],
                 dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v = np.zeros(FAMILY_CREST_DIM, dtype=np.float32); v[0] = 1.0
        return v
    return v / n


def slerp_crests(a, b, t=0.5):
    """Spherical linear interpolation on the unit sphere. a, b assumed unit."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(a, b))
    if dot > 1.0: dot = 1.0
    elif dot < -1.0: dot = -1.0
    if abs(dot) > 0.9995:
        # nearly parallel (or antipodal). Linear-interp + renormalize, with
        # antipodal fallback to an arbitrary perpendicular bisector.
        c = (1.0 - t) * a + t * b
        n = float(np.linalg.norm(c))
        if n < 1e-6:
            perp = np.zeros_like(a); perp[0] = 1.0
            if abs(float(np.dot(a, perp))) > 0.9:
                perp = np.zeros_like(a); perp[1] = 1.0
            perp = perp - float(np.dot(perp, a)) * a
            return (perp / float(np.linalg.norm(perp))).astype(np.float32)
        return (c / n).astype(np.float32)
    omega = math.acos(dot)
    so = math.sin(omega)
    return (math.sin((1.0 - t) * omega) / so * a
            + math.sin(t * omega) / so * b).astype(np.float32)


def crest_similarity(a, b):
    """Cosine similarity between two unit-vector crests."""
    return float(np.dot(np.asarray(a, dtype=np.float32),
                        np.asarray(b, dtype=np.float32)))


class agent:
    _next_id = 0

    def __init__(self):
        self.id = agent._next_id
        agent._next_id += 1
        self.hunger = 0.0
        self.cache = 0.0
        self.is_female = random.random() < 0.5
        self.tired = 0.0
        self.is_in_camp = True
        self.camp_id = 0                # which camp the agent belongs to
        # grid position; overwritten on init by callers. Default to a
        # passable cell so throwaway burn-in test agents (which never get a
        # real position assigned) aren't stranded on an impassable tile.
        self.x = PASSABLE_CENTER[0]
        self.y = PASSABLE_CENTER[1]
        # vision sentinels (overwritten per-visit in vectorized path; the
        # CPU eval_chunk path defaults to -1 = "nothing seen")
        self.nearest_camp_n = -1
        self.nearest_camp_s = -1
        self.nearest_camp_e = -1
        self.nearest_camp_w = -1
        self.sleep = False
        self.consecutive_sleep = 0
        self.net_debt_flow = 0.0
        self.weights = [random.gauss(0.0, WEIGHT_INIT_STD) for _ in range(N_FEATURES)]
        self.Q = [[random.gauss(0.0, 0.1) for _ in range(N_FEATURES)] for _ in range(N_FEATURES)]
        # family crest: random unit vector in R^FAMILY_CREST_DIM. Children inherit
        # via slerp(mom, dad, 0.5); mating gated on cosine similarity threshold.
        self.crest = random_unit_crest()
        self.age = 0.0                  # years
        self.is_pregnant = False
        self._is_menopausal = False
        self._pregnancy_hours = 0
        self._pregnancy_target = PREGNANCY_HOURS   # per-pregnancy stochastic target
        self._unwatched_hours = 0       # only meaningful for children
        self._is_watching = False
        self._mate_request = False
        self._gift_request = False
        self._make_camp_request = False
        self._join_camp_request = False
        self._dad_graph_snapshot = None
        self._dad_weights_snapshot = None
        self._dad_Q_snapshot = None
        self._was_in_camp = True
        self._talk_request = False
        self._listen_request = False
        self._mut_rate_per_hour = MUTATION_RATE_PER_HOUR
        self._adopt_rate = LISTEN_ADOPT_P
        self._forced_sleep_hours = 0   # 12h pass-out timer when tired hits cap
        self.season = 0.0              # set each hour from sim time
        # 'time' is supplied per-step; not stored on the agent


# --- 16 binary boolean operators, indexed by truth-table id 0..15 ---
# bit k of id corresponds to (a,b) pair (k=0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1))
def _make_link_table():
    table = []
    for op_id in range(16):
        def f(a, b, _id=op_id):
            k = (int(bool(a)) << 1) | int(bool(b))
            return bool((_id >> k) & 1)
        table.append(f)
    return table


APPLY_LINK = _make_link_table()

# named aliases for readability
LINK_FALSE = 0
LINK_AND = 8
LINK_A_AND_NOT_B = 4
LINK_A = 12
LINK_NOT_A_AND_B = 2
LINK_B = 10
LINK_XOR = 6
LINK_OR = 14
LINK_NOR = 1
LINK_XNOR = 9
LINK_NOT_B = 5
LINK_A_OR_NOT_B = 13
LINK_NOT_A = 3
LINK_NOT_A_OR_B = 11
LINK_NAND = 7
LINK_TRUE = 15


# --- evaluation ---
def eval_chunk(c, ag, hour):
    if c.var == 'rng':
        val = random.random()
    elif c.var == 'time':
        val = hour
    else:
        val = getattr(ag, c.var)
    if c.var in ORDERED_VARS:
        if c.op == '<':
            return val < c.value
        if c.op == '>':
            return val > c.value
        if c.op == '==':
            return val == c.value
        raise ValueError(f"ordered chunk needs op, got {c.op!r}")
    # boolean state var
    if c.value is None:
        return bool(val)
    return bool(val) == bool(c.value)


def eval_seq(seq, ag, hour):
    acc = eval_chunk(seq.chunks[0], ag, hour)
    for i, link in enumerate(seq.chunk_links):
        b = eval_chunk(seq.chunks[i + 1], ag, hour)
        acc = APPLY_LINK[link](acc, b)
    return acc


# --- actions ---
def agent_sleep(ag):
    ag.consecutive_sleep = ag.consecutive_sleep + 1 if ag.sleep else 1
    ag.sleep = True
    ag.is_in_camp = True
    if ag.consecutive_sleep > SLEEP_WARMUP:
        ag.tired = max(0.0, ag.tired - SLEEP_RECOVERY)


# NOTE: action function bodies are vestigial — main_vec dispatches on action
# enums (A_SLEEP…A_NONE), so these never execute. They survive as identity
# markers in ACTION_POOL for index lookups. agent_sleep (above) is the only
# action imported by name from run_vec.py.
def agent_wake(ag): pass
def agent_eat(ag): pass
def agent_forage(ag): pass
def agent_hunt(ag): pass
def agent_fish(ag): pass
def agent_gather(ag): pass
def agent_go_to_camp(ag): pass
def agent_leave_camp(ag): pass
def agent_move_north(ag): pass
def agent_move_south(ag): pass
def agent_move_east(ag): pass
def agent_move_west(ag): pass
def agent_talk(ag): pass
def agent_listen(ag): pass
def agent_deposit(ag): pass
def agent_withdraw(ag): pass
def agent_propose_mate(ag): pass
def agent_watch_children(ag): pass
def agent_idle(ag): pass
def agent_gift(ag): pass
def agent_make_camp(ag): pass
def agent_join_camp(ag): pass


# --- helpers to build chunks tersely ---
def ord_chunk(var, op, value):
    c = chunk()
    c.var, c.op, c.value = var, op, value
    return c


def bool_chunk(var, value=None):
    c = chunk()
    c.var, c.value = var, value
    return c


def make_seq(chunks, links):
    s = chunk_seq()
    s.chunks = list(chunks)
    s.chunk_links = list(links)
    return s


# --- hand-built decision graph: eat / forage / sleep cycle ---
#
# Walked top-down each hour:
#   1. tired_or_night_node — sleep if tired or it's nighttime
#   2. hungry_node         — if hungry and cache available, eat
#   3. forage_node         — if daytime, forage; else idle
#

# (1) sleep gate: tired > 0.7  OR  time > 20  OR  time < 5
sleep_seq = make_seq(
    [ord_chunk('tired', '>', 0.7),
     ord_chunk('time', '>', 20),
     ord_chunk('time', '<', 5)],
    [LINK_OR, LINK_OR],
)
tired_or_night_node = node()
tired_or_night_node.seq = sleep_seq
tired_or_night_node.true_action = agent_sleep
tired_or_night_node.true_end = True
tired_or_night_node.false_action = agent_wake          # wake (still in camp; watch first)
tired_or_night_node.false_end = False
# false_node assigned below once defined


# (1b) tired-OR-night wanderer make-camp gate: any wanderer about to
# sleep (because they're tired OR it's nighttime) must establish a camp
# first. Sleeping outside is the dominant cause of death in this sim —
# `death_mask` triggers exposure on `sleep & ~is_in_camp` regardless of
# age, so without this gate every wanderer who hits the sleep gate dies
# stochastically each night. This seq mirrors `sleep_seq` exactly with
# an added `is_in_camp == False` AND-gate, so the conditions for "make
# camp before sleeping" are the conditions for "would have slept."
# Left-fold: acc = (tired) OR (time>20) OR (time<5) AND (is_in_camp=False).
tired_make_camp_node = node()
tired_make_camp_node.seq = make_seq(
    [ord_chunk('tired', '>', 0.7),
     ord_chunk('time', '>', 20),
     ord_chunk('time', '<', 5),
     bool_chunk('is_in_camp', False)],
    [LINK_OR, LINK_OR, LINK_AND])
tired_make_camp_node.true_action = agent_make_camp
tired_make_camp_node.true_end = False
tired_make_camp_node.false_action = None
tired_make_camp_node.false_end = False

# (2) eat gate: hunger > 250 AND cache > 0
eat_seq = make_seq(
    [ord_chunk('hunger', '>', 250),
     ord_chunk('cache', '>', 0)],
    [LINK_AND],
)
hungry_node = node()
hungry_node.seq = eat_seq
hungry_node.true_action = agent_eat
hungry_node.true_end = False
hungry_node.false_action = None
hungry_node.false_end = False

# (3a) talk while in camp — pass through afterwards
talk_node = node()
talk_node.seq = make_seq([bool_chunk('is_in_camp')], [])
talk_node.true_action = agent_talk
talk_node.true_end = False
talk_node.false_action = None
talk_node.false_end = False

# (3b) listen while in camp
listen_node = node()
listen_node.seq = make_seq([bool_chunk('is_in_camp')], [])
listen_node.true_action = agent_listen
listen_node.true_end = False
listen_node.false_action = None
listen_node.false_end = False

# (3c) deposit when have a buffer
deposit_node = node()
deposit_node.seq = make_seq([ord_chunk('cache', '>', 500)], [])
deposit_node.true_action = agent_deposit
deposit_node.true_end = False
deposit_node.false_action = None
deposit_node.false_end = False

# (3d) withdraw when hungry
withdraw_node = node()
withdraw_node.seq = make_seq([ord_chunk('hunger', '>', 1500)], [])
withdraw_node.true_action = agent_withdraw
withdraw_node.true_end = False
withdraw_node.false_action = None
withdraw_node.false_end = False

# (3e) propose mating when in camp
mate_node = node()
mate_node.seq = make_seq([bool_chunk('is_in_camp')], [])
mate_node.true_action = agent_propose_mate
mate_node.true_end = False
mate_node.false_action = None
mate_node.false_end = False

# (3e2) gift when in camp + have surplus cache
gift_node = node()
gift_node.seq = make_seq(
    [bool_chunk('is_in_camp'),
     ord_chunk('cache', '>', 500)],
    [LINK_AND],
)
gift_node.true_action = agent_gift
gift_node.true_end = False
gift_node.false_action = None
gift_node.false_end = False

# (3f) watch children — fires regardless of camp location.
# watching_phase categorizes the watcher as in-camp vs out-camp via is_in_camp,
# so adults outside camp also protect any kids who tagged along.
watch_node = node()
watch_node.seq = make_seq([ord_chunk('hunger', '>', -1)], [])   # tautology
watch_node.true_action = agent_watch_children
watch_node.true_end = False
watch_node.false_action = None
watch_node.false_end = False

# (3c) leave-camp gate: affiliated AND (hungry OR random wanderlust).
# Two triggers ORed:
#   - hunger > 1500: "need to go forage" — same threshold as withdraw, so
#     agents have both options (raid the larder OR head into the wild) at
#     the same hunger pressure; mutation arbitrates.
#   - rng > 0.999: rare spontaneous wanderlust (~0.1%/visit ≈ 3%/hr).
# The whim rate is intentionally 10× stricter than make/join (0.99) so
# affiliation is a Markov sink — agents pour back into camps faster than
# they leave them. Protects kids (CHILD_OUT_CAMP_LIMIT=1) from cascading
# exposure when a parent gets wanderlust. Mutation can drift either gate.
in_camp_seq = make_seq([bool_chunk('is_in_camp')], [])
leave_camp_node = node()
# Seq: (hunger > 5000 OR rng > 0.9999) AND is_in_camp.
# Left-fold order: acc=hunger; acc = acc OR rng; acc = acc AND is_in_camp.
# Reduced from prior (1500 / 0.999) — wanderlust leak was dispersing
# agents too aggressively in the spatial sim; now they only leave when
# clearly hungry or on a truly rare (~0.01%/visit) whim.
leave_camp_node.seq = make_seq(
    [ord_chunk('hunger', '>', 5000),
     ord_chunk('rng', '>', 0.9999),
     bool_chunk('is_in_camp', True)],
    [LINK_OR, LINK_AND])
leave_camp_node.true_action = agent_leave_camp
leave_camp_node.true_end = False
leave_camp_node.false_action = None
leave_camp_node.false_end = False

# (4) forage variants: hunt, fish, gather — all gated by daylight + age + season.
# season > 0 is a near-tautology at non-winter-solstice days; mutators can shift
# the threshold to make foraging seasonal in any direction.
def _forage_seq(min_age):
    return make_seq(
        [ord_chunk('time', '>', 6),
         ord_chunk('time', '<', 19),
         ord_chunk('age', '>', min_age - 0.0001),
         ord_chunk('season', '>', 0.0)],
        [LINK_AND, LINK_AND, LINK_AND],
    )

hunt_node = node()
hunt_node.seq = _forage_seq(HUNT_AGE_YEARS)
hunt_node.true_action = agent_hunt
hunt_node.true_end = False
hunt_node.false_action = None
hunt_node.false_end = False

fish_node = node()
fish_node.seq = _forage_seq(FISH_AGE_YEARS)
fish_node.true_action = agent_fish
fish_node.true_end = False
fish_node.false_action = None
fish_node.false_end = False

gather_node = node()
gather_node.seq = _forage_seq(GATHER_AGE_YEARS)
gather_node.true_action = agent_gather
gather_node.true_end = False
gather_node.false_action = agent_go_to_camp       # off-hours: return to camp
gather_node.false_end = False

# (4b) make_camp gate: only wanderers (is_in_camp=False) can found a camp;
# fires on a random rng poll (~1% per visit). Initially stochastic — let
# memetic mutation evolve any structured triggers (e.g. drift toward
# hunger/age/season gates if those produce fitter cultures).
make_camp_node = node()
make_camp_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('rng', '>', 0.99996667)],
    [LINK_AND])
make_camp_node.true_action = agent_make_camp
make_camp_node.true_end = False
make_camp_node.false_action = None
make_camp_node.false_end = False

# (4c) join_camp gate: only wanderers (is_in_camp=False) can join an
# existing camp; fires on a random rng poll (~1% per visit). Initially
# stochastic — clustering vs nomadism is left for memetic mutation to
# evolve. With make_camp and join_camp both on identical 0.99 thresholds,
# the chain order (join_camp tried first, then make_camp) gives a mild
# clustering bias when at least one real camp exists.
join_camp_node = node()
join_camp_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('rng', '>', 0.99)],
    [LINK_AND])
join_camp_node.true_action = agent_join_camp
join_camp_node.true_end = False
join_camp_node.false_action = None
join_camp_node.false_end = False

# (4d) spatial movement — wanderers walk TOWARD visible camps. Each
# direction node fires when (a) wanderer (is_in_camp=False), (b) a camp
# is within vision range 2 in that direction (nearest_camp_X > 0; the
# sentinel -1 means "nothing"), and (c) a stochastic rng roll fires
# (~70%). With no camps anywhere in sight, no direction fires and the
# wanderer holds position. As a camp comes into vision, the agent drifts
# toward it. Mutation can drift these thresholds — e.g. an explorer
# lineage might drop the vision predicate and wander randomly again.
move_north_node = node()
move_north_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('nearest_camp_n', '>', 0),
     ord_chunk('rng', '>', 0.99)],
    [LINK_AND, LINK_AND])
move_north_node.true_action = agent_move_north
move_north_node.true_end = False
move_north_node.false_action = None
move_north_node.false_end = False

move_south_node = node()
move_south_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('nearest_camp_s', '>', 0),
     ord_chunk('rng', '>', 0.99)],
    [LINK_AND, LINK_AND])
move_south_node.true_action = agent_move_south
move_south_node.true_end = False
move_south_node.false_action = None
move_south_node.false_end = False

move_east_node = node()
move_east_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('nearest_camp_e', '>', 0),
     ord_chunk('rng', '>', 0.99)],
    [LINK_AND, LINK_AND])
move_east_node.true_action = agent_move_east
move_east_node.true_end = False
move_east_node.false_action = None
move_east_node.false_end = False

move_west_node = node()
move_west_node.seq = make_seq(
    [bool_chunk('is_in_camp', False),
     ord_chunk('nearest_camp_w', '>', 0),
     ord_chunk('rng', '>', 0.99)],
    [LINK_AND, LINK_AND])
move_west_node.true_action = agent_move_west
move_west_node.true_end = False
move_west_node.false_action = None
move_west_node.false_end = False


# (5) idle node — terminal sink; self-loops to consume any remaining budget
idle_node = node()
idle_node.seq = make_seq([ord_chunk('hunger', '>', -1)], [])   # tautology (always true)
idle_node.true_action = None
idle_node.true_end = False
idle_node.false_action = None
idle_node.false_end = False

# --- 3-clone canonical: each non-forage node group has 3 redundant copies ---
# helpers
def _make_clone(template):
    c = node()
    if template.seq:
        s2 = chunk_seq()
        for chk in template.seq.chunks:
            c2 = chunk()
            c2.var, c2.value, c2.op = chk.var, chk.value, chk.op
            s2.chunks.append(c2)
        s2.chunk_links = list(template.seq.chunk_links)
        c.seq = s2
    c.true_action = template.true_action
    c.false_action = template.false_action
    c.true_end = template.true_end
    c.false_end = template.false_end
    return c

def make_pass_clones(template, n_clones, fall_target):
    """n_clones in chain, both T/F → next clone (or fall_target for last)."""
    clones = [_make_clone(template) for _ in range(n_clones)]
    for i, c in enumerate(clones):
        nxt = clones[i+1] if i+1 < n_clones else fall_target
        c.true_node = nxt
        c.false_node = nxt
    return clones[0]

def make_choice_clones(template, n_clones, true_target, false_chain_target):
    """n_clones with T branch → true_target (always exits), F branch chains."""
    clones = [_make_clone(template) for _ in range(n_clones)]
    for i, c in enumerate(clones):
        c.true_node = true_target
        c.false_node = clones[i+1] if i+1 < n_clones else false_chain_target
    return clones[0]

N_CLONES = 3

# Build chain bottom-up so we have target references.
# idle group: 3 clones, mutual self-loop
idle_clones = [_make_clone(idle_node) for _ in range(N_CLONES)]
for i, c in enumerate(idle_clones):
    c.true_node = idle_clones[(i+1) % N_CLONES]
    c.false_node = idle_clones[(i+1) % N_CLONES]
idle_first = idle_clones[0]

# Settle-then-socialize chain (after eat / off-hours gather-skip)
# Order: eat → join_camp → make_camp → talk → listen → mate → deposit → gift → withdraw → idle.
# Wanderers first try to settle (join an existing camp, else found one);
# only once in-camp do the social/exchange nodes (which are all gated on
# is_in_camp) actually do anything. join_camp tried before make_camp gives
# a mild clustering bias when any local camp exists.
withdraw_first = make_pass_clones(withdraw_node, N_CLONES, idle_first)
gift_first = make_pass_clones(gift_node, N_CLONES, withdraw_first)
deposit_first = make_pass_clones(deposit_node, N_CLONES, gift_first)
mate_first = make_pass_clones(mate_node, N_CLONES, deposit_first)
listen_first = make_pass_clones(listen_node, N_CLONES, mate_first)
talk_first = make_pass_clones(talk_node, N_CLONES, listen_first)
make_camp_first = make_pass_clones(make_camp_node, N_CLONES, talk_first)
join_camp_first = make_pass_clones(join_camp_node, N_CLONES, make_camp_first)

# Eat group → settle (join/make camp) → social
eat_first = make_pass_clones(hungry_node, N_CLONES, join_camp_first)

# Forage chain: hunt → fish → gather, then eat.
hunt_node.true_node  = fish_node
hunt_node.false_node = fish_node
fish_node.true_node  = gather_node
fish_node.false_node = gather_node
gather_node.true_node  = eat_first         # forage succeeded → eat what you got
gather_node.false_node = join_camp_first   # off-hours: skip eat, settle then social

# Move block (sequential dice, first hit wins):
# Try N; if rng+vision fail, try S; if S fails, try E; if E fails, try W;
# if all fail, no move this hour. Each node's seq is
# (is_in_camp=False AND nearest_camp_X>0 AND rng>0.99) so the same chunk-eval
# (drawing a FRESH rng per visit) gives independent rolls per direction —
# moves are biased toward visible camps, and at most one direction fires per
# chain pass. T-branch jumps straight to the forage chain.
move_west_node.true_node  = hunt_node
move_west_node.false_node = hunt_node
move_east_node.true_node  = hunt_node
move_east_node.false_node = move_west_node
move_south_node.true_node = hunt_node
move_south_node.false_node = move_east_node
move_north_node.true_node = hunt_node
move_north_node.false_node = move_south_node

# Leave camp group → move-block (N first), then forage chain
leave_camp_first = make_pass_clones(leave_camp_node, N_CLONES, move_north_node)

# Watch group → leave_camp
watch_first = make_pass_clones(watch_node, N_CLONES, leave_camp_first)

# Sleep group: T branch loops within group (so csleep can build > warmup),
# F branch chains through clones then exits to watch
sleep_clones = [_make_clone(tired_or_night_node) for _ in range(N_CLONES)]
for i, c in enumerate(sleep_clones):
    c.true_node = sleep_clones[(i + 1) % N_CLONES]   # T loops inside the group
    c.false_node = sleep_clones[i + 1] if i + 1 < N_CLONES else watch_first
sleep_first = sleep_clones[0]

# Patch idle clones: chain forward, last clone redirects to the chain
# HEAD (tired_make_camp_node), not sleep_first — otherwise every cycle
# after agent birth skips the tired-wanderer gate and wanderers sleep
# in the wild instead of founding a camp first.
for i, c in enumerate(idle_clones):
    nxt = idle_clones[i + 1] if i + 1 < N_CLONES else tired_make_camp_node
    c.true_node = nxt
    c.false_node = nxt


# --- graph cloning (by node identity, handles shared/cyclic refs) ---
def clone_chunk(c):
    c2 = chunk()
    c2.var, c2.value, c2.op = c.var, c.value, c.op
    return c2


def clone_seq(s):
    s2 = chunk_seq()
    s2.chunks = [clone_chunk(c) for c in s.chunks]
    s2.chunk_links = list(s.chunk_links)
    return s2


def clone_graph(root):
    memo = {}

    def go(n):
        if n is None:
            return None
        if id(n) in memo:
            return memo[id(n)]
        n2 = node()
        memo[id(n)] = n2
        n2.seq = clone_seq(n.seq) if n.seq is not None else None
        n2.true_action = n.true_action
        n2.false_action = n.false_action
        n2.true_end = n.true_end
        n2.false_end = n.false_end
        n2.lineage = n.lineage          # heritable variant ID
        n2.true_node = go(n.true_node)
        n2.false_node = go(n.false_node)
        return n2

    return go(root)


def all_nodes(root):
    out, seen = [], set()

    def go(n):
        if n is None or id(n) in seen:
            return
        seen.add(id(n))
        out.append(n)
        go(n.true_node)
        go(n.false_node)

    go(root)
    return out


# --- mutation ---
ACTION_POOL = [agent_sleep, agent_wake, agent_eat,
               agent_hunt, agent_fish, agent_gather,
               agent_go_to_camp,
               agent_move_north, agent_move_south, agent_move_east, agent_move_west,
               agent_leave_camp,
               agent_talk, agent_listen,
               agent_deposit, agent_withdraw,
               agent_propose_mate, agent_watch_children,
               agent_idle, agent_gift,
               agent_make_camp, agent_join_camp, None]
ORDERED_OPS = ('<', '>', '==')
FLOAT_DELTA = {'hunger': 200, 'cache': 200, 'tired': 0.1, 'time': 1, 'age': 1.0, 'season': 0.1, 'rng': 0.05,
               'x': 1, 'y': 1,
               'nearest_camp_n': 1, 'nearest_camp_s': 1, 'nearest_camp_e': 1, 'nearest_camp_w': 1}


def mutate_chunk(c):
    if c.var in ORDERED_VARS:
        kind = random.choice(('inc', 'dec', 'op'))
        if kind == 'inc':
            c.value = (c.value or 0) + FLOAT_DELTA[c.var]
        elif kind == 'dec':
            c.value = (c.value or 0) - FLOAT_DELTA[c.var]
        else:
            c.op = random.choice([o for o in ORDERED_OPS if o != c.op])
    else:
        c.value = True if c.value is None else (not bool(c.value))


def random_chunk():
    """Generate a fresh random chunk."""
    var = random.choice(STATE_VARS)
    c = chunk()
    c.var = var
    if var in ORDERED_VARS:
        c.op = random.choice(ORDERED_OPS)
        if var == 'hunger':   c.value = random.uniform(0, 5000)
        elif var == 'cache':  c.value = random.uniform(0, 5000)
        elif var == 'tired':  c.value = random.uniform(0, 1)
        elif var == 'time':   c.value = random.randint(0, 23)
        elif var == 'age':    c.value = random.uniform(0, 50)
        elif var == 'season': c.value = random.uniform(0, 1)
        elif var == 'rng':    c.value = random.uniform(0, 1)
        elif var == 'x':      c.value = random.randint(0, GRID_W - 1)
        elif var == 'y':      c.value = random.randint(0, GRID_H - 1)
        elif var in ('nearest_camp_n', 'nearest_camp_s', 'nearest_camp_e', 'nearest_camp_w'):
            c.value = random.choice([-1] + list(range(1, VISION_RANGE + 1)))
    else:
        c.value = random.choice([None, True, False])
    return c


def mutate_seq(s):
    options = []
    if len(s.chunks) < MAX_SEQ_LEN:
        options.append('add_chunk')
        options.append('duplicate_chunk')
    if len(s.chunks) >= 2:
        options.append('swap_chunks')
        options.append('delete_chunk')
    if len(s.chunk_links) >= 1:
        options.append('mutate_link')
    if not options:
        return
    kind = random.choice(options)
    if kind == 'swap_chunks':
        i, j = random.sample(range(len(s.chunks)), 2)
        s.chunks[i], s.chunks[j] = s.chunks[j], s.chunks[i]
    elif kind == 'mutate_link':
        k = random.randrange(len(s.chunk_links))
        cur = s.chunk_links[k]
        s.chunk_links[k] = random.choice([x for x in range(16) if x != cur])
    elif kind == 'add_chunk':
        s.chunks.append(random_chunk())
        s.chunk_links.append(random.randrange(16))
    elif kind == 'duplicate_chunk':
        # copy a random existing chunk and append it
        src = s.chunks[random.randrange(len(s.chunks))]
        new_c = chunk()
        new_c.var, new_c.value, new_c.op = src.var, src.value, src.op
        s.chunks.append(new_c)
        s.chunk_links.append(random.randrange(16))
    elif kind == 'delete_chunk':
        i = random.randrange(len(s.chunks))
        del s.chunks[i]
        if s.chunk_links:
            del s.chunk_links[min(i, len(s.chunk_links) - 1)]


def mutate_node(n):
    kind = random.choice(('change_action', 'swap_actions', 'flip_end', 'duplicate'))
    if kind == 'change_action':
        side = random.choice(('true', 'false'))
        cur = n.true_action if side == 'true' else n.false_action
        new = random.choice([a for a in ACTION_POOL if a is not cur])
        if side == 'true':
            n.true_action = new
        else:
            n.false_action = new
    elif kind == 'swap_actions':
        n.true_action, n.false_action = n.false_action, n.true_action
    elif kind == 'flip_end':
        if random.random() < 0.5:
            n.true_end = not n.true_end
        else:
            n.false_end = not n.false_end
    elif kind == 'duplicate':
        # Insert a clone of n between n and its current targets:
        #   before:  n → (true_node, false_node)
        #   after:   n → n_copy → (true_node, false_node)
        n_copy = node()
        if n.seq:
            s2 = chunk_seq()
            for c in n.seq.chunks:
                c2 = chunk()
                c2.var, c2.value, c2.op = c.var, c.value, c.op
                s2.chunks.append(c2)
            s2.chunk_links = list(n.seq.chunk_links)
            n_copy.seq = s2
        n_copy.true_action = n.true_action
        n_copy.false_action = n.false_action
        n_copy.true_end = n.true_end
        n_copy.false_end = n.false_end
        n_copy.lineage = n.lineage      # duplicated slot inherits parent's lineage
        n_copy.true_node = n.true_node
        n_copy.false_node = n.false_node
        n.true_node = n_copy
        n.false_node = n_copy


def mutate_weights(ag):
    """Three flavors: PERTURB (gaussian additive), FLIP (negate), RESET (zero).
    Acts on a uniformly-chosen entry of either weights[i] or Q[i][j]."""
    idx = random.randrange(N_FEATURES + N_FEATURES * N_FEATURES)
    r = random.random()
    if r < WEIGHT_MUT_PERTURB_P:
        delta_op = 'perturb'
    elif r < WEIGHT_MUT_PERTURB_P + WEIGHT_MUT_FLIP_P:
        delta_op = 'flip'
    else:
        delta_op = 'reset'
    if idx < N_FEATURES:
        if delta_op == 'perturb':
            ag.weights[idx] += random.gauss(0.0, WEIGHT_MUT_STD)
        elif delta_op == 'flip':
            ag.weights[idx] = -ag.weights[idx]
        else:  # reset
            ag.weights[idx] = 0.0
    else:
        idx -= N_FEATURES
        i, j = idx // N_FEATURES, idx % N_FEATURES
        if delta_op == 'perturb':
            ag.Q[i][j] += random.gauss(0.0, WEIGHT_MUT_STD)
        elif delta_op == 'flip':
            ag.Q[i][j] = -ag.Q[i][j]
        else:
            ag.Q[i][j] = 0.0


def dual_graph(root):
    """Macro-mutation: take the structural dual of every node in the graph.
    For each node, swap (true_action ↔ false_action), (true_node ↔ false_node),
    (true_end ↔ false_end). Predicates left intact. Net effect: every
    if-then-else becomes if-then-else-with-branches-swapped, which is
    behaviorally equivalent to negating every node's seq. Drastic single event.
    """
    for n in all_nodes(root):
        n.true_action, n.false_action = n.false_action, n.true_action
        n.true_node,   n.false_node   = n.false_node,   n.true_node
        n.true_end,    n.false_end    = n.false_end,    n.true_end


def mutate(root, ag):
    if random.random() < P_MUT_WEIGHT:
        mutate_weights(ag)
        return
    nodes = all_nodes(root)
    if not nodes:
        return
    r = random.random()
    # post-weight split: chunk / seq / node / graph_dual.
    # Any content-changing mutation stamps a fresh lineage on the owning node
    # so the variant can be tracked through copies and assimilations.
    cum = P_MUT_CHUNK_FRAC
    if r < cum:
        owners = [(c, n) for n in nodes for c in (n.seq.chunks if n.seq else [])]
        if owners:
            c, owner = random.choice(owners)
            mutate_chunk(c)
            owner.lineage = _new_lineage()
        return
    cum += P_MUT_SEQ_FRAC
    if r < cum:
        owners = [n for n in nodes if n.seq is not None]
        if owners:
            owner = random.choice(owners)
            mutate_seq(owner.seq)
            owner.lineage = _new_lineage()
        return
    cum += P_MUT_NODE_FRAC
    if r < cum:
        n = random.choice(nodes)
        mutate_node(n)
        n.lineage = _new_lineage()
        return
    # remaining slice → whole-graph dual; every node is structurally changed
    dual_graph(root)
    for n in nodes:
        n.lineage = _new_lineage()


# --- social ranking ---
# Feature normalization (so quadratic terms don't blow up).
FEATURE_NORM = (
    1.0 / 10000.0,    # net_debt_flow
    1.0 / 72000.0,    # hunger
    1.0,              # tired (already [0,2])
    1.0,              # is_female (±1)
    1.0 / 10000.0,    # cache
    1.0 / 60.0,       # age
    1.0,              # is_pregnant (±1)
    1.0,              # is_menopausal (±1)
    MATING_NORM,      # matings_with_target — pairwise
    GIFT_NORM,        # gifts_with_target — pairwise (sum of bidirectional flow)
    FAMILY_NORM,      # family_similarity — cos(observer.crest, target.crest), ∈ [-1, 1]
    CAMP_CACHE_NORM,  # target_camp_cache — kcal in target's camp larder
)


def social_score(observer, target, camp_caches=None):
    """x^T Q x + c·x where x = normalized feature vector. Bool features encoded ±1.
    Features 9-11 are observer-target pairwise (mating, gift, family-similarity).
    Feature 12 is target-camp-cache (looked up via target's camp_id)."""
    matings = gifts = 0.0
    family_sim = 0.0
    if observer is not target:
        if hasattr(observer, '_mating_history'):
            matings = observer._mating_history.get(id(target), 0.0)
        if hasattr(observer, '_gift_history'):
            gifts = observer._gift_history.get(id(target), 0.0)
        if hasattr(observer, 'crest') and hasattr(target, 'crest'):
            family_sim = crest_similarity(observer.crest, target.crest)
    # Target's camp cache (look up in dict if provided, else fall back to legacy CAMP)
    if camp_caches is not None and hasattr(target, 'camp_id'):
        tcc = float(camp_caches.get(int(target.camp_id), 0.0))
    else:
        tcc = float(CAMP.get('cache', 0.0))
    f = (target.net_debt_flow * FEATURE_NORM[0],
         target.hunger * FEATURE_NORM[1],
         target.tired * FEATURE_NORM[2],
         (1.0 if target.is_female else -1.0) * FEATURE_NORM[3],
         target.cache * FEATURE_NORM[4],
         target.age * FEATURE_NORM[5],
         (1.0 if target.is_pregnant else -1.0) * FEATURE_NORM[6],
         (1.0 if target._is_menopausal else -1.0) * FEATURE_NORM[7],
         matings * FEATURE_NORM[8],
         gifts * FEATURE_NORM[9],
         family_sim * FEATURE_NORM[10],
         tcc * FEATURE_NORM[11])
    c = observer.weights
    Q = observer.Q
    linear = sum(c[i] * f[i] for i in range(N_FEATURES))
    quadratic = sum(Q[i][j] * f[i] * f[j]
                    for i in range(N_FEATURES) for j in range(N_FEATURES))
    return linear + quadratic


_LOG_HIGH_MUT = math.log(ADAPTIVE_MUT_HIGH)
_LOG_LOW_MUT = math.log(ADAPTIVE_MUT_LOW)

def adaptive_mut_rate(self_score, peer_mean_score):
    """Log-linear interpolation between LOW (high self-rank) and HIGH (low self-rank).
    Uses sigmoid of (self_score - peer_mean) to smoothly map to mutation rate."""
    delta = self_score - peer_mean_score
    if delta > 50: delta = 50
    elif delta < -50: delta = -50
    s = 1.0 / (1.0 + math.exp(-delta))   # 0..1, higher when self > peer
    log_rate = (1.0 - s) * _LOG_HIGH_MUT + s * _LOG_LOW_MUT
    return math.exp(log_rate)


def adaptive_adopt_rate(self_score, peer_mean_score):
    """Linear: high self-rank → 0% adoption, low self-rank → 30%."""
    delta = self_score - peer_mean_score
    if delta > 50: delta = 50
    elif delta < -50: delta = -50
    s = 1.0 / (1.0 + math.exp(-delta))
    return (1.0 - s) * ADAPTIVE_ADOPT_HIGH + s * ADAPTIVE_ADOPT_LOW


# --- memetic transmission ---
# Replace the *content* of `target` with the content of `source` while
# preserving target's outgoing edges (true_node/false_node). This swaps the
# decision-logic at a slot in the listener's graph without touching topology.
def assimilate_node(target, source):
    target.seq = clone_seq(source.seq) if source.seq is not None else None
    target.true_action = source.true_action
    target.false_action = source.false_action
    target.true_end = source.true_end
    target.false_end = source.false_end
    target.lineage = source.lineage     # memetic transmission carries the lineage


# --- serialization ---
def _action_name(fn):
    return None if fn is None else fn.__name__


def serialize_graph(root):
    nodes = all_nodes(root)
    idx = {id(n): i for i, n in enumerate(nodes)}
    out_nodes = []
    for n in nodes:
        out_nodes.append({
            'seq': {
                'chunks': [{'var': c.var, 'value': c.value, 'op': c.op}
                           for c in n.seq.chunks],
                'links': list(n.seq.chunk_links),
            } if n.seq is not None else None,
            'true_action': _action_name(n.true_action),
            'false_action': _action_name(n.false_action),
            'true_node': idx.get(id(n.true_node)) if n.true_node else None,
            'false_node': idx.get(id(n.false_node)) if n.false_node else None,
            'true_end': n.true_end,
            'false_end': n.false_end,
            'lineage': n.lineage,
        })
    return {'root': 0, 'nodes': out_nodes}


def serialize_agent(ag):
    return {
        'id': ag.id,
        'hunger': ag.hunger,
        'cache': ag.cache,
        'tired': ag.tired,
        'is_female': ag.is_female,
        'is_in_camp': ag.is_in_camp,
        'sleep': ag.sleep,
        'consecutive_sleep': ag.consecutive_sleep,
        'net_debt_flow': ag.net_debt_flow,
        'weights': list(ag.weights),
        'Q': [row[:] for row in ag.Q],
        'age': ag.age,
        'is_pregnant': ag.is_pregnant,
        'pregnancy_hours': ag._pregnancy_hours,
        'is_menopausal': ag._is_menopausal,
    }


def dump_population(path, day, agents, roots):
    payload = {
        'day': day,
        'n_agents': len(agents),
        'agents': [
            {'state': serialize_agent(agents[i]),
             'graph': serialize_graph(roots[i])}
            for i in range(len(agents))
        ],
    }
    with open(path, 'w') as f:
        json.dump(payload, f)


# --- population simulation with natural selection ---
# Chain head: tired-wanderer-make-camp → sleep group. Wanderers who get
# tired settle FIRST so they sleep in a camp rather than the wild.
tired_make_camp_node.true_node = sleep_first
tired_make_camp_node.false_node = sleep_first
canonical_root = tired_make_camp_node
# Stamp every node in the canonical seed with a distinct lineage so the
# initial population can be tracked back to each founder-slot variant.
for _n in all_nodes(canonical_root):
    _n.lineage = _new_lineage()
