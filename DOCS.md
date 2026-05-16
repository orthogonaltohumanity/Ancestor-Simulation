# anthrosim documentation

A walkthrough of the simulation, code section by code section, with explanations next to the code.

---

## Overview

`anthrosim` is a population-level agent simulation. Every agent runs a small interpreted program (a "decision graph") that decides what action they take each hour. Agents have:

- **biological state** — hunger, cache, tired, age, in_camp, sleep, pregnancy
- **a decision graph** — nodes referencing actions like `agent_eat`, `agent_forage`, `agent_propose_mate`, `agent_gift`, `agent_make_camp`, `agent_join_camp`, etc.
- **social weights** — an 11-dim linear vector + 11×11 quadratic matrix used to rank other agents
- **a family crest** — a 256-dim unit vector that makes kin recognizable and gates incest
- **a camp affiliation** — `camp_id` int; mating, gifting, communication, and watching are restricted to within-camp pairs. Camps form/dissolve dynamically via `agent_make_camp` / `agent_join_camp`

The hour-by-hour walk evaluates each agent's graph and produces actions. Mutation perturbs graphs; sexual reproduction averages parental weights and slerps parental crests; **memetic adoption** lets agents copy nodes from other agents into their own graphs. Selection prunes dysfunctional individuals. The whole system is designed to let cultural and behavioral patterns *emerge* from the basic mechanics.

There are three core Python files plus tooling:

| file | role |
|---|---|
| `main.py` | **library**: tunable constants, agent/node classes, canonical decision graph builder, mutation, social ranking, serialization. **Not directly runnable** — has no sim loop; the CPU reference loop was removed once the vectorized path stabilized. |
| `main_vec.py` | vectorized hot-loop: per-hour action dispatch, mating, watching, gifting, births, deaths, communication, sub-graph adoption. NumPy (or torch/cupy) arrays for all per-agent state. |
| `run_vec.py` | **entry point**: imports `main as ref` for constants + graph utilities and `main_vec as v` for the hot loop. Owns founder init, burn-in, the hour-by-hour driver, mutation step, and dump cadence. |
| `viz.py` | matplotlib charts: family clusters, cohort cohesion, population pyramid, resource histograms, weight heatmap, population trajectory, death-cause inference. |
| `analyze.py` | text-mode summaries over the dump JSON files (trajectory, cohorts, graphs). |
| `lineage_analysis.py` | population-genetics analyses on the memetic node-lineage system: SFS heatmap over time + selective-sweep trajectory detection. See §18. |
| `archetypes.py` | multi-subject archetypal analysis (ArchePy) over agent weight vectors — finds K archetypal value-coalitions across run history. *Requires `pip install archepy` to run.* |
| `graph_evo.py` | renders a single agent's decision graph as an animated gif across weekly dumps. *Requires `pip install networkx` to run.* |
| `viz_canonical.py` | one-shot render of the canonical founder graph. Depends on `graph_evo.py` (needs networkx). |

Run with `python3 run_vec.py`. Tunables live at the top of `main.py`. Architecture rule of thumb: if you're editing simulation *semantics*, look in `main_vec.py`. If you're editing *constants, the canonical graph, mutation/clone logic, or social scoring*, look in `main.py`. If you're editing the *driver, init, or dump cadence*, look in `run_vec.py`.

> **Note on line numbers.** Inline annotations like `(main.py:233)` in this doc were accurate at time of writing but drift as the codebase grows. Treat them as approximate pointers; trust function/class names over numeric refs. The constants in §1 are kept current with the canonical `main.py`.

---

## 1. Top-level constants

All tunables live at the top of `main.py`. **Note: many constants have been retuned over the project's life; numbers below match the current `main.py`. The principles are stable, the values aren't.**

### Walk geometry

```python
MAX_ACTION = 30   # nodes visited per hour-walk
MAX_SEQ_LEN = 10  # max chunks per seq
MAX_NODES = 10    # main.py cap; main_vec uses MAX_NODES = 250
```

`MAX_ACTION` is the **per-hour visit budget**. Each agent walks from their current `cur` (program counter) up to `MAX_ACTION` nodes per hour. With persistent `cur`, the walk continues across hours. A 30-step budget through a ~42-node canonical graph means agents complete a full canonical traversal in ~1.4 hours.

### Adaptive mutation / adoption rates

```python
MUTATION_RATE_PER_HOUR = 3.0 / 24.0           # default; overridden by adaptive
ADAPTIVE_MUT_HIGH = 1.0 / (3 *  24.0)         # max mut rate (low-self-rank)
ADAPTIVE_MUT_LOW  = 1.0 / (3 * 365.0 * 24.0)  # min mut rate (high-self-rank)

LISTEN_ADOPT_P = 0.30                         # default; overridden by adaptive
ADAPTIVE_ADOPT_HIGH = 1.0                     # low-self-rank cap (full copy)
ADAPTIVE_ADOPT_LOW  = 0.001                   # high-self-rank floor (~never)

# Cultural-memory parameters (see §9b)
FRESHNESS_MAX = 168                           # hours = 7 days
FRESHNESS_DECAY = 1                           # per hour
GC_INTERVAL_HOURS = 24                        # daily GC sweep
```

Per-hour probability that an agent's graph mutates / adopts. Both rates are **adaptive** — agents who self-rank below peer-mean (using their *own* social score function) hit `*_HIGH` (explore + copy), confident agents hit `*_LOW` (conserve genome). Mut interpolates in log-space, adopt linearly. See §14.

Each node also carries a per-agent freshness counter that ticks down by `FRESHNESS_DECAY` each hour and is reset to `FRESHNESS_MAX` whenever the agent's chain walk visits that slot. Nodes that go un-visited for the full window die at the next GC sweep. The chain still has them while alive; visiting them is what keeps them alive.

### Social ranking features (12-dim)

```python
SOCIAL_FEATURES = ('net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
                   'age', 'is_pregnant', 'is_menopausal',
                   'matings_with_target', 'gifts_with_target',
                   'family_similarity', 'target_camp_cache')
N_FEATURES = 12
```

Each agent has a **12-dim weight vector** + **12×12 Q matrix** for social ranking. The last four features are observer/target-conditional, not pure intrinsics of the target:

- `matings_with_target` — symmetric pair count (decays per hour)
- `gifts_with_target` — sum of bidirectional gift-flow kcal (decays per hour)
- `family_similarity` — `cos(observer.crest, target.crest)` ∈ [−1, +1]
- `target_camp_cache` — kcal in the camp the *target* belongs to (looked up via target's camp_id). Observer-independent in value but threaded through the same expand-with-pairwise pipeline. Lets agents prefer or avoid targets in well-stocked camps — and crucially, lets `join_camp` softmax distinguish camps by larder.

```python
MATING_DECAY = 0.5 ** (1.0 / (30 * 24))      # 30-day half-life
MATING_NORM  = 1.0 / 10.0                    # feature normalizer
GIFT_AMOUNT  = 1000                          # kcal moved per gift action
GIFT_DECAY   = 0.5 ** (1.0 / (30 * 24))      # 30-day half-life
GIFT_NORM    = 1.0 / 1000.0
FAMILY_NORM  = 1.0                           # cosine is already in [-1, 1]
CAMP_CACHE_NORM = 1e-6                       # 1M kcal larder → feature value 1.0
```

### Weight-mutation flavors

```python
WEIGHT_INIT_STD     = 0.5
WEIGHT_MUT_STD      = 0.1
P_MUT_WEIGHT        = 0.20    # 20% of all mutations are weight; rest are graph

# Within a weight mutation:
WEIGHT_MUT_PERTURB_P = 0.70   # additive Gaussian
WEIGHT_MUT_FLIP_P    = 0.15   # negate the chosen weight
WEIGHT_MUT_RESET_P   = 0.15   # set the chosen weight to 0
```

Picks one of 132 entries (11 linear + 121 Q) uniformly, then applies one of three operations. Flip and reset are *discrete* — they enable rapid cultural drift not reachable by Gaussian walks alone.

### Mutation dispatch ratios

```python
P_MUT_CHUNK_FRAC = 58/80      # 58% of all events
P_MUT_SEQ_FRAC   = 12/80      # 12%
P_MUT_NODE_FRAC  =  9/80      #  9%
P_MUT_DUAL_FRAC  =  1/80      #  1%  (whole-graph dual macro)
```

These shares are **of the 80% non-weight slice**, so absolute proportions are 20% weight / 58% chunk / 12% seq / 9% node / 1% graph dual.

### Family crests (kinship)

```python
FAMILY_CREST_DIM       = 256   # dim of unit-vector crest
FAMILY_INCEST_THRESHOLD = 0.6  # mating refused if cos(a, b) > this
```

Every agent carries a 256-d unit vector. Initial agents get random unit vectors; children inherit `slerp(mom, dad, 0.5)`. The mating phase refuses pairs with cos > 0.6 — at this threshold, **siblings, parent-child, and aunt/uncle-nephew/niece are blocked**; half-sibs, first cousins, and grandparent-grandkid pass. Random unrelated pairs land cos ~ N(0, 1/√256) ≈ N(0, 0.063), well below threshold. See §15.

### Energy & metabolism

```python
EAT_RATE = 1000                  # kcal moved per eat action
NODE_VISIT_HUNGER = 3            # per-visit cost
KID_METAB_FRAC = 300.0 / 2400.0  # newborn metabolism fraction (12.5%)
```

Per-visit hunger cost is `NODE_VISIT_HUNGER * metabolic_scale(age)`. For a walking adult, `3 × 1.0 = 3 kcal/visit`, times `MAX_ACTION = 30` visits/hour = 90/hour = ~2160/day. Kids ramp from 12.5% at birth to 100% at `METABOLIC_ADULT_AGE = 15`.

### Cache / debt decay

```python
CACHE_LIMIT = 12000                          # max personal kcal store
CACHE_DECAY = 0.5 ** (1.0 / (10.0*24.0))     # half-life 10 days
NDF_DECAY   = 0.5 ** (1.0 / (30 * 24))       # net_debt_flow half-life 30d
DEPOSIT_AMOUNT = WITHDRAW_AMOUNT = 1000
CAMP = {'cache': 0.0}                        # shared in-camp larder
```

Both personal cache and `net_debt_flow` (signed deposit-vs-withdraw tally) decay. Without decay, ndf grew unbounded and broke the social-score sigmoid that gates adoption rates.

### Resource pools (logistic-growth game/fish/plants)

```python
RESOURCE_K_HUNT   = 5_000_000.0   # carrying capacity (kcal) of game pool
RESOURCE_K_FISH   = 5_000_000.0
RESOURCE_K_GATHER = 5_000_000.0
LOGISTIC_GROWTH_R = 0.005         # per-hour intrinsic growth (logistic r)
MIN_RESOURCE_FRAC = 0.05          # pool can't fall below this fraction of K
```

Each forage type extracts from its own pool. Per-hour growth: `R += r * R * (1 - R/K)`. Pool depletion changes the harvest model in a real-world-inspired way:

- **`agent_hunt` scales SUCCESS PROBABILITY by `R/K`**: a kill still yields a full carcass (`HUNT_YIELD`), but encounters are rarer when the pool is depleted ("animals get harder to find"). Effective fire rate is `HUNT_SUCCESS_P * (R/K)`.
- **`agent_fish` and `agent_gather` scale YIELD by `R/K`**: catches are easy but small when stocks are low. Effective yield = `YIELD * seasonal * (R/K)`.

Extraction is subtracted from R, clipped at the `MIN_RESOURCE_FRAC * K` floor. **Resources can never go extinct** — even under sustained over-harvest, the floor keeps the pool alive and the seasonal/logistic dynamics restore it once pressure relents.

Held in the driver as a `resources = {'hunt': float, 'fish': float, 'gather': float}` dict, threaded through `step_all` → `_apply_actions`. Logistic growth applied once per hour via `resource_growth_step(resources)`. Persisted in dumps as `payload['resources']`.

### Seasonal forage scaling and per-action params

```python
HUNT_WINTER_FACTOR   = 1.0   # × in winter (1.0 = unaffected)
FISH_WINTER_FACTOR   = 1.0
GATHER_WINTER_FACTOR = 1.0

# Per-action success prob and summer yield:
HUNT_SUCCESS_P   = 0.01      # rare jackpot
HUNT_YIELD       = CACHE_LIMIT   # fills cache: cache <- max(cache, HUNT_YIELD * f)
FISH_SUCCESS_P   = 0.3333
FISH_YIELD       = 6000      # additive
GATHER_SUCCESS_P = 1.0
GATHER_YIELD     = 1000      # additive
```

Yields scale with season: `factor = WINTER + (1 - WINTER) * season`, where `season ∈ [0, 1]` is `sin²(π · day_of_year / 365)`. **Hunt is "fill" semantics** (raises cache toward a target), fish/gather are additive. Setting any `WINTER_FACTOR < 1.0` makes that activity seasonal.

### Selection pressure / burn-in

```python
N_AGENTS = 280                   # current run size
K_SAMPLE = 200                   # listener/proposer top-K candidates
BURN_IN_ASEX_CYCLES = 0          # phase 1 disabled in current configs
BURN_IN_TEST_HOURS = 144         # 6-day solo test
BURN_IN_HUNGER_LIMIT = 500       # post-test cull threshold
BURN_IN_MIN_CAMP_FRAC = 0.25     # min in-camp fraction during test
BURN_IN_SEXUAL = 0               # phase 2 disabled in current configs
BURN_IN_CACHE_DIR = 'burnin_cache'
```

When phases are non-zero: **Phase 1** runs each cycle as `mutate-once + 72-144h solo-walk test`, culling agents whose hunger exceeds `BURN_IN_HUNGER_LIMIT` *or* whose in-camp fraction drops below `BURN_IN_MIN_CAMP_FRAC`. **Phase 2** does ranking-based sexual reproduction. Recent runs skip both phases and let the live sim do the selection.

### Death thresholds

```python
HUNGER_DEATH = 60 * 2000              # 120,000 kcal accumulated → death
TIRED_DEATH  = 1.0                    # tired ≥ 1.0 → forced 12h pass-out (not death)
SLEEP_EXPOSED_DEATH_P = 1.0/100000.0  # per-hour base risk sleeping outside camp
SLEEP_EXPOSED_WINTER_FACTOR = 1.0     # winter multiplier (1.0 = no extra winter risk)
HAZARD_PER_HOUR = 0.005 / HOURS_PER_YEAR   # ambient annual hazard
MAX_AGE_YEARS = 120                   # hard cap (rarely reached in practice)
```

### Demography (per-action ages)

```python
HUNT_AGE_YEARS   = 13   # age to start hunting
FISH_AGE_YEARS   =  8   # age to start fishing
GATHER_AGE_YEARS =  4   # age to start gathering
FORAGE_AGE_YEARS = GATHER_AGE_YEARS  # earliest forage age (also kid leave-camp gate)
WATCH_AGE_YEARS  = 6    # can leave camp solo
METABOLIC_ADULT_AGE = 15
MATE_AGE_YEARS   = 16   # propose mating

PREGNANCY_HOURS = 9 * 30 * 24      # 9-month mean
PREGNANCY_HOURS_RANGE = 20 * 24    # ± stochastic spread
PREGNANCY_FORAGE_FRACTION = 0.33   # forage permitted first 1/3 of pregnancy
PREGNANCY_BIRTH_P = 0.75           # P(pregnancy | het mating)
INIT_AGE_LO  = 0                   # founder age range
INIT_AGE_HI  = 25
INIT_AGE_DECAY_TAU = 8.0           # τ in years for truncated-exponential
                                   # founder age distribution (smaller →
                                   # more young founders). 0 → uniform.
TWIN_BETA = 4.0                    # P(N+1 babies | N) = exp(-beta)
MENOPAUSE_AGE = 40
MENOPAUSE_BETA = 0.10
WATCH_FEED = 500                   # hunger reduction per watch
CHILD_IN_CAMP_LIMIT  = 3           # die after 3h unwatched in camp
CHILD_OUT_CAMP_LIMIT = 1           # die after 1h unwatched outside camp
```

Forage activities have **separate age thresholds** — gathering opens at age 4, fishing at 8, hunting at 13. The canonical forage-gate seq encodes this per-action (each of `hunt_node`/`fish_node`/`gather_node` checks `age > X-ε`). Combined with the leave-camp gate (kids 4-6 may leave only with an adult outside), this creates a graded developmental sequence.

Pregnancy lasts 270 days ± 20; pregnant women can still forage during the first ~3 months.

---

## 2. Core data structures (`main.py`)

### `class chunk`

```python
class chunk:                                                                      (main.py:131-135)
    def __init__(self):
        self.var = None      # which state variable to read
        self.value = None    # comparison value
        self.op = None       # '<', '>', '==' for ordered vars; None for bool
```

A single boolean predicate over agent state. E.g. `hunger > 250` is `var='hunger', op='>', value=250`. For boolean variables (`is_female`, `is_in_camp`, `sleep`) `op` is None and `value` is True/False/None (the None case means "just bool(state[var])").

The pseudo-var **`rng`** draws a fresh uniform `[0, 1)` at evaluation time — used like `rng > 0.99` for a ~1%-per-visit stochastic gate. It's an ordered var (lives in `ORDERED_VARS`); mutation can drift its threshold by `FLOAT_DELTA['rng'] = 0.05` per step. Within a single `_eval_seqs` call all rng chunks see the same per-agent draw (correlated), so multi-chunk rng seqs in one node are perfectly correlated — not a problem for single-chunk gates.

### `class chunk_seq`

```python
class chunk_seq:                                                                  (main.py:138-141)
    def __init__(self):
        self.chunks = []
        self.chunk_links = []   # ints in 0..15, length == len(chunks)-1
```

A left-folded boolean expression. `chunk_links[k]` is a **truth-table id** in `[0, 15]` that combines the running accumulator with `chunks[k+1]`. The 16 ids cover all binary boolean operators (AND, OR, XOR, NAND, NOR, A, B, NOT_A, etc.). See `LINK_AND = 8`, `LINK_OR = 14`, etc. at lines 168-181.

This means a sequence can express any boolean expression over its chunks — and mutation can change a single link to flip the operator type.

### `class node`

```python
class node:                                                                       (main.py:144-150)
    def __init__(self):
        self.seq = None
        self.true_action = None
        self.false_action = None
        self.true_node = None
        self.false_node = None
        self.true_end = True
        self.false_end = True
```

A decision-graph node:
- evaluate `seq` to a bool `cond`
- fire `true_action` or `false_action` based on `cond`
- advance `cur` to `true_node` or `false_node`
- `true_end`/`false_end` flags exist but the walk is currently bounded by `MAX_ACTION` rather than these.

### `class agent`

```python
class agent:                                                                      (main.py:152-180)
    def __init__(self):
        self.hunger = 0.0
        self.cache = 0.0
        self.is_female = random.random() < 0.5
        self.tired = 0.0
        self.is_in_camp = True
        self.sleep = False
        self.consecutive_sleep = 0
        self.net_debt_flow = 0.0
        self.weights = [random.gauss(0.0, WEIGHT_INIT_STD) for _ in range(N_FEATURES)]
        self.Q = [[random.gauss(0.0, 0.1) for _ in range(N_FEATURES)] for _ in range(N_FEATURES)]
        ...
        self._mut_rate_per_hour = MUTATION_RATE_PER_HOUR
        self._adopt_rate = LISTEN_ADOPT_P
        self._forced_sleep_hours = 0
        self.season = 0.0              # set each hour from sim time
```

Every agent owns:
- biological state (hunger, cache, tired, age...)
- two genotype components: `weights` (9-vector) and `Q` (9×9 matrix), used by `social_score`
- adaptive rates (mut/adopt) recomputed daily based on self-rank
- forced-sleep counter (12-hour pass-out timer when tired hits cap)

Note: in the vec sim, the **flat numpy state arrays** (in `main_vec.py`) are the source of truth for scalar state; agent objects store graph references and are kept loosely in sync.

---

## 3. The 16 boolean operators (`main.py`)

```python
def _make_link_table():                                                           (main.py:148-156)
    table = []
    for op_id in range(16):
        def f(a, b, _id=op_id):
            k = (int(bool(a)) << 1) | int(bool(b))
            return bool((_id >> k) & 1)
        table.append(f)
    return table
APPLY_LINK = _make_link_table()
```

Each truth-table id is a 4-bit number where bit `k` is the output for input `(a,b)` encoded as `(a<<1)|b`. So `op_id=8` has bit 3 set → `1 only when (a,b)=(1,1)` → AND. `op_id=14` → bits 1,2,3 set → OR. There's a named alias for each:

```python
LINK_FALSE = 0; LINK_AND = 8; LINK_A_AND_NOT_B = 4; LINK_A = 12;
LINK_NOT_A_AND_B = 2; LINK_B = 10; LINK_XOR = 6; LINK_OR = 14;
LINK_NOR = 1; LINK_XNOR = 9; LINK_NOT_B = 5; LINK_A_OR_NOT_B = 13;
LINK_NOT_A = 3; LINK_NOT_A_OR_B = 11; LINK_NAND = 7; LINK_TRUE = 15;
```

The mutation operator that picks a new random link can flip an `AND` to an `OR` or even to `XOR` — this is how complex boolean predicates can evolve from the canonical `tired > 0.7 OR night` into any other expression.

---

## 4. Evaluation engine (`main.py`)

### `eval_chunk(c, ag, hour)`

```python
def eval_chunk(c, ag, hour):                                                      (main.py:185-198)
    val = hour if c.var == 'time' else getattr(ag, c.var)
    if c.var in ORDERED_VARS:
        if c.op == '<':  return val < c.value
        if c.op == '>':  return val > c.value
        if c.op == '==': return val == c.value
    if c.value is None:
        return bool(val)
    return bool(val) == bool(c.value)
```

Reads the agent attribute named `c.var` (special-casing `time` since it's per-hour, not per-agent) and applies the comparison. `season` is treated like an ordered var; it's stamped on each agent each hour by the sim driver.

### `eval_seq(seq, ag, hour)`

```python
def eval_seq(seq, ag, hour):                                                      (main.py:201-206)
    acc = eval_chunk(seq.chunks[0], ag, hour)
    for i, link in enumerate(seq.chunk_links):
        b = eval_chunk(seq.chunks[i + 1], ag, hour)
        acc = APPLY_LINK[link](acc, b)
    return acc
```

Left-fold: start with the first chunk's bool, combine with each subsequent chunk via the corresponding link operator. Note this is left-associative and has no precedence — a `chunk1 AND chunk2 OR chunk3` is parsed as `(chunk1 AND chunk2) OR chunk3`.

### Per-hour walk

The per-hour CPU `step()` function that used to live in `main.py` has been **removed** — the vectorized `step_all` in `main_vec.py` is now the only runtime walker. `eval_seq` (and its inner `eval_chunk`) survive in `main.py` because the burn-in's solo-walk test (`run_vec.py:_burn_in_asex_cycle`) still uses them on individual agents.

The conceptual per-visit cycle is unchanged: accrue `NODE_VISIT_HUNGER * metabolic_scale(age)` hunger, evaluate the seq, fire the appropriate action, reset sleep streak unless the action was `agent_sleep`, advance to `true_node` or `false_node`. See `main_vec.py:step_all` for the vectorized form.

---

## 5. Action functions (`main.py`)

There are 16 actions (15 agent operations + `None` no-op). **Status note:** since the CPU `step()` was retired, the action function bodies in `main.py` are now `pass`-only stubs — they exist solely as identity markers in `ACTION_POOL` so mutation can pick/swap them and serialization can write their names. The actual *semantics* live in `main_vec.py:apply_actions` (line ~418+), which masks the population on the action enum (`A_SLEEP`, `A_EAT`, `A_FORAGE`, …) and applies the effect in bulk. The pseudocode below documents the intended behavior — `main_vec.py` is the source of truth.

### `agent_sleep`

```python
def agent_sleep(ag):                                                              (main.py:233-238)
    ag.consecutive_sleep = ag.consecutive_sleep + 1 if ag.sleep else 1
    ag.sleep = True
    ag.is_in_camp = True   # voluntary sleep returns to camp
    if ag.consecutive_sleep > SLEEP_WARMUP:
        ag.tired = max(0.0, ag.tired - SLEEP_RECOVERY)
```

Sleeping recovers `tired` by `SLEEP_RECOVERY = 0.3` per visit, **after** a `SLEEP_WARMUP = 4` visit warmup. Sleeping always returns the agent to camp (this is how leave-camp/forage cycles always end up back home — reaching idle → redirected to sleep_first → eventually sleep fires).

### `agent_eat`

```python
def agent_eat(ag):                                                                (main.py:246-252)
    if ag.cache <= 0: return
    amount = min(EAT_RATE, ag.cache)
    ag.cache -= amount
    ag.hunger = max(0.0, ag.hunger - amount)
    ag.sleep = False
    ag.consecutive_sleep = 0
```

Converts cache to negative hunger at `EAT_RATE = 1000` kcal per visit.

### Forage actions (`agent_hunt`, `agent_fish`, `agent_gather`)

Each action has its **own age threshold and yield params**. Common gates: must be outside camp, age ≥ that activity's threshold, not late-pregnant (`can_forage_pregnant`).

```python
def agent_hunt(ag):
    eligible = (not ag.is_in_camp) and (ag.age >= HUNT_AGE_YEARS) and can_forage_pregnant(ag)
    if eligible and random.random() < HUNT_SUCCESS_P:
        f = HUNT_WINTER_FACTOR + (1.0 - HUNT_WINTER_FACTOR) * ag.season
        ag.cache = min(CACHE_LIMIT, max(ag.cache, HUNT_YIELD * f))   # FILL semantics

def agent_fish(ag):
    eligible = (not ag.is_in_camp) and (ag.age >= FISH_AGE_YEARS) and can_forage_pregnant(ag)
    if eligible and random.random() < FISH_SUCCESS_P:
        f = FISH_WINTER_FACTOR + (1.0 - FISH_WINTER_FACTOR) * ag.season
        ag.cache = min(CACHE_LIMIT, ag.cache + FISH_YIELD * f)       # additive

def agent_gather(ag):
    eligible = (not ag.is_in_camp) and (ag.age >= GATHER_AGE_YEARS) and can_forage_pregnant(ag)
    if eligible and random.random() < GATHER_SUCCESS_P:
        f = GATHER_WINTER_FACTOR + (1.0 - GATHER_WINTER_FACTOR) * ag.season
        ag.cache = min(CACHE_LIMIT, ag.cache + GATHER_YIELD * f)
```

Default config: hunt is rare-and-fill (1% × CACHE_LIMIT), fish is mid-rate-additive (33% × 6000), gather is always-on (100% × 1000). Per-activity age thresholds (4/8/13) create a developmental sequence where children take up activities in order over years.

### `agent_propose_mate`

```python
def agent_propose_mate(ag):                                                       (main.py:357-363)
    if ag.age < MATE_AGE_YEARS or not ag.is_in_camp:
        return
    if ag.is_female and ag.is_pregnant:
        return
    ag._mate_request = True
```

Just sets a request flag. The actual matchmaking happens in `mating_phase` (in `main_vec.py:mating_phase`), which scans all agents with `_mate_request=True`, ranks them by social score, and pairs them.

### `agent_deposit` / `agent_withdraw`

```python
def agent_deposit(ag):                                                            (main.py:329-336)
    if not ag.is_in_camp: return
    amt = min(DEPOSIT_AMOUNT, ag.cache)
    ag.cache -= amt
    CAMP['cache'] += amt
    ag.net_debt_flow += amt
```

Move kcal to/from the agent's *own camp's* larder. The single legacy `CAMP` global in `main.py` is only consulted as a fallback inside `social_score`; the live multi-camp pipeline maintains per-camp caches as `camp_caches: dict[camp_id → float]` inside `main_vec.py`. **`net_debt_flow` is the bookkeeping variable** — positive = generous depositor, negative = chronic withdrawer. It feeds the `social_score` 1st feature, so depositors and withdrawers look "salient" to peers (and the quadratic Q term punishes large magnitudes in either direction).

### `agent_watch_children`

```python
def agent_watch_children(ag):                                                     (main.py:367-371)
    ag.sleep = False
    ag.consecutive_sleep = 0
    ag._is_watching = True
```

Sets the watching flag. The `watching_phase` then categorizes watchers as in-camp or out-camp based on `is_in_camp`, feeds them and the children, and increments `unwatched_hours` for unwatched kids. After `CHILD_IN_CAMP_LIMIT=3` hours unwatched in camp (or `CHILD_OUT_CAMP_LIMIT=1` outside), the kid dies of neglect.

---

## 6. The canonical decision graph (`main.py`)

The starting graph for every fresh agent. Three-clone redundancy on most node groups so mutation has spare copies to evolve without breaking core behavior.

### Sleep gate

```python
sleep_seq = make_seq(                                                              (main.py:415-419)
    [ord_chunk('tired', '>', 0.7),
     ord_chunk('time', '>', 20),
     ord_chunk('time', '<', 5)],
    [LINK_OR, LINK_OR],
)
tired_or_night_node = node()                                                       (main.py:420-425)
tired_or_night_node.seq = sleep_seq
tired_or_night_node.true_action = agent_sleep
tired_or_night_node.false_action = agent_wake
```

"If tired > 0.7 OR night (time > 20 or < 5), sleep. Otherwise wake." Three clones of this gate are at the front of the canonical graph, so agents who reach the sleep gate while tired will T-loop within the group and accumulate consecutive sleep visits.

### Forage gate (per-action ages)

```python
def _forage_seq(min_age):
    return make_seq(
        [ord_chunk('time', '>', 6),
         ord_chunk('time', '<', 19),
         ord_chunk('age', '>', min_age - 0.0001),
         ord_chunk('season', '>', 0.0)],
        [LINK_AND, LINK_AND, LINK_AND],
    )

hunt_node.seq   = _forage_seq(HUNT_AGE_YEARS)    # age > 12.9999
fish_node.seq   = _forage_seq(FISH_AGE_YEARS)    # age > 7.9999
gather_node.seq = _forage_seq(GATHER_AGE_YEARS)  # age > 3.9999
```

"Daylight (6 < time < 19) AND old enough for THIS activity AND season > 0." Each forage node carries its own age threshold. Mutators can drift the season threshold to make foraging seasonal in any direction.

### The full chain

```
sleep[0..2] ──► watch[0..2] ──► leave_camp[0..2] ──► hunt ──► fish ──► gather
                                                                          │
                                                                          ├─T─► eat[0..8] ──► talk[0..2]
                                                                          └─F─►   talk[0..2]
                              talk[0..2] ──► listen[0..2] ──► mate[0..2]
                                ──► gift[0..2] ──► deposit[0..2] ──► withdraw[0..2]
                                ──► join_camp[0..2] ──► make_camp[0..2] ──► idle[0..2] ──► tired_make_camp ──► sleep_first  (loop)
```

`tired_make_camp` is the chain root: a single node fired once per cycle with seq `[is_in_camp == False, tired > 0.7]` (LINK_AND). When a wanderer gets tired they `make_camp` first, then on the next visit reach the sleep gate and pass out in the freshly-founded camp — the "build a bivouac before sleeping" rule. Affiliated agents pass straight through to sleep.

Order: **forage runs FIRST, then eat**. After leaving camp, agents go hunt → fish → gather. The gather node branches: T (gathered something) → eat[0..8] (9 eat clones) → social phase; F (off-hours/etc) → straight to social phase. The 9 eat clones form a long eat chain so any cached agent gets several attempts to convert cache → fed.

After the productive walk the agent reaches **idle**, whose true/false edges redirect to `sleep_first` so the program counter resets each cycle. With persistent `cur` and `MAX_ACTION = 30`, an agent walks the ~42-node canonical graph in ~1.4 hours.

### Idle redirect

```python
for i, c in enumerate(idle_clones):                                                (main.py:594-597)
    nxt = idle_clones[i + 1] if i + 1 < N_CLONES else sleep_first
    c.true_node = nxt
    c.false_node = nxt
```

Critical fix: idle clones used to self-loop within their group, which became a death trap once `cur` became persistent. They now chain forward to root.

---

## 7. Mutation

Five mutation tiers, with rates **20% weight / 58% chunk / 12% seq / 9% node / 1% graph_dual**:

```python
def mutate(root, ag):
    if random.random() < P_MUT_WEIGHT:                # 20%
        mutate_weights(ag); return
    nodes = all_nodes(root)
    r = random.random()
    cum = P_MUT_CHUNK_FRAC                            # 58/80
    if r < cum:
        mutate_chunk(random.choice([c for n in nodes for c in (n.seq.chunks if n.seq else [])]))
        return
    cum += P_MUT_SEQ_FRAC                             # 12/80
    if r < cum:
        mutate_seq(random.choice([n.seq for n in nodes if n.seq]))
        return
    cum += P_MUT_NODE_FRAC                            # 9/80
    if r < cum:
        mutate_node(random.choice(nodes)); return
    # remaining 1/80 slice
    dual_graph(root)
```

### Chunk mutation (`mutate_chunk`)

```python
def mutate_chunk(c):                                                               (main.py:660-673)
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
```

Either nudges a threshold (e.g., `hunger > 250` → `hunger > 450`), changes an op (`>` → `<`), or for boolean vars cycles the value. Most common mutation type.

### Seq mutation

```python
def mutate_seq(s):                                                                 (main.py:706-734)
    options = []
    if len(s.chunks) < MAX_SEQ_LEN:
        options.append('add_chunk'); options.append('duplicate_chunk')
    if len(s.chunks) >= 2:
        options.append('swap_chunks'); options.append('delete_chunk')
    if len(s.chunk_links) >= 1:
        options.append('mutate_link')
    ...
```

Add/dup/swap/delete chunks within a seq, or change a link operator. `mutate_link` is how an `AND` becomes an `OR` or `XOR`.

### Node mutation

```python
def mutate_node(n):                                                                (main.py:737-770)
    kind = random.choice(('change_action', 'swap_actions', 'flip_end', 'duplicate'))
    if kind == 'duplicate':
        # Insert a clone of n between n and its current targets:
        #   before:  n → (true_node, false_node)
        #   after:   n → n_copy → (true_node, false_node)
        ...
```

Most interesting is `duplicate`, which **adds a node** to the graph. Combined with `change_action`, this is how genuinely new behaviors evolve.

### Weight mutation (3 flavors)

Picks one entry uniformly from the 132 (= 11 linear + 121 Q) weight slots, then applies one of three operations:

```python
def mutate_weights(ag):
    idx = random.randrange(N_FEATURES + N_FEATURES**2)
    r = random.random()
    if r < WEIGHT_MUT_PERTURB_P:           # 70% → +N(0, 0.1)
        op = 'perturb'
    elif r < WEIGHT_MUT_PERTURB_P + WEIGHT_MUT_FLIP_P:   # 15% → negate
        op = 'flip'
    else:                                  # 15% → reset to 0
        op = 'reset'
    # ... apply op to weights[idx] or Q[i][j]
```

**Perturb** is small Gaussian drift (slow random walk). **Flip** negates the entry (instant sign reversal — preferences invert). **Reset** zeros it (turn off that feature's influence). Flip and reset are *discrete* and enable rapid, large cultural shifts that pure Gaussian drift would take generations to reach. Empirically these are responsible for the faster cultural reorganization seen in current runs vs the original Gaussian-only weight system.

### Whole-graph dual (`dual_graph`)

The 1% slice of mutation events. Operates on the entire graph in one event:

```python
def dual_graph(root):
    for n in all_nodes(root):
        n.true_action, n.false_action = n.false_action, n.true_action
        n.true_node,   n.false_node   = n.false_node,   n.true_node
        n.true_end,    n.false_end    = n.false_end,    n.true_end
```

Swaps T/F branches on every node simultaneously. Predicates left intact. Net effect: every "if X then A else B" becomes "if X then B else A" — behaviorally equivalent to negating every seq's evaluation. **Involutive**: dual twice = identity. A drastic single event that radically reshapes graph behavior; populations will split by parity of cumulative dual events.

---

## 8. Social ranking

```python
FEATURE_NORM = (
    1.0 / 10000.0,    # net_debt_flow
    1.0 / 72000.0,    # hunger
    1.0,              # tired
    1.0,              # is_female (±1)
    1.0 / 10000.0,    # cache
    1.0 / 60.0,       # age
    1.0,              # is_pregnant (±1)
    1.0,              # is_menopausal (±1)
    MATING_NORM,      # matings_with_target — pairwise (observer × target)
    GIFT_NORM,        # gifts_with_target — pairwise, sum of bidirectional flow
    FAMILY_NORM,      # family_similarity — cos(observer.crest, target.crest)
)
```

Each feature has its own normalization so the quadratic Q term doesn't explode. The last **three** features (indices 8, 9, 10) are **observer-target pairwise** — they depend on both ends of the relationship and are read at score time:

- `matings_with_target` — looked up in the symmetric `mating_matrix` (decays per hour).
- `gifts_with_target` — sum of `gift_matrix[obs, target] + gift_matrix[target, obs]` (directional matrix, but the feature is the bidirectional sum).
- `family_similarity` — `cos(observer.crest, target.crest)`.

```python
def social_score(observer, target):
    matings = gifts = family_sim = 0.0
    if observer is not target:
        if hasattr(observer, '_mating_history'):
            matings = observer._mating_history.get(id(target), 0.0)
        if hasattr(observer, '_gift_history'):
            gifts = observer._gift_history.get(id(target), 0.0)
        if hasattr(observer, 'crest') and hasattr(target, 'crest'):
            family_sim = crest_similarity(observer.crest, target.crest)
    f = (target.net_debt_flow * FEATURE_NORM[0],
         target.hunger        * FEATURE_NORM[1],
         target.tired         * FEATURE_NORM[2],
         (1.0 if target.is_female else -1.0)     * FEATURE_NORM[3],
         target.cache         * FEATURE_NORM[4],
         target.age           * FEATURE_NORM[5],
         (1.0 if target.is_pregnant else -1.0)   * FEATURE_NORM[6],
         (1.0 if target._is_menopausal else -1.0)* FEATURE_NORM[7],
         matings    * FEATURE_NORM[8],
         gifts      * FEATURE_NORM[9],
         family_sim * FEATURE_NORM[10])
    linear    = sum(observer.weights[i] * f[i] for i in range(N_FEATURES))
    quadratic = sum(observer.Q[i][j] * f[i] * f[j]
                    for i in range(N_FEATURES) for j in range(N_FEATURES))
    return linear + quadratic
```

For self-scoring (in `update_adaptive_rates`), all three pairwise features are zeroed — otherwise self-cosine = 1.0 would dominate the self-rank signal.

`score = c·x + xᵀQx` where `x` is the normalized feature vector. The quadratic term lets agents express **interactions** like "pregnant × generous" or "female × matings_with_target". This is what makes the system express *preference patterns* not just additive feelings.

---

## 9. Vectorized hot loop (`main_vec.py`)

The reference Python walk is too slow for N=1500 over 2000 sim-years. `main_vec.py` re-encodes the per-hour walk as numpy operations.

### Flat graph layout

```python
MAX_NODES = 500                   # hard array cap (main_vec.py)
ADOPTION_CAP = 500                # adoption skips when listener at this cap
                                  # (no REPLACE — see communication_phase)
K_GRAFT = 10                      # BFS depth-limit for subgraph adoption
```

Each agent's graph is a row in `(N, MAX_NODES, MAX_SEQ_LEN)` arrays:
- `seq_var[i, k, s]` — variable index (0..8) for chunk s of node k of agent i
- `seq_op[i, k, s]` — operator code (0:<, 1:>, 2:==, 3:bool)
- `seq_value[i, k, s]` — comparison value
- `seq_links[i, k, s]` — truth-table id linking chunk s and s+1
- `seq_len[i, k]` — actual number of chunks
- `true_action[i, k]`, `false_action[i, k]` — action index 0..15
- `true_node[i, k]`, `false_node[i, k]` — next-node indices

```python
def flatten_graph(root, max_nodes=MAX_NODES, max_seq=MAX_SEQ_LEN):                 (main_vec.py:49-99)
    nodes = ref.all_nodes(root)
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    idx = {id(n): i for i, n in enumerate(nodes)}
    ...
```

Convert one agent's object graph into row-format. **Truncates** if the graph exceeds `max_nodes` (rare; mutation duplicate can push past `ADOPTION_CAP`).

### Vectorized chunk evaluation

```python
def _eval_seqs(seq_var, seq_op, seq_value, seq_links, seq_len, state_mat):         (main_vec.py:212-237)
    N, S = seq_var.shape
    ar = np.arange(N)
    safe_var = np.where(seq_var >= 0, seq_var, 0).astype(np.intp)
    val = state_mat[ar[:, None], safe_var]   # (N, S)
    cr_lt = (seq_op == OP_LT) & (val <  seq_value)
    cr_gt = (seq_op == OP_GT) & (val >  seq_value)
    cr_eq = (seq_op == OP_EQ) & (val == seq_value)
    val_b = val != 0.0
    target_b = seq_value > 0.5
    cr_bn = (seq_op == OP_BOOL) & (seq_value < 0) & val_b
    cr_be = (seq_op == OP_BOOL) & (seq_value >= 0) & (val_b == target_b)
    cr = cr_lt | cr_gt | cr_eq | cr_bn | cr_be       # (N, S)
    acc = cr[:, 0].copy()
    for k in range(S - 1):
        b = cr[:, k + 1]
        idx = (acc.astype(np.uint8) << 1) | b.astype(np.uint8)
        new = TT[seq_links[:, k].astype(np.intp), idx].astype(bool)
        active = k < (seq_len - 1)
        acc = np.where(active, new, acc)
    return acc                                       # (N,)
```

Builds a `(N, S)` chunk-result matrix in one shot, then folds over the S dimension applying truth-table operators. The for-loop is bounded by `S=10` regardless of N, so this is essentially O(N) per call.

### `step_all` — the per-hour walk for everyone

```python
def step_all(graphs, s, hour, season_val, camp_cache, rng, n_nodes_arr=None,       (main_vec.py:380-432)
             budget=MAX_ACTION):
    N = s['hunger'].shape[0]
    walking = s['forced_sleep_hours'] == 0
    cur = s['cur'].astype(np.intp)
    if n_nodes_arr is not None:
        cur = np.where(cur < n_nodes_arr, cur, 0)
    visit_cost = ref.NODE_VISIT_HUNGER * metabolic_scale_vec(s['age'])

    for _ in range(budget):
        s['hunger'] = np.where(walking, s['hunger'] + visit_cost, s['hunger'])
        state_mat = _build_state_matrix(s, hour, season_val, N)
        sv = graphs['seq_var'][ar, cur]
        ...
        cond = _eval_seqs(...)
        action = np.where(cond, graphs['true_action'][ar, cur],
                                  graphs['false_action'][ar, cur])
        camp_cache = _apply_actions(action, s, camp_cache, rng,
                                     walking=walking, season_val=season_val)
        new_cur = np.where(cond, graphs['true_node'][ar, cur],
                                  graphs['false_node'][ar, cur]).astype(np.intp)
        cur = np.where(walking, new_cur, cur)
        if n_nodes_arr is not None:
            cur = np.where(cur < n_nodes_arr, cur, 0)

    s['cur'] = cur.astype(np.int16)
    return camp_cache
```

Walks all N agents `budget` times in lockstep. Forced-sleep agents have `walking=False` and their cur, hunger, and actions are gated off. The 33-step-budget loop body is constant-cost per step; the inner `_eval_seqs` and `_apply_actions` are vectorized over the N agents.

### `_apply_actions` — vectorized action dispatch

```python
def _apply_actions(action, s, camp_cache, rng, walking=None, season_val=1.0):     (main_vec.py:255-358)
    ...
    not_sleep = walking & (action != A_SLEEP)
    s['sleep'] = np.where(not_sleep, False, s['sleep'])
    s['consecutive_sleep'] = np.where(not_sleep, 0, s['consecutive_sleep'])

    m = walking & (action == A_SLEEP)
    new_cs = np.where(s['sleep'], s['consecutive_sleep'] + 1, 1)
    s['consecutive_sleep'] = np.where(m, new_cs, s['consecutive_sleep'])
    s['sleep'] = np.where(m, True, s['sleep'])
    s['is_in_camp'] = np.where(m, True, s['is_in_camp'])
    recovered = m & (s['consecutive_sleep'] > ref.SLEEP_WARMUP)
    s['tired'] = np.where(recovered, np.maximum(0.0, s['tired'] - ref.SLEEP_RECOVERY), s['tired'])

    m = walking & (action == A_EAT) & (s['cache'] > 0)
    amt = np.minimum(ref.EAT_RATE, s['cache'])
    s['cache']  = np.where(m, s['cache'] - amt, s['cache'])
    s['hunger'] = np.where(m, np.maximum(0.0, s['hunger'] - amt), s['hunger'])

    fish_factor = np.float32(ref.FISH_WINTER_FACTOR + (1.0 - ref.FISH_WINTER_FACTOR) * season_val)
    m = (action == A_FISH) & forage_ok & (rng.random(N) < 0.05)
    s['cache'] = np.where(m, np.minimum(ref.CACHE_LIMIT, s['cache'] + 6000 * fish_factor), s['cache'])
    ...
```

Every action's effect is encoded as a masked numpy update. `walking` gates everything so forced-sleep agents become no-ops. Order matters: the `not_sleep` mask resets sleep state before `agent_sleep` re-establishes it, so the right thing happens for both walking-and-sleeping and walking-and-acting agents.

---

## 10. Two-phase burn-in (`run_vec.py`)

Before the actual sim starts, founders go through evolutionary pre-conditioning.

### Phase 1: asexual mutation + per-agent solo-walk test

```python
def _starvation_test_72h(root, test_ag, hours=BURN_IN_TEST_HOURS):                 (run_vec.py:107-148)
    """Run a solo walk over `hours` hours and return final hunger.
    Tests whether the graph regulates feeding well enough to survive alone."""
    test_ag.hunger = 0.0
    test_ag.cache = ref.CACHE_LIMIT
    ...
    cur_node = root
    for t in range(hours):
        hour = t % 24
        test_ag.tired += ref.TIRED_PER_HOUR
        test_ag.cache *= ref.CACHE_DECAY
        test_ag.season = ref.season(t)
        if test_ag._forced_sleep_hours == 0:
            cur_node = _solo_step_persistent(test_ag, root, hour, cur_node)
        # forced sleep trigger + override (mirrors main loop semantics)
        ...
    return test_ag.hunger


def _burn_in_asex_cycle(agents, roots, rng_py, test_ag):                           (run_vec.py:151-167)
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
```

Each cycle iterates every agent and:
1. apply ONE mutation
2. run a 72-hour solo simulation: mirror the main loop's per-hour behavior (passive drift, persistent-cur walk, forced-sleep handling) but with no other agents involved
3. **fail-fast on hunger**: if it ever exceeds `BURN_IN_HUNGER_LIMIT`, cull immediately → replace with a clone of a random other agent
4. **after the test, check time-in-camp**: if `in_camp_hours / total_hours < BURN_IN_MIN_CAMP_FRAC` (default 0.20), also cull. This prevents the obligate-forager failure mode where an agent stays out of camp the whole time, has perfect hunger, but never returns home to mate.

The fail-fast is intentional: agents must keep hunger below threshold *the entire test*, not just at the end. The camp-fraction requirement is intentional too: solo-fitness selection alone optimized for permanent foragers who never came home — adding 20% camp-time grounds them socially.

Run for `BURN_IN_ASEX_CYCLES` cycles. Each agent receives that many mutation-and-test rounds; bad mutations get reverted (via clone-replacement), good ones accumulate. The thresholds are tuned to discriminate functional graphs (low peak hunger AND meaningful in-camp time) from broken or anti-social ones.

### Phase 2: ranking-based sexual reproduction

```python
def _burn_in_sex(agents, roots, n_iterations, rng_py):                             (run_vec.py:75-118)
    for it in range(n_iterations):
        i = rng_py.randint(0, N - 1)
        ag_i = agents[i]
        opposite = [k for k in range(N) if k != i and agents[k].is_female != ag_i.is_female]
        if not opposite: continue
        scores = [ref.social_score(ag_i, agents[k]) for k in opposite]
        # softmax-pick j
        ...
        for _ in range(2):
            ch = ref.agent()
            ch.weights = [(mw + dw) / 2.0 for mw, dw in zip(mom.weights, dad.weights)]
            ch.Q = [[(mom.Q[a][b] + dad.Q[a][b]) / 2.0 for b in range(ref.N_FEATURES)] for a in range(ref.N_FEATURES)]
            ch_root = ref.clone_graph(roots[mom_idx] if rng_py.random() < 0.5 else roots[dad_idx])
            for _ in range(3): ref.mutate(ch_root, ch)
            new_pair.append((ch, ch_root))
        agents[i], roots[i] = new_pair[0]
        agents[j], roots[j] = new_pair[1]
```

Run 2000 reproductive events:
1. random `i`
2. softmax-pick opposite-sex `j` using `i`'s social ranking
3. produce 2 averaged-genome offspring, mutate them lightly
4. **kill both parents**, replace with offspring

Selection pressure: agents whose `social_score` ranks high in others' eyes reproduce more. Their preferences propagate.

After both phases, `init_population` finally assigns starting ages, in-camp status, and initial pregnancies.

---

## 11. The main hour loop (`run_vec.py`)

```python
for t in range(T):                                                                 (run_vec.py:339-)
    hour = t % 24
    N = state['hunger'].shape[0]
    if N == 0: print(f"extinction at hour {t}"); break

    if hour == 0:
        v.update_adaptive_rates(state, weights, Q, rng, mating_matrix=mating_matrix)

    passive_drift(state)                       # tired+=, cache decay, ndf decay, age+=
    menopause_step(state, rng)                 # stochastic menopause for women > 40

    mating_matrix *= np.float32(ref.MATING_DECAY)

    season_val = ref.season(t)
    camp_cache = v.step_all(graphs, state, hour, season_val, camp_cache, rng, n_nodes)
    camp_cache *= ref.CACHE_DECAY

    v.forced_sleep_step(state)

    nt, nl, na = v.communication_phase(state, graphs, n_nodes, weights, Q, rng,
                                        roots=roots, mating_matrix=mating_matrix)

    n_mat, dad_pairs, mating_events = v.mating_phase(state, weights, Q, rng,
                                                      mating_matrix=mating_matrix)
    for a, b in mating_events:
        mating_matrix[a, b] += 1.0
        mating_matrix[b, a] += 1.0
    snapshot_dad_for_pregnancies(agents, roots, weights, Q, dad_pairs)

    camp_cache = v.watching_phase(state, camp_cache)
    mutation_step(roots, agents, graphs, n_nodes, weights, Q, state, rng)
    out = births_step(agents, roots, state, weights, Q, graphs, n_nodes, mating_matrix, rng)
    if isinstance(out, tuple):
        n_births, n_nodes, weights, Q, mating_matrix = out

    dead, why = death_mask_with_counts(state, rng, season_val=season_val)
    weights, Q, n_nodes, mating_matrix = compact_dead(
        agents, roots, state, weights, Q, graphs, n_nodes, mating_matrix, dead)
```

Order of operations per hour:
1. **adaptive rates** (only at hour 0): each agent computes their self-rank vs peer-mean and updates `mut_rate` and `adopt_rate`
2. **passive drift**: hunger += metabolism, tired += per-hour, cache *= decay, ndf *= decay, age += hour
3. **menopause roll**: women over 40 stochastically become menopausal
4. **mating matrix decay**: pairwise mating counts decay one hour
5. **step_all**: all agents walk their graphs (skipped for forced-sleep)
6. **camp cache decay**
7. **forced_sleep_step**: agents who hit `tired ≥ TIRED_DEATH` get a 12-hour pass-out timer; clamps tired to 0
8. **communication_phase**: listeners adopt nodes from talkers
9. **mating_phase**: ranking-based mate selection produces mating events + pregnancies
10. **dad snapshot**: capture dad's genotype on mom for later birth
11. **watching_phase**: kids in/out of camp watched or accumulate unwatched_hours
12. **mutation_step**: per-agent dice roll for graph mutation
13. **births_step**: pregnancies that hit their stochastic target produce 1+ babies
14. **death_mask + compact_dead**: kill the dead, contract all parallel arrays

---

## 12. Phases in detail

### `update_adaptive_rates` (`main_vec.py`)

```python
def update_adaptive_rates(s, weights, Q, rng, mating_matrix=None, K=ref.K_SAMPLE):
    F = features_vec(s)
    self_scores = social_score_self(weights, Q, F)
    K = min(K, N)
    peer_idx = rng.integers(0, N, size=(N, K))
    obs_idx = np.arange(N)
    peer_scores = social_score_pairs(weights, Q, F, peer_idx,
                                      mating_matrix=mating_matrix, obs_idx=obs_idx)
    peer_means = peer_scores.mean(axis=1)
    mut, adopt = adaptive_rates(self_scores, peer_means)
    s['mut_rate']   = mut
    s['adopt_rate'] = adopt
```

Each agent samples K random peers, scores themselves and each peer using their own (weights, Q). The delta `self_score - peer_mean` runs through a sigmoid to produce a value in [0, 1] that interpolates between the LOW/HIGH bounds for both rates.

### `communication_phase` (`main_vec.py`)

Cultural transmission is **K=10 subgraph adoption** (no longer single-node).
Each accepting listener clones a whole BFS subgraph from the chosen talker —
modeling the idea that culture spreads as *rituals* / coordinated multi-node
practices, not isolated atoms.

Flow:
1. **find talkers and listeners** — agents who fired `agent_talk` /
   `agent_listen` actions this hour while in camp.
2. **each talker exposes a random node from their graph** (the "topic").
3. **each listener (per their adopt_rate) softmax-picks a talker** using
   their own social-ranking weights (cross-camp talkers are −∞'d so
   adoption stays within-camp).
4. **Subgraph adoption**:
   - BFS up to `K_GRAFT=10` nodes outward from the exposed talker-node in
     the talker's Python object graph.
   - If listener has headroom (`n_nodes + K ≤ MAX_NODES`):
       - Deep-clone the K-node subgraph into the listener's object graph.
         Internal edges (target inside the K-set) are wired to the
         corresponding clone. External "exit" edges (target outside the
         K-set) are rewired to the listener's *splice-target* — the node
         at the other end of the edge we're about to redirect.
       - Splice listener's chosen edge to point at the subgraph entry.
   - If listener is at the cap: **the adoption silently skips**. No
     REPLACE — earlier versions overwrote random existing nodes, but
     that was destroying basal-survival nodes (eat/sleep) and
     bottlenecking the population. Now adoption is purely additive,
     and slot reclamation is handled by cultural-memory decay (§9b).
5. **Sync flat ↔ object**: `reflatten_row` re-DFS's the listener's graph
   into the numpy arrays. `cur` is re-anchored via Python-node identity.
   `node_freshness` is carried through via `reindex_freshness` (§9b).

The re-flatten is critical — `flatten_graph` uses DFS order from root, so
flat indices change after each adoption. Re-anchoring `cur` and freshness
via Python-node identity preserves both the agent's program counter and
its cultural-memory state through these renumberings.

### 9b. Cultural memory: per-node freshness + daily GC

Every graph node has a per-agent `freshness` counter (`s['node_freshness']`,
shape `(N, MAX_NODES)`, int16). Refreshed to `FRESHNESS_MAX=168` (one week
of hours) every time the agent's chain-walk visits the slot in `step_all`'s
hot loop. Decayed by `FRESHNESS_DECAY=1` per hour everywhere else.

Once per simulated day (`GC_INTERVAL_HOURS=24`), `gc_dead_nodes` prunes
every slot whose freshness has decayed to 0 (except the chain root at
slot 0, which is preserved structurally):

1. For each agent, identify dead-slot Python nodes.
2. Redirect every incoming edge (in `roots[i]`) that targets a dead node
   to instead target the dead node's `true_node` — skipping it.
3. Re-flatten the row. Unreachable-from-root dead nodes drop out of the
   new DFS, freeing slots and shrinking `n_nodes_arr[i]`.
4. `reindex_freshness` carries the surviving slots' freshness through
   to the new DFS layout via Python-node identity.
5. `cur` is re-anchored the same way.

This is the **forgotten-because-unwalked** mechanism. Nodes the agent's
chain trajectory never actually reaches age out; nodes on the
walked-every-cycle critical path stay fresh. Combined with the no-REPLACE
adoption rule, it gives the population a regulated cultural-turnover
mechanism: practices that are *used* persist, practices that are *not*
fade, slots reopen for new adoptions to splice in. Mirrors real cultural
disuse extinction (you forget rituals you stop performing).

Caveat: nothing protects basal-survival actions (eat, sleep, hunt, etc.)
from freshness decay specifically — if a chain trajectory happens to
route around them for `FRESHNESS_MAX` hours, those nodes will be reaped.
Mutation's `swap_actions` mode is the more dangerous failure path here
(turning eat into a no-op rather than removing it), and selection on
mutation rate is left to handle it.

### `mating_phase` (with incest gate)

```python
def mating_phase(s, weights, Q, rng, mating_matrix=None, gift_matrix=None,
                 crests=None, K=ref.K_SAMPLE):
    proposers = np.flatnonzero(s['mate_request'])
    if proposers.size <= 1: return 0, [], []
    F_base = features_vec(s)

    for p in proposers:
        pool = proposers[proposers != p]
        # INCEST GATE — pre-filter the pool by kin similarity, BEFORE scoring.
        # The softmax is taken over the actually-available (non-kin) set, so a
        # proposer doesn't waste their hour just because their highest-ranked
        # candidate happened to be a sibling.
        if crests is not None:
            kin_sims = crests[pool] @ crests[int(p)]
            pool = pool[kin_sims <= ref.FAMILY_INCEST_THRESHOLD]
            if pool.size == 0: continue
        # p ranks remaining candidates by social score; pick top-K, softmax-sample
        sp_all = _score_against_pool(int(p), pool, F_base, weights, Q,
                                       mating_matrix, gift_matrix=gift_matrix,
                                       crests=crests)
        chosen = top_k_softmax(sp_all, pool, K, rng)

        # chosen's accept pool — also pre-filtered by chosen's own kin-gate
        # (symmetric to p's, since cos is commutative — p is guaranteed
        # to be in chosen's filtered pool)
        # ... softmax accept probability over their top-K, then if accepted:
        # record mating event, possibly trigger pregnancy
```

Three-stage selection: (1) **incest pre-filter** drops kin from the candidate pool, (2) proposer's ranking softmax-picks one of the remaining via top-K, (3) chosen's reciprocal accept-or-reject (also using their kin-filtered pool). Only opposite-sex pairs whose mom is fertile (not pregnant, not menopausal) and pass `PREGNANCY_BIRTH_P = 0.75` produce pregnancies. Same-sex pairs still record mating events (which feed the pairwise mating-history feature).

    return n_resolved, dad_pairs, mating_events
```

Notable design choices:
- **same-sex pairs allowed** — they still get recorded in `mating_events` and feed the mating_matrix, but only opposite-sex pairs can produce pregnancy
- **Top-K by ranking, not random sampling** — popular agents get all the proposals, creating winner-take-all dynamics
- **Pregnancy only at 10% per successful mating** — most matings produce no offspring, which keeps fertile females in the mating market longer

### `forced_sleep_step` (`main_vec.py`)

```python
def forced_sleep_step(s):
    trigger = (s['tired'] >= ref.TIRED_DEATH) & (s['forced_sleep_hours'] == 0)
    s['forced_sleep_hours'] = np.where(trigger, 12, s['forced_sleep_hours'])
    forced = s['forced_sleep_hours'] > 0
    if not forced.any(): return
    s['sleep'] = np.where(forced, True, s['sleep'])
    s['consecutive_sleep'] = np.where(forced, s['consecutive_sleep'] + 1,
                                       s['consecutive_sleep'])
    # tired hard-clamped to 0 each hour during forced sleep
    s['tired'] = np.where(forced, 0.0, s['tired'])
    # forced agents pay metabolism here since they skip step_all's per-visit cost
    metab = ref.NODE_VISIT_HUNGER * ref.MAX_ACTION * metabolic_scale_vec(s['age'])
    s['hunger'] = np.where(forced, s['hunger'] + metab, s['hunger'])
    s['forced_sleep_hours'] = np.where(forced, s['forced_sleep_hours'] - 1,
                                        s['forced_sleep_hours'])
```

When `tired >= TIRED_DEATH`, the agent is "passed out" for 12 hours. They're forced asleep, can't walk their graph, and their tired is clamped to 0 each hour. They still pay full metabolism (the 30 kcal/hour cost is added explicitly here since `step_all` skipped them). Where the agent passes out matters — if outside camp, they're vulnerable to seasonal exposure death.

### `watching_phase`

Watchers feed kids from their **own personal cache** (not the shared camp larder). Watchers donate but **do not eat** their own donation — kid-feeding is asymmetric. Kids' demand is capped at their current hunger so no kcal is wasted on full kids.

```python
def watching_phase(s, camp_cache):
    fa = ref.WATCH_AGE_YEARS                # 6
    is_kid = s['age'] < fa
    in_camp_w  = s['is_watching'] & s['is_in_camp']
    out_camp_w = s['is_watching'] & ~s['is_in_camp']
    in_camp_k  = is_kid & s['is_in_camp']
    out_camp_k = is_kid & ~s['is_in_camp']

    # mark watched kids
    if in_camp_w.any():  s['unwatched_hours'][in_camp_k] = 0
    if out_camp_w.any(): s['unwatched_hours'][out_camp_k] = 0

    def _feed_pool(watchers_mask, kids_mask):
        if not (watchers_mask.any() and kids_mask.any()): return
        kid_demand = np.minimum(ref.WATCH_FEED, s['hunger'][kids_mask])
        need = float(kid_demand.sum())
        if need <= 0: return
        pool = float(s['cache'][watchers_mask].sum())
        avail = min(need, pool)
        if avail <= 0: return
        # watchers donate uniform fraction of their cache
        donate_frac = np.float32(avail / pool)
        s['cache'][watchers_mask] *= (1.0 - donate_frac)
        ratio = np.float32(avail / need)
        s['hunger'][kids_mask] = np.maximum(0.0,
            s['hunger'][kids_mask] - kid_demand * ratio)

    _feed_pool(in_camp_w,  in_camp_k)
    _feed_pool(out_camp_w, out_camp_k)
    # unwatched-hours accumulation (sleeping kids are safe in camp)
    ...
```

**Implication:** the kid-feeding economy depends on adults running successful forage cycles. Watchers who watch a lot while not foraging will drain their own cache and starve, while still keeping kids fed. This creates a **selfless-watcher fitness penalty** that didn't exist when feeding came from the camp larder.

Unwatched kids accumulate `unwatched_hours`; they die at `CHILD_IN_CAMP_LIMIT = 3` (or `CHILD_OUT_CAMP_LIMIT = 1` outside).

### `mutation_step` (`run_vec.py:206-238`)

```python
def mutation_step(roots, agents, graphs, n_nodes, weights, Q, s, rng):
    fire = rng.random(N) < s['mut_rate']
    idx = np.flatnonzero(fire)
    for i in idx:
        ag = agents[i]
        ag.weights = list(weights[i])
        ag.Q = [list(row) for row in Q[i]]
        old_a_nodes = ref.all_nodes(roots[i])
        old_cur = int(s['cur'][i])
        cur_pynode = old_a_nodes[old_cur] if old_cur < len(old_a_nodes) else None
        ref.mutate(roots[i], ag)
        weights[i] = ag.weights
        Q[i] = ag.Q
        nn = v.reflatten_row(graphs, i, roots[i])
        n_nodes[i] = nn
        # re-anchor cur via python-node identity (mutations don't delete nodes)
        if cur_pynode is not None:
            new_a = ref.all_nodes(roots[i])
            new_cur = 0
            for k, n in enumerate(new_a):
                if n is cur_pynode:
                    new_cur = k; break
            s['cur'][i] = new_cur
```

Same identity-preservation trick as adoption. Mutation operates on the python object graph; we re-flatten the row afterwards and remap `cur` via the python-node identity.

### `births_step` (`run_vec.py:243-303`)

```python
def births_step(agents, roots, state, weights, Q, graphs, n_nodes, mating_matrix, rng):
    moms = np.flatnonzero(state['is_female'] & state['is_pregnant'])
    state['pregnancy_hours'][moms] += 1
    due = moms[state['pregnancy_hours'][moms] >= state['pregnancy_target'][moms]]
    if due.size == 0: return 0
    
    for mom_i in due:
        m = agents[mom_i]
        n_babies = 1
        while rng.random() < math.exp(-ref.TWIN_BETA): n_babies += 1
        for _ in range(n_babies):
            child = ref.agent()
            ...
            if m._dad_graph_snapshot is not None and rng.random() < 0.5:
                child_root = ref.clone_graph(m._dad_graph_snapshot)
            else:
                child_root = ref.clone_graph(roots[mom_i])
            ...
            cw = (mw + dw) / 2.0
            cQ = (mQ + dQ) / 2.0
            new_agents.append(child)
            new_roots.append(child_root)
            new_flats.append(v.flatten_graph(child_root))
        # extend mating_matrix with zero rows + columns for newborns
        ...
```

Each pregnant mom advances by 1 hour; if past her stochastic target, deliver. Twin probability follows `exp(-TWIN_BETA)` per extra. Babies inherit:
- weights = (mom + dad) / 2
- Q = (mom + dad) / 2
- graph = 50/50 clone of mom's or dad's-at-mating snapshot

Then state arrays, weights, Q, n_nodes, mating_matrix all extend by `n_babies` rows/cols.

### `death_mask` (`main_vec.py`)

```python
def death_mask(s, rng, season_val=1.0):
    N = s['hunger'].shape[0]
    hazard = rng.random(N) < ref.HAZARD_PER_HOUR
    oldage = s['age'] > ref.MAX_AGE_YEARS
    hunger = s['hunger'] > ref.HUNGER_DEATH
    exposed_p = ref.SLEEP_EXPOSED_DEATH_P * (
        ref.SLEEP_EXPOSED_WINTER_FACTOR
        + (1.0 - ref.SLEEP_EXPOSED_WINTER_FACTOR) * season_val)
    exposed = s['sleep'] & ~s['is_in_camp'] & (rng.random(N) < exposed_p)
    is_kid = (s['age'] < ref.WATCH_AGE_YEARS) & ~s['sleep']
    limit = np.where(s['is_in_camp'], ref.CHILD_IN_CAMP_LIMIT, ref.CHILD_OUT_CAMP_LIMIT)
    neglect = is_kid & (s['unwatched_hours'] >= limit)
    return hazard | oldage | hunger | exposed | neglect, dict(...)
```

Five death modes:
- **hazard**: random ambient death (~0.5%/year)
- **old age**: age > 120
- **hunger**: hunger > 120k
- **exposed**: sleeping outside camp, scaled seasonally (5× more dangerous in winter)
- **neglect**: unwatched kid past their limit

Returns the mask + a count breakdown for daily reporting.

### `compact_dead` (`run_vec.py:284-303`)

```python
def compact_dead(agents, roots, state, weights, Q, graphs, n_nodes,
                 mating_matrix, dead_mask):
    if not dead_mask.any():
        return weights, Q, n_nodes, mating_matrix
    keep = ~dead_mask
    keep_idx = np.flatnonzero(keep)
    agents[:] = [agents[i] for i in keep_idx]
    roots[:] = [roots[i] for i in keep_idx]
    for k in state:
        state[k] = state[k][keep].copy()
    for k in graphs:
        graphs[k] = graphs[k][keep].copy()
    mating_matrix = mating_matrix[keep][:, keep].copy()
    weights = weights[keep].copy()
    Q = Q[keep].copy()
    n_nodes = n_nodes[keep].copy()
    return weights, Q, n_nodes, mating_matrix
```

Drop dead rows from every parallel array. Note that `mating_matrix` filters rows AND columns to maintain its symmetric N×N shape.

---

## 13. Family crests (kinship system)

Each agent carries a **256-d unit vector** ("crest") in addition to their genotype. Crests serve two purposes: (1) make kinship recognizable to the social-ranking function via the `family_similarity` feature, and (2) gate incest in `mating_phase`.

### Initialization & inheritance

```python
def random_unit_crest():
    v = np.array([random.gauss(0, 1) for _ in range(FAMILY_CREST_DIM)],
                 dtype=np.float32)
    return v / np.linalg.norm(v)

def slerp_crests(a, b, t=0.5):
    """Spherical linear interpolation on the unit sphere."""
    dot = clip(np.dot(a, b), -1, 1)
    if abs(dot) > 0.9995:    # nearly parallel/antipodal → linear-interp + renormalize
        c = (1-t)*a + t*b
        return c / norm(c)
    omega = acos(dot); so = sin(omega)
    return sin((1-t)*omega)/so * a + sin(t*omega)/so * b
```

- **Initial agents:** random unit vectors. Two random vectors in 256-d have cos ~ N(0, 1/√256) ≈ N(0, 0.063) — strangers reliably land near zero.
- **Children:** `slerp(mom.crest, dad.crest, 0.5)` — exact midpoint on the unit sphere. Slerp is **deterministic**, so full siblings share *identical* crests (cos = 1.0).

### Empirical kin distances

| relation | mean cos | gate at 0.6 |
|---|---:|---|
| full siblings | 1.00 | blocked |
| parent-child | 0.71 | blocked |
| aunt/uncle ↔ nephew/niece | 0.71 | blocked (siblings share crest exactly) |
| half-siblings | 0.50 | allowed |
| first cousins | 0.50 | allowed |
| grandparent-grandkid | 0.50 | allowed |
| second cousins | 0.25 | allowed |
| random unrelated | 0.00 | allowed |

The empty band around cos ≈ 0.3 cleanly separates "kin" from "stranger." Setting `FAMILY_INCEST_THRESHOLD = 0.6` blocks the closest 4 relationships and admits everything weaker.

**Quirk:** under deterministic slerp, aunts/uncles share their sibling-parent's crest exactly, so they get blocked alongside parent-child. Adding small Gaussian noise to the slerp output (and renormalizing) would distinguish them — currently disabled.

### Storage and threading

- Per-agent: `agent.crest` (numpy array, 256-d unit) — set in `__init__` via `random_unit_crest()`.
- Per-population: `crests` array (N, 256) maintained in parallel with `weights`/`Q`/`mating_matrix`/`gift_matrix`.
- Snapshot at mating: dad's crest copied to mom (`m._dad_crest_snapshot`) so birth uses dad's crest *at conception time* not death-time.
- Threaded through: `update_adaptive_rates`, `communication_phase`, `mating_phase`, `gift_phase`, `_score_against_pool`, `social_score_pairs`, `births_step`, `compact_dead`, `dump_population`.

### Family-similarity in social score

In all pairwise scoring (mate proposal, gift target, listener-talker selection):

```python
family_col = einsum('ad,kd->ak', crests[obs_idx], crests[peer_idx])
```

For self-scoring (in `update_adaptive_rates`), the family-similarity feature is **zeroed** — otherwise self-cosine = 1.0 would make every agent rank themselves as elite for the family axis, distorting the `delta = self_score - peer_mean` signal that drives adaptive rates.

The 11th linear weight (`family_w`) is what selection acts on. A clannish agent (`family_w > 0`) ranks kin higher in social-score; an exogamous agent (`family_w < 0`) ranks them lower. The gate prevents the most extreme matings regardless of weight, so this preference operates on top of the gate.

---

## 14. Multi-camp system

Agents are partitioned into camps by an int `camp_id`, with `camp_id = -1` reserved as the **wanderer sentinel** (unaffiliated, in the wild). Each real camp has its own larder (`camp_caches: dict[int, float]`); the wanderer pool has no larder. All within-camp interactions (mating, gifting, communication adoption, watching, deposit, withdraw) are gated on **same `camp_id`** *and* (for interactions) **both currently `is_in_camp = True`**.

The model maintains a strict invariant: **`is_in_camp ↔ (camp_id != -1)`**. To go foraging in the wild you must drop affiliation first; rejoining the world means committing to a camp.

### Three-stage migration

- **`agent_leave_camp`** — drops the agent into the wild: sets `camp_id = -1` and `is_in_camp = False` atomically. Only affiliated agents can leave. Kids in the FORAGE..WATCH age band can only leave when at least one adult is already a wanderer (tag-along).
- **`agent_make_camp`** — sets `make_camp_request`. Resolved in `make_camp_phase`: filter to wanderers only (so a stale request from an agent who fired both join and make in the same hour doesn't orphan them from the camp they just joined), allocate a fresh `camp_id` from a monotonic counter, set the agent's `is_in_camp = True`, initialize the new camp's cache to 0.
- **`agent_join_camp`** — sets `join_camp_request`. Resolved in `join_camp_phase`: filter to wanderers (`camp_id < 0`), then pick a real camp via softmax with **preferential-attachment size weighting**:
  - Candidate set capped at `JOIN_CAMP_CANDIDATE_CAP=12` via Efraimidis–Spirakis weighted sampling with weight `camp_size^JOIN_CAMP_SIZE_WEIGHT`, so big camps are *more likely to be considered* (not just preferred once shortlisted).
  - Final score per camp = `mean(social_score over K-sample of members) / JOIN_CAMP_TEMPERATURE + JOIN_CAMP_SIZE_WEIGHT * log(camp_size)`. Softmax picks the winner.
  - Result: `P(join camp c) ∝ exp(social_c / T) * size_c^SIZE_WEIGHT`. With `SIZE_WEIGHT=1.0`, this is canonical Barabási–Albert preferential attachment combined with social-affinity ranking. Temperature only affects the social term — size weight is stable across temp changes.
  - On pick: reassign `camp_id`, set `is_in_camp = True`.
  - **Fallback**: if no real camps exist yet (typical in the first few hours), the join request transparently becomes a make — each requester founds a fresh camp.

### Canonical gates (initially stochastic)

The starting behavior tree uses pure rng gates for all three migration actions — clustering vs nomadism is left for memetic mutation to evolve:

- `leave_camp_node.seq = [hunger > 1500, rng > 0.999, is_in_camp == True]` (links: `[LINK_OR, LINK_AND]`): leaves either because hungry (same threshold as `withdraw`, so larder-raid and foraging-trip are alternatives at the same pressure) or on a *rare* random whim (~0.1%/visit). The whim rate is intentionally stricter than make/join so affiliation remains a **Markov sink** — without that asymmetry, agents bounce between states fast enough that kids pile up exposure deaths (CHILD_OUT_CAMP_LIMIT = 1 hour).
- `make_camp_node.seq = [is_in_camp == False, rng > 0.99996667]` (LINK_AND): wanderers found new camps very rarely — threshold tuned so per-chain-pass (3 clones) fire probability ≈ 1/10,000. Making a new camp from scratch is a much rarer event than joining one. Mutation can drift this much lower if fissioning into smaller camps proves fitter.
- `join_camp_node.seq = [is_in_camp == False, rng > 0.99]` (LINK_AND): wanderers join existing camps frequently (~1%/visit, 100× more often than make). Chain order is `join_camp` before `make_camp`. When join fires but no real camps exist, it **falls back to make** so a wanderer's "I want to settle" intent is never wasted.

### Phase resolution order

Each hour after `step_all` sets the migration request flags:

1. **`join_camp_phase` runs first**, filtered to wanderers (`camp_id < 0`). It softmax-picks an existing camp by social score, or falls back to allocating a fresh camp if none exist. Both outcomes set `is_in_camp = True`.
2. **`make_camp_phase` runs second**, also filtered to wanderers. Any agent still wandering with a make-request founds a fresh camp.

Filtering both phases to wanderers prevents the same agent from being processed twice when they fire both requests in one hour (which the chain order makes possible).

### Camp lifecycle

- All founders start as wanderers (`camp_id = -1`, `is_in_camp = False`, `camp_caches = {}`, `next_camp_id = 0`). Camps form entirely via in-sim `make_camp` actions, seeded by the join→make fallback on the first hours of the run. The birth-window safety net keeps the bootstrap phase from spawning wild newborns (any near-term pregnant wanderer is force-settled before delivery).
- New camps start with cache = 0. They must accumulate deposits or absorb new members from `join_camp` to grow.
- Newborns inherit `mom.camp_id`, including the `-1` sentinel — kids born to a wandering mom are wanderers themselves (and will suffer watching-phase neglect, which is realistic selection pressure against giving birth in the wild).
- `cleanup_empty_camps` runs after every hour's deaths: any real camp with zero members is destroyed; its cache is discarded. The wanderer sentinel is never inserted into `camp_caches`.
- `next_camp_id` is monotonic — never reused, so a brand-new camp can't accidentally inherit a defunct camp's history.

### Action handlers that respect the invariant

`agent_sleep` and `agent_go_to_camp` both used to set `is_in_camp = True` unconditionally; both are now gated on `camp_id >= 0` so wanderers can sleep/walk in the wild without spuriously becoming affiliated.

### Per-camp deposit/withdraw (in `_apply_actions`)

Deposits bin by `s['camp_id']` and credit each agent's amount to `camp_caches[their_camp]`. Withdraws are pro-rated per camp — each camp's withdrawers split that camp's larder independently. With insufficient larder, the pro-rata ratio scales everyone down.

### Within-camp filter implementation

- **`mating_phase`** pre-filters the proposer pool by `same camp_id AND target is_in_camp` *before* the kin filter. Mating refused if either fails.
- **`gift_phase`** restricts the recipient pool to in-camp same-camp agents.
- **`communication_phase`** masks the talker softmax to set non-same-camp talker scores to −∞ before normalization. Adopters with no same-camp talker in their K-sample are dropped.
- **`watching_phase`** loops over `np.unique(camp_id)` and runs the in-camp / out-camp watcher-feeds-kids pool independently per camp.

### Constants

```python
CAMP_INIT_CACHE = 0.0              # founder camp starts empty; deposits build it
JOIN_CAMP_TEMPERATURE = 1.0        # softmax temp for join_camp
```

### Dump format additions

- Each agent's state dict gets `'camp_id': int`.
- Top-level `payload['camps'] = [{'id': int, 'cache': float}, ...]`.

### Rationale

Single-camp runs hit founder-effect → kinship-saturation extinction (incest gate refused 66% of mate pairs by yr 105 in one observed run). Multi-camp sub-structures the breeding pool; differential drift across camps preserves lineage diversity; `join_camp` provides a migration channel so isolated camps can mix when one becomes preferable.

---

## 15. The dump format

```python
def dump_population(path, day, state, weights, Q, roots, mating_matrix=None,
                     gift_matrix=None, crests=None, agents=None):
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
    with open(path, 'w') as f:
        json.dump(payload, f)
```

A weekly (configurable) JSON dump of the entire population. Per agent: state dict + serialized graph. Plus a sparse upper-triangle of the mating_matrix (`mating_pairs`).

The `analyze.py` script reads these dumps and produces aggregate cohort/culture/graph statistics — see top of `analyze.py` for usage.

---

## 16. Adaptive rates math

```python
_LOG_HIGH_MUT = math.log(ADAPTIVE_MUT_HIGH)                                        (main.py:933)
_LOG_LOW_MUT = math.log(ADAPTIVE_MUT_LOW)                                          (main.py:934)

def adaptive_mut_rate(self_score, peer_mean_score):                                (main.py:936-942)
    delta = self_score - peer_mean_score
    if delta > 50: delta = 50
    elif delta < -50: delta = -50
    s = 1.0 / (1.0 + math.exp(-delta))
    log_rate = (1.0 - s) * _LOG_HIGH_MUT + s * _LOG_LOW_MUT
    return math.exp(log_rate)
```

Sigmoid on `delta = self_score - peer_mean`. Mut rate log-interpolates between `LOG_HIGH_MUT` (when delta is very negative — agent feels low-status) and `LOG_LOW_MUT` (when delta is very positive — agent feels elite). Adopt rate uses the same shape with `ADAPTIVE_ADOPT_HIGH/LOW`.

The clip at ±50 prevents overflow but also caps the dynamic range. With the NDF_DECAY fix, normal `delta` values stay in the responsive [−5, +5] range and the sigmoid actually does work.

**Peer pool is per-camp.** `update_adaptive_rates` samples K peers from the agent's *own camp* (excluding self). The "relevant social context" for the self-rank delta is the people you actually interact with daily, not a global slice. This matters for multi-camp diversity: a splinter-camp member calibrates their cultural-conservation knobs against their splinter-mates, so the camp can drift culturally without the founder being pinned to camp-0's averages. Singletons (camps with no other members) fall back to sampling globally, since there's no within-camp peer to compare against.

---

## 17. Visualization (`viz.py`)

`python viz.py [week_NN]` runs all plots against one dump (defaults to the latest) plus a population-level trajectory across all dumps. Outputs land in `viz/<week>/`.

Per-week plots:
- **`family_clusters.png`** — pairwise crest-similarity matrix with single-linkage clan groups annotated.
- **`cohort_cohesion.png`** — cosine similarity between cohort-mean weight vectors. Diagonal-strong = cohorts internally agree; off-diagonals show generational drift.
- **`population_pyramid.png`** — age × sex demographics with overlaid hunger/pregnancy indicators.
- **`resource_distributions.png`** — cache + hunger histograms by life-stage.
- **`weight_heatmap.png`** — cohort × ranking-feature mean weight grid. Cells show `mean±std` so intra-cohort cultural variance is visible (tight ±0.05 = consensus, loose ±0.4 = split opinion).
- **`action_by_sex.png`** — sex × cohort × action true_action frequencies + cross-sex gift/mating matrices.
- **`camp_density.png`** — camp-size histogram + larder distribution.
- **`time_of_day_actions.png`** — population-aggregated daily rhythm. Replays each sampled agent's chain through 24 hours against their dumped state (state held frozen — captures time-gated patterns accurately but not state-evolution). Stacked-area by semantic category (sleep/forage/eat/migrate/social/economic) plus per-action heatmap. Reveals emergent diurnal patterns: synchronized "meeting hours" (chain-position-as-circadian-clock), morning social burst before differential foraging-departure, etc.
- **`time_of_day_by_camp.png`** — same as above but as small-multiples grid, one panel per top-N camps. Shows inter-camp cultural divergence in daily schedules.
- **`camp_encounters.png`** — two histograms: per-agent passive camp-mate count vs distinct active social-tie partners (gift + mating). Captures realized social-pool size — Dunbar-style.
- **`ties_by_sex_age.png`** — heatmap grid (rows: M/F, cols: age cohort) showing mean distinct partner count per cell. Three panels: mating partners, gift partners, union. Reveals life-stage shape of the social network (grandmother-effect signal: F partner count climbs into old age while M's plateaus or declines).

Across-dumps plots:
- **`viz/population_trajectory.png`** — population + mean hunger over time.
- **`viz/mating_sex_composition.png`** — M-F / M-M / F-F mating-pair mass per agent over time, normalized by pop. Top panel: absolute per-capita rates; bottom: stacked composition fractions. Useful for tracking shifts in homosocial vs reproductive bonding under selection pressure.

The time-of-day replay (`_replay_agent_day`) uses a Python re-implementation of `_eval_seqs` so it works directly from the dump JSON without needing main_vec's numpy machinery. State is held frozen during the replay — accurate for time-gated patterns, approximate for state-evolving ones.

---

## 18. Lineage tracking & pop-gen analyses (`lineage_analysis.py`)

Every `node` carries a `lineage: int` field — a heritable variant ID treated as the "allele" at that position in the decision graph. Stamping rules:

- **Canonical seed nodes** each get a unique founding lineage (IDs 1..N_canonical) at module-import time.
- **Content-changing mutations** (chunk-level, seq-level, node-level, or whole-graph dual via `mutate()`) → fresh ID via `_new_lineage()` on the affected node(s). Pure weight mutations (`mutate_weights`) do *not* bump node lineage — they're separate from the graph-structure substrate.
- **Cloning** (`clone_graph`, the duplicate branch of `mutate_node`, birth via `clone_graph(_dad_graph_snapshot)`) preserves lineage unchanged.
- **Memetic transmission** (`assimilate_node` and the subgraph adoption path in `main_vec.py:communication_phase`) → target inherits source's lineage. This is what makes the system *memetic* — the same lineage can spread horizontally, not just vertically.

`serialize_graph` emits `lineage` per node so `dumps/week_*.json` carry the data. `lineage_analysis.py` reads the dump series and produces:

- **`viz/sfs_over_time.png`** — allele-frequency-spectrum heatmap. X = sim day; Y = log-spaced carrier-count bins (how many agents currently hold a given lineage); color = log10(# distinct lineages in that bin). Bottom row bright + dim top = healthy diversity. Bright diagonal streak migrating up-right = a selective sweep. Sudden top-row dimming = founder lineages going extinct.
- **`viz/sweep_trajectories.png`** — top-20 candidate sweeps (lineages that started <5% carrier frequency and rose past 50%), each plotted as a frequency-vs-time line. Companion summary table prints estimated per-day selection coefficient *s* from the rising-portion slope (logistic fit between the 10% and 90% crossings).

This is the foundation for further pop-gen ports — phylogenies, Fst between camps, linkage disequilibrium, heritability — all unlock from this same `lineage` field if/when needed.

---

## Things that are NOT obvious

1. **Two graph representations co-exist**: object graphs (in `roots[i]`) for mutation, cloning, and serialization; flat numpy arrays (`graphs` dict) for the hot loop. They're synchronized via `reflatten_row` after every change.

2. **`cur` is preserved through re-flattens** by remapping via python-node identity. Subgraph adoption + GC change the DFS order, so the lookup is non-trivial — `reindex_freshness` does the same trick for the per-node freshness counters.

3. **Three pairwise matrices** (mating, gift, family-crest) — each contributes a column to the social-score feature vector. Mating and gift both decay each hour (30-day half-life); family-similarity is computed on the fly from per-agent crests and never decays.

4. **The canonical idle redirects to `tired_make_camp_node` (chain head)** so persistent `cur` doesn't trap agents and so the head gate (which force-settles tired/nighttime wanderers) is revisited each cycle. Without this the population would either collapse (cur traps) or have exposure deaths (tired wanderers sleeping in the wild).

5. **Same-sex mating is allowed** (recorded in mating_events even during pregnancy — the latest revision lets pregnant females also pair-bond) and updates the mating_matrix, but only opposite-sex pairs whose mom passes the fertility check (`is_pregnant == False AND is_menopausal == False`) and clears `PREGNANCY_BIRTH_P` produce pregnancies. The mating_matrix preserves all bonds regardless of sex — so social and reproductive bonding are decoupled.

6. **Out-camp = unaffiliated** (`camp_id = -1`). Forage requires `is_in_camp = False` (you can't hunt while at camp). So leaving to forage *is* becoming a wanderer. To rejoin camp life you must explicitly `make_camp` or `join_camp`. The `tired_make_camp_node` chain head force-settles wanderers at nightfall to prevent exposure-death cascades. Pregnant females in the last week of pregnancy and the first week post-partum are also forced to settle (`birth_window_force_settle_phase`) to prevent newborn deaths in the wild.

7. **Cultural-memory decay** (§9b): per-node freshness counters tick down each hour; visited nodes refresh. Daily GC reaps dead nodes. This bounds graph bloat from subgraph adoption (K=10 per accepted adoption) without losing nodes that are actually in use. The trade-off is that a chain trajectory routing around a node for `FRESHNESS_MAX` consecutive hours kills it — including, in principle, basal-survival nodes. Selection on mutation rate is the safety net.

6. **`step_all` skips forced-sleep agents** by gating every effect with `walking = forced_sleep_hours == 0`. They still pay metabolism — but in `forced_sleep_step` rather than per-visit.

7. **Top-K mating, not random sampling** — proposers rank ALL candidates by their own social score, take top-K, then softmax-pick. This concentrates mating opportunity on popular agents.

8. **Family crests are deterministic on inheritance.** `slerp(mom, dad, 0.5)` always produces the same midpoint, so full siblings have crest cosine = 1.0 exactly. This makes the gate trivially block sibling matings and is the desired behavior.

9. **Aunts/uncles get blocked by the gate** because under deterministic slerp they share their sibling-parent's crest exactly. Aunt-nephew cos ≈ 0.71, identical to parent-child. To distinguish them you'd need to add small Gaussian noise to slerp.

10. **Self-pairwise features are zeroed** in `social_score_self` (used by `update_adaptive_rates`). Otherwise `family_similarity = 1.0` for every agent's self-comparison, distorting the self-rank delta that drives mut/adopt rates.

11. **Weight mutation has 3 flavors** (perturb/flip/reset, 70/15/15). Flip and reset are *discrete* — they enable cultural drift far faster than pure Gaussian walks. Recent runs show cultural reorganization in 8 sim years that took 30+ years under the old Gaussian-only system.

12. **Graph dual is involutive.** Two consecutive duals on the same agent revert the graph entirely. Over long timescales an agent's behavior depends on the *parity* of cumulative dual events.

13. **Watching feeds kids from the watcher's personal cache, not the camp larder.** Watchers donate but don't eat their own donation. Selfless-watcher agents who spend all their time watching can starve themselves while keeping kids fed.

14. **Forage activities have separate age thresholds** (gather=4, fish=8, hunt=13). The canonical genome encodes these per-action, and mutations can shift each independently. Children take up activities in a developmental sequence over years.

15. **Hunt is "fill" semantics, not additive.** `cache <- max(cache, HUNT_YIELD * f)` — a successful hunt raises cache toward `HUNT_YIELD * seasonal_factor`, but never lowers it. Fish and gather are normal additive yields.

16. **Population size dictates cultural equilibrium.** Same mechanics produce three observed attractors: small (~70) → clannish + pro-deposit; medium (~170) → undecided; large (~380) → exogamous + anti-deposit. The selection pressure on family-weight depends on kin density, which depends on pop size.
