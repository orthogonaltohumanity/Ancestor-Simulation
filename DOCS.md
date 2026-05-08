# anthrosim documentation

A walkthrough of the simulation, code section by code section, with explanations next to the code.

---

## Overview

`anthrosim` is a population-level agent simulation. Every agent runs a small interpreted program (a "decision graph") that decides what action they take each hour. Agents have:

- **biological state** — hunger, cache, tired, age, in_camp, sleep, pregnancy
- **a decision graph** — nodes referencing actions like `agent_eat`, `agent_forage`, `agent_propose_mate`, etc.
- **social weights** — a linear vector + quadratic matrix used to rank other agents

The hour-by-hour walk evaluates each agent's graph and produces actions. Mutation perturbs graphs; sexual reproduction averages parental weights; **memetic adoption** lets agents copy nodes from other agents into their own graphs. Selection prunes dysfunctional individuals. The whole system is designed to let cultural and behavioral patterns *emerge* from the basic mechanics.

There are three Python files:

| file | role |
|---|---|
| `main.py` | reference Python implementation + canonical decision graph + tunable constants |
| `main_vec.py` | vectorized hot-loop using NumPy arrays for the per-hour walk |
| `run_vec.py` | the actual driver that uses `main_vec` for hot work and `main` for object-graph operations |

Run with `.venv/bin/python run_vec.py`. Tunables live at the top of `main.py`.

---

## 1. Top-level constants (`main.py:7-110`)

### Walk geometry

```python
MAX_ACTION = 10   # exactly N nodes visited per hour-walk          (main.py:7)
MAX_SEQ_LEN = 10  # max chunk sequence length                      (main.py:8)
MAX_NODES = 10    # max nodes per agent's decision graph (in main.py — main_vec uses its own 300)  (main.py:9)
```

`MAX_ACTION` is the **per-hour visit budget**. Each agent walks from their current `cur` (program counter) up to `MAX_ACTION` nodes per hour. With persistent `cur`, the walk continues across hours. A 10-step budget through a 33-node canonical graph means agents complete a full canonical traversal every ~3.3 hours.

### Mutation rates

```python
MUTATION_RATE_PER_HOUR = 3.0 / 24.0            # default; overridden by adaptive  (main.py:11)
ADAPTIVE_MUT_HIGH = 1.0 / (1 *  24.0)    # max mut rate (low-self-rank agents)    (main.py:12)
ADAPTIVE_MUT_LOW  = 1.0 / (3 * 365.0 * 24.0)    # min mut rate (high-self-rank)   (main.py:13)
```

Per-hour probability that an agent's graph mutates. The actual rate is **adaptive** — agents who self-rank below their peer-mean (using their own social score function) hit `ADAPTIVE_MUT_HIGH` (more exploration), confident agents hit `ADAPTIVE_MUT_LOW` (genome conservation).

### Adoption rates

```python
LISTEN_ADOPT_P = 0.30                          # default               (main.py:14)
ADAPTIVE_ADOPT_HIGH = 1.0                     # low-self-rank cap      (main.py:15)
ADAPTIVE_ADOPT_LOW  = 0.001                   # high-self-rank floor   (main.py:16)
```

Per-listener-per-hour probability of copying a node from someone else into your own graph. Same self-rank-driven adaptive mechanism as mutation.

### Social ranking features

```python
SOCIAL_FEATURES = ('net_debt_flow', 'hunger', 'tired', 'is_female', 'cache',
                   'age', 'is_pregnant', 'is_menopausal', 'matings_with_target')   # (main.py:21-22)
N_FEATURES = len(SOCIAL_FEATURES)                                                  # = 9
```

Each agent has a **9-dim weight vector** + **9×9 Q matrix** for social ranking. The 9th feature, `matings_with_target`, is the *pairwise* mating count between observer and target — observer-dependent and tracked in a per-pair matrix. The other 8 features are agent-properties.

```python
MATING_DECAY = 0.5 ** (1.0 / (7 * 24))   # 7-day half-life            (main.py:27)
MATING_NORM  = 1.0 / 5.0                 # feature normalizer         (main.py:28)
```

The mating-history matrix decays each hour like `net_debt_flow` — the count represents *recent* mating frequency, not lifetime totals.

### Energy & metabolism

```python
EAT_RATE = 500                                # kcal moved per eat action       (main.py:37)
NODE_VISIT_HUNGER = 3                         # per-visit cost                  (main.py:44)
KID_METAB_FRAC = 300.0 / 2400.0               # newborn metabolism fraction     (main.py:45)
```

Per-visit hunger cost is `NODE_VISIT_HUNGER * MAX_ACTION * metabolic_scale(age)`. For a walking adult, `3 × 10 × 1.0 = 30 kcal/hour = 720/day`. Kids ramp from 12.5% at birth to 100% at `METABOLIC_ADULT_AGE = 15`.

```python
def metabolic_scale(age):                                                         # (main.py:47-51)
    if age >= METABOLIC_ADULT_AGE:
        return 1.0
    return KID_METAB_FRAC + (age / METABOLIC_ADULT_AGE) * (1.0 - KID_METAB_FRAC)
```

Linear interpolation between newborn (12.5%) and adult (100%) metabolism.

### Cache / debt decay

```python
CACHE_LIMIT = 10000                           # max personal kcal store         (main.py:52)
CACHE_DECAY = 0.5 ** (1.0 / 36.0)             # personal cache half-life 1.5d   (main.py:53)
NDF_DECAY = 0.5 ** (1.0 / (7 * 24))           # net_debt_flow half-life 7d      (main.py:54)
DEPOSIT_AMOUNT = 1000                          # kcal moved per deposit/withdraw (main.py:55-56)
WITHDRAW_AMOUNT = 1000
CAMP = {'cache': 0.0}                         # shared in-camp larder           (main.py:57)
```

Both personal cache and `net_debt_flow` (running tally of deposit-vs-withdraw) **decay**. This was a critical fix — without decay, ndf grew unbounded and broke the social-score sigmoid that gates adoption rates.

### Seasonal forage scaling

```python
HUNT_WINTER_FACTOR = 1.0                      # hunting unaffected             (main.py:62)
FISH_WINTER_FACTOR = 0.6                      # fishing × 0.6 in winter        (main.py:63)
GATHER_WINTER_FACTOR = 0.2                    # gathering × 0.2 in winter      (main.py:64)
```

Yields scale with season: `factor = WINTER + (1 - WINTER) * season`, where `season ∈ [0, 1]` is `sin²(π·day_of_year/365)`. Winter is harsh for gatherers, mild for fishermen, neutral for hunters.

### Selection pressure

```python
N_AGENTS = 500                                                                     (main.py:67)
K_SAMPLE = 200    # listener / proposer samples K candidates                       (main.py:68)
BURN_IN_ASEX_CYCLES = 200      # cycles of (mutate-once + 72h solo-walk test)      (main.py:69)
BURN_IN_TEST_HOURS = 72        # solo simulation hours per test                    (main.py:70)
BURN_IN_HUNGER_LIMIT = 1000    # post-test cutoff — over this, the agent is culled (main.py:71)
BURN_IN_SEXUAL = 2000          # ranking-based sexual reproduction events          (main.py:72)
```

Two-phase founder evolution before the actual sim starts:
- **Phase 1 (asexual, per-agent test)**: 200 cycles. Each cycle iterates every agent: apply ONE mutation, run a 72-hour solo-walk simulation on their (just-mutated) graph, and if their final hunger exceeds `BURN_IN_HUNGER_LIMIT`, replace them with a clone of a randomly-chosen other agent. This selects for individual graph-level survivability — agents whose mutations broke their food-acquisition path get culled before they can pass on broken behavior.
- **Phase 2 (sexual)**: 2000 ranking-based sexual reproduction events. Random `i` picks opposite-sex `j` via softmax of `i`'s social ranking. Both are killed, replaced by 2 averaged-genome offspring.

Calibration: a fully-functional canonical agent finishes the 72-hour test at ~300 hunger; a maximally-broken zero-action agent reaches ~2200. Default `BURN_IN_HUNGER_LIMIT = 1000` therefore catches broken graphs (those that don't manage to net-eat at all over 72 hours) while passing functional ones.

### Death thresholds

```python
HUNGER_DEATH = 60 * 2000                        # 120,000 kcal accumulated       (main.py:73)
TIRED_DEATH = 1.0                               # triggers forced sleep, not death  (main.py:74)
SLEEP_EXPOSED_DEATH_P = 1.0/100000.0            # base risk while exposed         (main.py:75)
SLEEP_EXPOSED_WINTER_FACTOR = 5.0               # 5× risk in winter               (main.py:76)
```

Hunger over 120k = death. Tired over 1.0 = forced 12h pass-out. Sleeping outside camp = stochastic death, scaled seasonally.

### Demography

```python
FORAGE_AGE_YEARS = 6     # can hunt/fish/gather                                   (main.py:82)
WATCH_AGE_YEARS = 13     # no longer needs a watcher; can leave camp solo         (main.py:83)
METABOLIC_ADULT_AGE = 15 # full adult metabolism                                  (main.py:84)
MATE_AGE_YEARS = 16      # can propose mating                                     (main.py:85)
PREGNANCY_HOURS = 9 * 30 * 24                  # 9 months mean                    (main.py:86)
PREGNANCY_HOURS_RANGE = 20 * 24                # ± stochastic spread              (main.py:87)
PREGNANCY_FORAGE_FRACTION = 0.33               # forage allowed first 1/3        (main.py:88)
PREGNANCY_BIRTH_P = 0.10                       # P(pregnancy | successful mating)  (main.py:103)
```

Four life stages: dependent (0-6) → forager (6-13) → independent (13-16) → adult/mate (16+). Pregnancy lasts 270 days ± 20, and pregnant women can still forage during the first ~3 months.

---

## 2. Core data structures (`main.py:131-159`)

### `class chunk`

```python
class chunk:                                                                      (main.py:131-135)
    def __init__(self):
        self.var = None      # which state variable to read
        self.value = None    # comparison value
        self.op = None       # '<', '>', '==' for ordered vars; None for bool
```

A single boolean predicate over agent state. E.g. `hunger > 250` is `var='hunger', op='>', value=250`. For boolean variables (`is_female`, `is_in_camp`, `sleep`) `op` is None and `value` is True/False/None (the None case means "just bool(state[var])").

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

## 3. The 16 boolean operators (`main.py:148-181`)

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

## 4. Evaluation engine (`main.py:182-231`)

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

### `step(ag, root, hour, budget)`

```python
def step(ag, root, hour, budget=MAX_ACTION):                                      (main.py:209-225)
    n = root
    visited = 0
    visit_cost = NODE_VISIT_HUNGER * metabolic_scale(ag.age)
    while n is not None and visited < budget:
        visited += 1
        ag.hunger += visit_cost
        cond = eval_seq(n.seq, ag, hour)
        action = n.true_action if cond else n.false_action
        if action is not None:
            action(ag)
        if action is not agent_sleep:
            ag.sleep = False
            ag.consecutive_sleep = 0
        n = n.true_node if cond else n.false_node
```

The reference (object-graph) walk. Each visit:
1. accrue `visit_cost` hunger
2. evaluate the seq
3. fire the appropriate action
4. reset sleep streak if action wasn't sleep
5. advance to true_node or false_node

Bounded by `budget = MAX_ACTION`. Note the vectorized version in `main_vec.py:step_all` does the same logic over all N agents at once.

---

## 5. Action functions (`main.py:233-373`)

There are 16 actions (15 agent operations + `None` no-op). The most important ones:

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

Converts cache to negative hunger at `EAT_RATE = 500` kcal per visit.

### Forage actions (`agent_hunt`, `agent_fish`, `agent_gather`)

```python
def agent_fish(ag):                                                               (main.py:265-273)
    in_camp = ag.is_in_camp
    eligible = (not in_camp) and (ag.age >= FORAGE_AGE_YEARS) and can_forage_pregnant(ag)
    if eligible and random.random() < 0.05:
        f = FISH_WINTER_FACTOR + (1.0 - FISH_WINTER_FACTOR) * ag.season
        ag.cache = min(CACHE_LIMIT, ag.cache + 6000 * f)
```

All three require:
- not in camp
- age ≥ 6 (forage age)
- not late-pregnant (early pregnancy still allows foraging — see `can_forage_pregnant`)

Hunt: 1% success → fills cache (high-variance jackpot). Fish: 5% success → +6000 × seasonal factor. Gather: 100% success → +400 × seasonal factor (most affected by winter).

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

Move kcal to/from the shared `CAMP` larder. **`net_debt_flow` is the bookkeeping variable** — positive = generous depositor, negative = chronic withdrawer. It feeds the `social_score` 1st feature, so depositors and withdrawers look "salient" to peers (and the quadratic Q term punishes large magnitudes in either direction).

### `agent_watch_children`

```python
def agent_watch_children(ag):                                                     (main.py:367-371)
    ag.sleep = False
    ag.consecutive_sleep = 0
    ag._is_watching = True
```

Sets the watching flag. The `watching_phase` then categorizes watchers as in-camp or out-camp based on `is_in_camp`, feeds them and the children, and increments `unwatched_hours` for unwatched kids. After `CHILD_IN_CAMP_LIMIT=3` hours unwatched in camp (or `CHILD_OUT_CAMP_LIMIT=1` outside), the kid dies of neglect.

---

## 6. The canonical decision graph (`main.py:411-595`)

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

### Forage gate

```python
def _forage_seq():                                                                 (main.py:497-505)
    return make_seq(
        [ord_chunk('time', '>', 6),
         ord_chunk('time', '<', 19),
         ord_chunk('age', '>', FORAGE_AGE_YEARS - 0.0001),
         ord_chunk('season', '>', 0.0)],
        [LINK_AND, LINK_AND, LINK_AND],
    )
```

"Daylight (6 < time < 19) AND old enough AND season > 0 (anytime except winter solstice exactly)." Used by hunt, fish, gather. Mutators can drift the season threshold to make foraging seasonal in any direction.

### The full chain

```
sleep[0..2]  →  watch[0..2]  →  leave_camp[0..2]  →  eat[0..2]
                                                       ↓ (after eat clone chain)
                                        hunt → fish → gather  
                                                       ↓ (gather false=go_to_camp; true=eat-loop)
                              talk[0..2] → listen[0..2] → mate[0..2]
                                  → deposit[0..2] → withdraw[0..2]
                                                       ↓
                                              idle[0..2] → sleep_first  (loop)
```

Each non-forage action has 3 clones in chain. After the productive walk the agent reaches **idle**, whose true/false edges redirect to `sleep_first` so the program counter resets each cycle. With persistent `cur`, this means the agent walks all 33 nodes over ~3.3 hours.

### Idle redirect

```python
for i, c in enumerate(idle_clones):                                                (main.py:594-597)
    nxt = idle_clones[i + 1] if i + 1 < N_CLONES else sleep_first
    c.true_node = nxt
    c.false_node = nxt
```

Critical fix: idle clones used to self-loop within their group, which became a death trap once `cur` became persistent. They now chain forward to root.

---

## 7. Mutation (`main.py:651-794`)

Five mutation types, with rates `10% weight / 80% chunk / 9% seq / 1% node`:

```python
def mutate(root, ag):                                                              (main.py:781-794)
    if random.random() < P_MUT_WEIGHT:           # 10%
        mutate_weights(ag);  return
    nodes = all_nodes(root)
    r = random.random()
    if r < 80.0 / 90.0:                          # 80% chunk
        chunks = [c for n in nodes for c in (n.seq.chunks if n.seq else [])]
        if chunks: mutate_chunk(random.choice(chunks))
    elif r < 89.0 / 90.0:                        # 9% seq
        seqs = [n.seq for n in nodes if n.seq is not None]
        if seqs: mutate_seq(random.choice(seqs))
    else:                                        # 1% node
        mutate_node(random.choice(nodes))
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

### Weight mutation

```python
def mutate_weights(ag):                                                            (main.py:773-779)
    idx = random.randrange(N_FEATURES + N_FEATURES * N_FEATURES)
    if idx < N_FEATURES:
        ag.weights[idx] += random.gauss(0.0, WEIGHT_MUT_STD)
    else:
        idx -= N_FEATURES
        i, j = idx // N_FEATURES, idx % N_FEATURES
        ag.Q[i][j] += random.gauss(0.0, WEIGHT_MUT_STD)
```

Pick uniformly from the 9 weights or 81 Q entries, perturb by `N(0, 0.1)`. This is genetic drift on social preferences.

---

## 8. Social ranking (`main.py:875-911`)

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
    MATING_NORM,      # matings_with_target — pairwise, observer-dep
)
```

Each feature has its own normalization so the quadratic Q term doesn't explode. The 9th feature is **observer-target pairwise** — read at score time from a per-pair matrix, not from agent state.

```python
def social_score(observer, target):                                                (main.py:888-911)
    matings = 0.0
    if observer is not target and hasattr(observer, '_mating_history'):
        matings = observer._mating_history.get(id(target), 0.0)
    f = (target.net_debt_flow * FEATURE_NORM[0],
         target.hunger * FEATURE_NORM[1],
         target.tired * FEATURE_NORM[2],
         (1.0 if target.is_female else -1.0) * FEATURE_NORM[3],
         target.cache * FEATURE_NORM[4],
         target.age * FEATURE_NORM[5],
         (1.0 if target.is_pregnant else -1.0) * FEATURE_NORM[6],
         (1.0 if target._is_menopausal else -1.0) * FEATURE_NORM[7],
         matings * FEATURE_NORM[8])
    c = observer.weights
    Q = observer.Q
    linear = sum(c[i] * f[i] for i in range(N_FEATURES))
    quadratic = sum(Q[i][j] * f[i] * f[j]
                    for i in range(N_FEATURES) for j in range(N_FEATURES))
    return linear + quadratic
```

`score = c·x + xᵀQx` where `x` is the normalized feature vector. The quadratic term lets agents express **interactions** like "pregnant × generous" or "female × matings_with_target". This is what makes the system express *preference patterns* not just additive feelings.

---

## 9. Vectorized hot loop (`main_vec.py`)

The reference Python walk is too slow for N=1500 over 2000 sim-years. `main_vec.py` re-encodes the per-hour walk as numpy operations.

### Flat graph layout

```python
MAX_NODES = 300                   # hard array cap                                 (main_vec.py:15)
ADOPTION_CAP = 100                # adoption stops at this                         (main_vec.py:16)
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

## 10. Two-phase burn-in (`run_vec.py:73-200`)

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
3. **after every hour, check hunger**. If it ever exceeds `BURN_IN_HUNGER_LIMIT = 1000`, fail immediately → cull and replace with a clone of a random other agent

The fail-fast behavior is intentional: agents must keep hunger below the threshold *the entire test*, not just at the end. This prevents "gaming" the selection by letting hunger spike then dropping it via a single eat-action right at the end. The actual sim is hour-by-hour and an agent who lets their hunger spike will die regardless of what they do later, so the test mirrors that constraint.

Run for `BURN_IN_ASEX_CYCLES = 200` cycles. Each agent therefore receives 200 mutation-and-test rounds; bad mutations are reverted (via clone-replacement) and good ones accumulate. The threshold of 1000 hunger discriminates functional graphs (~300 peak hunger over 72h) from broken ones (which exceed 1000 within hours).

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

## 11. The main hour loop (`run_vec.py:339-410`)

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

### `update_adaptive_rates` (`main_vec.py:497-510`)

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

### `communication_phase` (`main_vec.py:602-803`)

```python
def communication_phase(s, graphs, n_nodes_arr, weights, Q, rng,
                        roots=None, mating_matrix=None, K=ref.K_SAMPLE):
    talker_mask = s['is_in_camp'] & s['talk_request']
    listener_mask = s['is_in_camp'] & s['listen_request']
    s['talk_request'][:] = False
    s['listen_request'][:] = False
    talker_idx = np.flatnonzero(talker_mask)
    listener_idx = np.flatnonzero(listener_mask)
    if not talker_idx.size or not listener_idx.size: return 0, 0, 0
    
    talker_node = (rng.random(n_talks) * n_nodes_arr[talker_idx]).astype(np.int32)
    adopt = s['adopt_rate'][listener_idx]
    rolls = rng.random(n_listens)
    adopters = listener_idx[rolls < adopt]
    ...
    # softmax-pick a talker per adopter using social score
    scores = ...   # uses weights, Q, features, mating_matrix
    pick = (r < cum).argmax(axis=1)
    src_agents = chosen_t_global[np.arange(adopters.size), pick]
    src_nodes  = talker_node[chosen_t_local[...]]
    
    # APPEND mode if room, REPLACE mode if at adoption cap
    has_room = n_nodes_arr[adopters] < ADOPTION_CAP
    ...
```

Flow:
1. **find talkers and listeners** — agents who fired `agent_talk` and `agent_listen` actions this hour while in camp
2. **each talker exposes a random node from their graph**
3. **each listener (per their adopt_rate) softmax-picks a talker** using their own social-ranking weights
4. **Adoption mode**:
   - **APPEND** (if `n_nodes < ADOPTION_CAP`): create a new python node with the source's content, splice it into one randomly-chosen edge of an existing node. Existing `A → B` becomes `A → new → B`.
   - **REPLACE** (if at cap): overwrite the content of a randomly-chosen existing node with the source's content (preserving edges).
5. **Sync flat ↔ object**: re-flatten the listener's row from the now-modified object graph; re-anchor `cur` to the same python-node identity in the new flat indexing.

The re-flatten is critical — `flatten_graph` uses DFS order from root, so flat indices change after each adoption. Re-anchoring `cur` via python-node identity preserves the agent's program counter through these renumberings.

### `mating_phase` (`main_vec.py:823-927`)

```python
def mating_phase(s, weights, Q, rng, mating_matrix=None, K=ref.K_SAMPLE):
    proposers = np.flatnonzero(s['mate_request'])
    s['mate_request'][:] = False
    if proposers.size <= 1: return 0, [], []
    F_base = features_vec(s)
    n_resolved = 0
    dad_pairs = []
    mating_events = []

    for p in proposers:
        pool = proposers[proposers != p]
        if pool.size == 0: continue
        # p's full ranking of all candidates, then take TOP K by score
        sp_all = _score_against_pool(int(p), pool, F_base, weights, Q, mating_matrix)
        k = min(K, pool.size)
        if k < pool.size:
            top_k_idx = np.argpartition(-sp_all, k - 1)[:k]
        else:
            top_k_idx = np.arange(pool.size)
        cand = pool[top_k_idx]
        sp = sp_all[top_k_idx]
        chosen = int(cand[_softmax_pick(sp, rng)])

        # chosen's accept pool — score everyone, top K, must include p
        ...
        accept_prob = float(probs[p_at[0]])
        if rng.random() >= accept_prob: continue

        # SUCCESSFUL MATING
        mating_events.append((int(p), int(chosen)))

        # Pregnancy only if hetero, fertile, eligible
        if is_f[p] != is_f[chosen]:
            mom = p if is_f[p] else chosen
            dad = chosen if is_f[p] else p
            if is_p[mom] or is_meno[mom]: continue
            if rng.random() < ref.PREGNANCY_BIRTH_P:
                s['is_pregnant'][mom] = True
                s['pregnancy_hours'][mom] = 0
                s['pregnancy_target'][mom] = int(rng.integers(...))
                dad_pairs.append((int(mom), int(dad)))

    return n_resolved, dad_pairs, mating_events
```

Notable design choices:
- **same-sex pairs allowed** — they still get recorded in `mating_events` and feed the mating_matrix, but only opposite-sex pairs can produce pregnancy
- **Top-K by ranking, not random sampling** — popular agents get all the proposals, creating winner-take-all dynamics
- **Pregnancy only at 10% per successful mating** — most matings produce no offspring, which keeps fertile females in the mating market longer

### `forced_sleep_step` (`main_vec.py:826-848`)

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

### `watching_phase` (`main_vec.py:520-572`)

```python
def watching_phase(s, camp_cache):
    fa = ref.WATCH_AGE_YEARS                                # = 13
    is_kid = s['age'] < fa
    in_camp_w  = s['is_watching'] & s['is_in_camp']
    out_camp_w = s['is_watching'] & ~s['is_in_camp']
    in_camp_k  = is_kid & s['is_in_camp']
    out_camp_k = is_kid & ~s['is_in_camp']

    if in_camp_w.any():
        s['unwatched_hours'][in_camp_k] = 0
    if out_camp_w.any():
        s['unwatched_hours'][out_camp_k] = 0

    # in-camp feed: watchers and watched both fed proportionally
    if in_camp_w.any():
        n_w = int(in_camp_w.sum()); n_c = int(in_camp_k.sum())
        need = ref.WATCH_FEED * (n_w + n_c)
        avail = min(need, camp_cache)
        ratio = (avail / need) if need > 0 else 0.0
        camp_cache -= avail
        per = np.float32(ref.WATCH_FEED * ratio)
        s['hunger'][in_camp_w] = np.maximum(0.0, s['hunger'][in_camp_w] - per)
        s['hunger'][in_camp_k] = np.maximum(0.0, s['hunger'][in_camp_k] - per)

    s['unwatched_hours'][in_camp_k & s['sleep']] = 0   # sleeping kids are safe

    if not in_camp_w.any():
        s['unwatched_hours'][in_camp_k & ~s['sleep']] += 1
    if not out_camp_w.any():
        s['unwatched_hours'][out_camp_k & ~s['sleep']] += 1

    s['is_watching'][:] = False
```

The "watch_children" action sets a flag; this phase consumes the flags. In-camp watchers feed the kids and themselves from camp larder. Unwatched kids accumulate `unwatched_hours`; if they exceed `CHILD_IN_CAMP_LIMIT=3` (or `CHILD_OUT_CAMP_LIMIT=1` outside), they die of neglect.

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

### `death_mask` (`main_vec.py:856-886`)

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

## 13. The dump format

```python
def dump_population(path, day, state, weights, Q, roots, mating_matrix=None):     (run_vec.py:23-46)
    payload = {
        'day': day,
        'n_agents': N,
        'agents': [
            {'state': serialize_agent_from_flat(i, state, weights, Q),
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

## 14. Adaptive rates math

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

---

## Things that are NOT obvious

1. **Two graph representations co-exist**: object graphs (in `roots[i]`) for mutation, cloning, and serialization; flat numpy arrays (`graphs` dict) for the hot loop. They're synchronized via `reflatten_row` after every change.

2. **`cur` is preserved through re-flattens** by remapping via python-node identity. Mutation never deletes nodes, so the lookup always succeeds.

3. **The mating_matrix is symmetric and decays each hour** — a pair "forgets" their mating history at a 7-day half-life rate. Used both for ranking (the 9th feature) and as data exposed in dumps.

4. **The canonical idle redirects to root** so persistent `cur` doesn't trap agents. This single edit was the difference between population collapse and persistence.

5. **Same-sex mating is allowed** (recorded in mating_events) but only opposite-sex pairs with a fertile mom have a `PREGNANCY_BIRTH_P=10%` chance of producing pregnancy. The mating_matrix preserves all matings regardless.

6. **`step_all` skips forced-sleep agents** by gating every effect with `walking = forced_sleep_hours == 0`. They still pay metabolism — but in `forced_sleep_step` rather than per-visit.

7. **Top-K mating, not random sampling** — proposers rank ALL opposite-sex candidates and softmax-pick from the top K. This concentrates mating opportunity on popular agents.

8. **Burn-in does selection on food-actions** — bottom 30% by `count(eat/hunt/fish/gather)` is replaced with clones of survivors. Then 2000 sexual reproduction events with ranking-based mate choice.
