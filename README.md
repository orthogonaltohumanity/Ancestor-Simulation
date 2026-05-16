# anthrosim

A population-level agent simulation where each agent runs a small interpreted "decision graph" that decides their actions hour-by-hour. Decision graphs and ranking weights are heritable and mutable; agents pair-bond, gift, communicate (memetic node-graft from peer to peer), form camps, and reproduce. Cultural and behavioral patterns are designed to *emerge* from the basic mechanics rather than being hard-coded.

See [`DOCS.md`](DOCS.md) for the full code walkthrough.

## Install

```bash
pip install -r requirements.txt
```

For GPU runs, also install a CuPy build matching your CUDA toolchain (see comments in `requirements.txt`). CPU-only runs work without it.

## Run

```bash
python3 run_vec.py                          # default run, dumps to dumps/
python3 run_vec.py --N 400                  # override population size
python3 run_vec.py --dump-every 1           # daily dumps (for animation viz)
```

Tunables live at the top of `main.py`.

## Analyze

```bash
python3 viz.py                              # static charts → viz/
python3 analyze.py                          # text reports
python3 lineage_analysis.py                 # SFS + selective-sweep plots → viz/
python3 graph_evo.py dumps <agent_id>       # decision-graph animation → .gif
python3 archetypes.py                       # archetypal value-coalitions (requires archepy)
```

## File layout

| file | role |
|---|---|
| `main.py` | constants, agent/node classes, canonical decision graph, mutation, social ranking. **Library only** — no sim loop. |
| `main_vec.py` | vectorized hot loop: per-hour walk, mating, watching, gifting, births, deaths, communication, sub-graph adoption. NumPy or CuPy. |
| `run_vec.py` | **entry point** — driver, founder init, burn-in, hour loop, dump cadence. |
| `viz.py` | matplotlib charts over dumps |
| `analyze.py` | text-mode summaries |
| `lineage_analysis.py` | pop-gen analyses on the memetic node-lineage system (SFS, sweep detection) |
| `graph_evo.py` | animated decision-graph gif for one agent across dumps |
| `viz_canonical.py` | renders the canonical founder graph |
| `archetypes.py` | multi-subject archetypal analysis of weight vectors |

If you're editing simulation *semantics*, look in `main_vec.py`. If you're editing *constants, the canonical graph, mutation/clone logic, or social scoring*, look in `main.py`. If you're editing the *driver, init, or dump cadence*, look in `run_vec.py`.

## Note on AI Usage
This project was coded with assistance from Claude Opus 4.7
