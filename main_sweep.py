import numpy as np
import random
import json
from itertools import chain, combinations
from tqdm import tqdm

# Sweep variant of main.py: fixed C = 1.0, sweeps ZETA over {0.5, 0.75, 1.0, 1.25}.
# Writes one sim_N{N}_T{T}_C{C}_Z{ZETA}.jsonl file per ZETA value.

N = 7
delta = 0.5
beta = 1.0
T = 100_000
SAVE_EVERY = 250 
SEEDS = [0, 1, 2, 3, 4]
C = 1.0
ZETA_VALUES = [round(0.5 + 0.1 * i, 2) for i in range(16)]  # 0.5..2.0 step 0.1
HALF_LIFE = N
DECAY = 0.5 ** (1.0 / HALF_LIFE)


def powerset(it):
    s = list(it)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


P = powerset(range(N))
P_idx = {g: i for i, g in enumerate(P)}
G = [[g for g in P if n in g] for n in range(N)]
group_idx = [{g: i for i, g in enumerate(G[n])} for n in range(N)]
singleton_idx = np.array([P_idx[(n,)] for n in range(N)], dtype=int)
group_indices = [np.array([P_idx[g] for g in G[n]], dtype=int) for n in range(N)]


def run(ZETA, SEED):
    random.seed(SEED)
    np.random.seed(SEED)

    k = 1.0 * np.ones(N)
    mu = [np.full(len(G[n]), 1.0 / len(G[n])) for n in range(N)]
    rho = np.zeros(len(P))
    rho[singleton_idx] = 1.0 / N

    def snapshot(t):
        return {
            't': t,
            'mu': [
                {','.join(map(str, g)): float(mu[n][i]) for i, g in enumerate(G[n])}
                for n in range(N)
            ],
            'rho': {','.join(map(str, g)): float(rho[P_idx[g]]) for g in P},
        }

    snap_path = f'sim_N{N}_T{T}_C{C}_Z{ZETA}_s{SEED}.jsonl'
    snap_file = open(snap_path, 'w')
    snap_file.write(json.dumps(snapshot(0)) + '\n')

    for t in tqdm(range(1, T + 1), desc=f'ZETA={ZETA} seed={SEED}'):
        agent = random.randrange(N)
        mu_a = mu[agent]
        idx = group_indices[agent]

        w = (1.0 - mu_a) * rho[idx]
        w_sum = w.sum()
        if w_sum > 0.0:
            j = np.random.choice(len(mu_a), p=w / w_sum)
            g = G[agent][j]
            g_idx_flat = P_idx[g]
            amount = min(C, rho[g_idx_flat])
            rho[g_idx_flat] -= amount

            if amount > 0.0:
                mu_a[j] *= delta ** (-amount)
                mu_a /= mu_a.sum()
                for m in g:
                    if m == agent:
                        continue
                    jm = group_idx[m][g]
                    mu_m = mu[m]
                    mu_m[jm] *= delta ** (amount)
                    mu_m /= mu_m.sum()

        rho *= DECAY

        if np.random.random() < k[agent]:
            j = np.random.choice(len(mu_a), p=mu_a)
            g = G[agent][j]
            rho[P_idx[g]] += ZETA

            mu_a[j] *= delta ** C
            mu_a /= mu_a.sum()
            for m in g:
                if m == agent:
                    continue
                jm = group_idx[m][g]
                mu_m = mu[m]
                mu_m[jm] *= delta ** (-C)
                mu_m /= mu_m.sum()

        if t % SAVE_EVERY == 0:
            snap_file.write(json.dumps(snapshot(t)) + '\n')
            snap_file.flush()

    snap_file.close()
    print(f'  saved {snap_path}  sum(rho)={rho.sum():.4f}  min(rho)={rho.min():.3e}')


for ZETA in ZETA_VALUES:
    for SEED in SEEDS:
        run(ZETA, SEED)
