"""graph_evo.py — animate a single agent's decision graph across weekly dumps.

Usage:
    python3 graph_evo.py <dump_dir> <agent_id> [--out output.gif] [--fps 4] [--every N]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import numpy as np


# ── action color + short label ───────────────────────────────────────────────
ACTION_COLORS = {
    'agent_sleep':          '#5b8dd9',
    'agent_wake':           '#a8d0f0',
    'agent_eat':            '#f0a830',
    'agent_hunt':           '#c0392b',
    'agent_fish':           '#1a7aaa',
    'agent_gather':         '#27ae60',
    'agent_go_to_camp':     '#8e44ad',
    'agent_leave_camp':     '#e67e22',
    'agent_talk':           '#d4ac0d',
    'agent_listen':         '#f7dc6f',
    'agent_deposit':        '#117a65',
    'agent_withdraw':       '#e74c3c',
    'agent_propose_mate':   '#ff69b4',
    'agent_watch_children': '#a29bfe',
    'agent_idle':           '#bdc3c7',
    'agent_make_camp':      '#6c3483',
    'agent_join_camp':      '#9b59b6',
    'agent_gift':           '#fd79a8',
    'agent_move_n':         '#74b9ff',
    'agent_move_s':         '#0984e3',
    'agent_move_e':         '#81ecec',
    'agent_move_w':         '#00cec9',
    None:                   '#dfe6e9',
}
ACTION_SHORT = {
    'agent_sleep': 'ZZZ', 'agent_wake': 'wake', 'agent_eat': 'eat',
    'agent_hunt': 'hunt', 'agent_fish': 'fish', 'agent_gather': 'gath',
    'agent_go_to_camp': '→camp', 'agent_leave_camp': '←camp',
    'agent_talk': 'talk', 'agent_listen': 'lstn',
    'agent_deposit': 'dep', 'agent_withdraw': 'wdw',
    'agent_propose_mate': 'mate', 'agent_watch_children': 'wtch',
    'agent_idle': 'idle', 'agent_make_camp': 'mkcp',
    'agent_join_camp': 'join', 'agent_gift': 'gift',
    'agent_move_n': '↑', 'agent_move_s': '↓',
    'agent_move_e': '→', 'agent_move_w': '←',
    None: '—',
}
DEFAULT_COLOR = '#b2bec3'

# ── variable abbreviations for condition display ─────────────────────────────
VAR_SHORT = {
    'tired': 'T', 'hunger': 'H', 'cache': 'C', 'time': 't',
    'age': 'A', 'season': 'S', 'is_female': '♀', 'is_in_camp': '⛺',
    'sleep': 'Z', 'rng': 'R', 'x': 'x', 'y': 'y',
    'nearest_camp_n': 'cN', 'nearest_camp_s': 'cS',
    'nearest_camp_e': 'cE', 'nearest_camp_w': 'cW',
}
LINK_SYM = {14: '∨', 8: '∧'}


def fmt_chunks_short(seq):
    if seq is None:
        return '★'
    chunks = seq.get('chunks', [])
    links = seq.get('links', [])
    parts = []
    for i, c in enumerate(chunks):
        var = VAR_SHORT.get(c.get('var', '?'), c.get('var', '?'))
        op = c.get('op') or ''
        val = c.get('value')
        if val is None:
            parts.append(var)
        elif isinstance(val, bool):
            parts.append(f'{var}={"T" if val else "F"}')
        elif isinstance(val, float):
            parts.append(f'{var}{op}{val:.2g}')
        else:
            parts.append(f'{var}{op}{val}')
        if i < len(links):
            parts.append(LINK_SYM.get(links[i], '?'))
    return ' '.join(parts) if parts else '★'


def action_color(name):
    return ACTION_COLORS.get(name, DEFAULT_COLOR)


def action_short(name):
    return ACTION_SHORT.get(name, (name or '—').replace('agent_', '')[:4])


# ── build networkx graph ─────────────────────────────────────────────────────

def build_nx(nodes):
    G = nx.DiGraph()
    for i, n in enumerate(nodes):
        cond = fmt_chunks_short(n.get('seq'))
        ta = n.get('true_action')
        fa = n.get('false_action')
        G.add_node(i, cond=cond, ta=ta, fa=fa)
    for i, n in enumerate(nodes):
        tn = n.get('true_node')
        fn = n.get('false_node')
        if tn is not None and tn != i:
            G.add_edge(i, tn, kind='T')
        if fn is not None and fn != i and fn != tn:
            G.add_edge(i, fn, kind='F')
    return G


def hierarchical_pos(G, root=0):
    """Assign each node a depth via BFS, then within each depth, spread nodes
    evenly across [0,1] in x. Depth maps to y ∈ [0,1] (top = root).
    Returns positions normalized to fill the unit square exactly."""
    depth = {root: 0}
    order = [root]
    visited = {root}
    qi = 0
    while qi < len(order):
        u = order[qi]; qi += 1
        for _, v in G.out_edges(u):
            if v not in visited:
                visited.add(v); depth[v] = depth[u] + 1; order.append(v)
    # any orphans go at the bottom
    max_d = max(depth.values()) if depth else 0
    for n in G.nodes():
        if n not in depth:
            depth[n] = max_d + 1
    # bucket by depth
    by_depth = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)
    n_levels = max(by_depth) + 1
    pos = {}
    for d, nodes in by_depth.items():
        # sort nodes by id for stable layout
        nodes = sorted(nodes)
        m = len(nodes)
        y = 1.0 - (d / max(1, n_levels - 1)) if n_levels > 1 else 0.5
        for i, n in enumerate(nodes):
            x = (i + 0.5) / m if m > 0 else 0.5
            pos[n] = (x, y)
    return pos


# ── render one frame ─────────────────────────────────────────────────────────

def render_frame(ax_graph, ax_info, snapshot, agent_id):
    ax_graph.clear()
    ax_info.clear()
    ax_graph.set_facecolor('#1a1a2e')
    ax_info.set_facecolor('#16213e')

    nodes = snapshot['graph']['nodes']
    st = snapshot['state']
    day = snapshot['day']

    G = build_nx(nodes)
    if len(G.nodes()) == 0:
        ax_graph.text(0.5, 0.5, 'no graph', ha='center', va='center',
                      transform=ax_graph.transAxes, fontsize=14, color='#aaa')
        return

    root = snapshot['graph'].get('root', 0)
    try:
        pos = hierarchical_pos(G, root=root)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    node_colors = [action_color(G.nodes[n]['ta']) for n in G.nodes()]
    node_border  = [action_color(G.nodes[n]['fa']) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax_graph,
                           node_color=node_colors,
                           edgecolors=node_border,
                           linewidths=2,
                           node_size=400, alpha=0.95)

    # node label: condition + actions as plain text beside each node
    for n in G.nodes():
        x, y = pos[n]
        cond = G.nodes[n]['cond']
        ta   = action_short(G.nodes[n]['ta'])
        fa   = action_short(G.nodes[n]['fa'])
        ax_graph.text(x, y, f'{cond}\n✓{ta}  ✗{fa}',
                      ha='center', va='center', fontsize=6, color='white',
                      bbox=dict(boxstyle='round,pad=0.1', fc='#0f3460', alpha=0.6, lw=0))

    t_edges = [(u, v) for u, v, d in G.edges(data=True) if d['kind'] == 'T']
    f_edges = [(u, v) for u, v, d in G.edges(data=True) if d['kind'] == 'F']
    nx.draw_networkx_edges(G, pos, edgelist=t_edges, ax=ax_graph,
                           edge_color='#00b894', arrows=True, arrowsize=18,
                           width=2.5, connectionstyle='arc3,rad=0.08',
                           min_source_margin=28, min_target_margin=28)
    nx.draw_networkx_edges(G, pos, edgelist=f_edges, ax=ax_graph,
                           edge_color='#d63031', arrows=True, arrowsize=18,
                           width=2.5, connectionstyle='arc3,rad=0.08',
                           min_source_margin=28, min_target_margin=28)

    ax_graph.set_title(f'Agent {agent_id}   day {day}   {len(nodes)} nodes',
                       fontsize=11, color='white', pad=6)
    ax_graph.set_xlim(-0.04, 1.04)
    ax_graph.set_ylim(-0.06, 1.06)
    ax_graph.set_aspect('auto')
    ax_graph.axis('off')

    # ── sidebar ──────────────────────────────────────────────────────────────
    ax_info.axis('off')
    sex = '♀' if st.get('is_female') else '♂'
    age_y = st.get('age', 0)
    hunger = st.get('hunger', 0)
    cache  = st.get('cache', 0)
    tired  = st.get('tired', 0)
    camp   = st.get('camp_id', '—')
    mut    = st.get('mut_rate', 0)
    adopt  = st.get('adopt_rate', 0)

    lines = [
        ('Sex',    sex),
        ('Age',    f'{age_y:.1f} yr'),
        ('Hunger', f'{hunger:.0f}'),
        ('Cache',  f'{cache:.0f}'),
        ('Tired',  f'{tired:.2f}'),
        ('Camp',   str(camp)),
        ('Mut',    f'{mut:.4f}'),
        ('Adopt',  f'{adopt:.4f}'),
        ('Nodes',  str(len(nodes))),
    ]
    for i, (k, v) in enumerate(lines):
        y_pos = 0.93 - i * 0.085
        ax_info.text(0.08, y_pos, k, transform=ax_info.transAxes,
                     va='top', ha='left', fontsize=9, color='#a29bfe',
                     fontfamily='monospace')
        ax_info.text(0.55, y_pos, v, transform=ax_info.transAxes,
                     va='top', ha='left', fontsize=9, color='white',
                     fontfamily='monospace')

    # action legend (only actions present in this frame)
    seen = set(G.nodes[n]['ta'] for n in G.nodes()) | set(G.nodes[n]['fa'] for n in G.nodes())
    seen.discard(None)
    patches = [mpatches.Patch(color=action_color(a), label=action_short(a))
               for a in sorted(seen)]
    leg = ax_info.legend(handles=patches, loc='lower center', fontsize=7,
                         framealpha=0.3, ncol=2,
                         labelcolor='white', facecolor='#0f3460',
                         edgecolor='none', title='actions', title_fontsize=7)
    plt.setp(leg.get_title(), color='#a29bfe')


# ── load + main ──────────────────────────────────────────────────────────────

def find_longest_lived(dump_dir):
    """Scan all dumps, return the agent ID with the most appearances."""
    from collections import Counter
    files = sorted(
        glob.glob(os.path.join(dump_dir, 'week_*.json')),
        key=lambda f: int(os.path.basename(f).replace('week_', '').replace('.json', ''))
    )
    counts = Counter()
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for ag in d['agents']:
            counts[ag['state']['id']] += 1
    agent_id, n = counts.most_common(1)[0]
    print(f'Longest-lived agent: {agent_id} ({n} snapshots)')
    return agent_id


def load_snapshots(dump_dir, agent_id, every=1):
    files = sorted(
        glob.glob(os.path.join(dump_dir, 'week_*.json')),
        key=lambda f: int(os.path.basename(f).replace('week_', '').replace('.json', ''))
    )
    snapshots = []
    for i, f in enumerate(files):
        if i % every != 0:
            continue
        with open(f) as fh:
            d = json.load(fh)
        for ag in d['agents']:
            if ag['state']['id'] == agent_id:
                snapshots.append({'day': d['day'], 'state': ag['state'],
                                  'graph': ag['graph']})
                break
    return snapshots


def make_gif(dump_dir, agent_id, out_path, fps=4, every=1):
    if agent_id is None:
        agent_id = find_longest_lived(dump_dir)
    if out_path is None:
        out_path = f'graph_evo_{agent_id}.gif'
    print(f'Loading snapshots for agent {agent_id}…')
    snapshots = load_snapshots(dump_dir, agent_id, every=every)
    if not snapshots:
        print(f'Agent {agent_id} not found in {dump_dir}')
        sys.exit(1)
    print(f'{len(snapshots)} frames  (day {snapshots[0]["day"]} → {snapshots[-1]["day"]})')

    fig, (ax_graph, ax_info) = plt.subplots(
        1, 2, figsize=(20, 16),
        gridspec_kw={'width_ratios': [4, 1]},
        facecolor='#1a1a2e'
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.03)

    done = [0]
    def update(i):
        render_frame(ax_graph, ax_info, snapshots[i], agent_id)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f'  {done[0]}/{len(snapshots)}…')

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps)
    print(f'Writing {out_path}…')
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'Done → {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dump_dir')
    p.add_argument('agent_id', type=int, nargs='?', default=None,
                   help='agent ID to track (omit to auto-pick longest-lived)')
    p.add_argument('--out', default=None)
    p.add_argument('--fps', type=int, default=4)
    p.add_argument('--every', type=int, default=1)
    args = p.parse_args()
    make_gif(args.dump_dir, args.agent_id, args.out, fps=args.fps, every=args.every)
