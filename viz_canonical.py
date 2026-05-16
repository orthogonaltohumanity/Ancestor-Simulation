"""Render the canonical decision graph (the starting graph every agent is born with)."""
import sys
sys.path.insert(0, '/mnt/anthrosim')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import main as ref
import graph_evo as ge


def main(out='viz/canonical_graph.png'):
    graph = ref.serialize_graph(ref.canonical_root)
    snapshot = {
        'day': 0,
        'state': {
            'id': -1, 'is_female': False, 'age': 0.0,
            'hunger': 0, 'cache': 0, 'tired': 0,
            'camp_id': '—', 'mut_rate': 0, 'adopt_rate': 0,
        },
        'graph': graph,
    }
    fig, (ax_graph, ax_info) = plt.subplots(
        1, 2, figsize=(20, 16),
        gridspec_kw={'width_ratios': [4, 1]},
        facecolor='#1a1a2e'
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.03)
    ge.render_frame(ax_graph, ax_info, snapshot, agent_id='CANONICAL')
    ax_graph.set_title(f'Canonical decision graph   {len(graph["nodes"])} nodes',
                       fontsize=13, color='white', pad=8)
    fig.savefig(out, dpi=130, facecolor='#1a1a2e')
    plt.close(fig)
    print(f'wrote {out}  ({len(graph["nodes"])} nodes)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'viz/canonical_graph.png')
