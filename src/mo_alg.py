# Algortihm overview:
# 1. Randomly select vt_size number of test nodes vt, induce the L-hop neighbor subgraph.
# 2. Identify the 1-hop nodes, 2-hop nodes, etc.
# 3. Delete one edge at each iteration (among most-outsider nodes).
# 4. Maintain a bucket of size k for explanation subgraphs.
# 5. Iterate the algorithm for H (# of most-outsider nodes) iterations. 
# 6. Identify the pareto paths and udpate the bucket with nodes (explanation subgraphs) in the paths.
# 7. The bucket will be updated L times. 

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from src.plot import plot_hop_nodes





def get_nodes_at_hops(VT, L, data):
    hop_nodes = {}
    all_previous_hops = torch.tensor([]).long()  # Keep track of previous hops
    hop_nodes[0] = VT
    
    for hop in range(1, L + 1):
        subset, edge_index, _, _ = k_hop_subgraph(VT, hop, data.edge_index, relabel_nodes=True)
        mask = torch.isin(subset, all_previous_hops)
        current_hop_nodes = subset[~mask]
        hop_nodes[hop] = current_hop_nodes
        all_previous_hops = torch.cat([all_previous_hops, current_hop_nodes])

    subg = Data(
        x = data.x[subset],
        edge_index = edge_index,
    )
    
    return subset, subg, hop_nodes


def mo_alg(data, model, VT, L):

    subset, subg, hop_nodes = get_nodes_at_hops(VT, L, data)

    # Initialize a color_y feature for all nodes (with -1 indicating uncolored)
    num_nodes = subg.edge_index.max().item() + 1  # Assuming node indices start from 0
    color_y = torch.full((num_nodes,), 0, dtype=torch.long)

    # Assign hop levels (colors) to nodes
    for hop, nodes in hop_nodes.items():
        color_y[nodes] = hop  # Assign the hop number as the color

    # Add the color_y feature to the graph data
    subg.color_y = color_y

    plot_hop_nodes(subg, subset)

    print('Done.')