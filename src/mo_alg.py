# Algortihm overview:
# 1. Randomly select vt_size number of test nodes vt, induce the L-hop neighbor subgraph.
# 2. Identify the 1-hop nodes, 2-hop nodes, etc.
# 3. Delete one edge at each iteration (among most-outsider nodes).
# 4. Maintain a bucket of size k for explanation subgraphs.
# 5. Iterate the algorithm for H (# of most-outsider nodes) iterations. 
# 6. Identify the pareto paths and udpate the bucket with nodes (explanation subgraphs) in the paths.
# 7. The bucket will be updated L times. 

import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data
from itertools import combinations
import torch.nn.functional as F

from src.plot import plot_hop_nodes


def get_L_hop_subg(VT, L, data):

    subset_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(VT, L, data.edge_index, relabel_nodes=True)

    subg = Data(
        x = data.x[subset_nodes],
        edge_index = sub_edge_index,
        y = data.y[subset_nodes],
        color_y = data.color_y[subset_nodes],
    )

    return subg, subset_nodes, sub_edge_index, mapping


def get_nodes_edges_at_hops(VT, L, edge_index):
    hop_nodes = {}
    hop_edges = {}
    all_previous_hops = VT
    hop_nodes[0] = VT
    hop_edges[0], _ = subgraph(VT, edge_index)
    
    for hop in range(1, L + 1):
        subset, sub_edge_index, _, _ = k_hop_subgraph(VT, hop, edge_index, relabel_nodes=False)
        mask = torch.isin(subset, all_previous_hops)
        current_hop_nodes = subset[~mask]
        
        edge_mask = torch.isin(sub_edge_index[0], current_hop_nodes) | torch.isin(sub_edge_index[1], current_hop_nodes)
        current_hop_edges = sub_edge_index[:, edge_mask]
        
        hop_nodes[hop] = current_hop_nodes
        hop_edges[hop] = current_hop_edges
        all_previous_hops = torch.cat([all_previous_hops, current_hop_nodes])
    
    return subset, hop_nodes, hop_edges


def remove_edges(subset_edges, edge_indices_to_remove):
    mask = torch.ones(subset_edges.shape[1], dtype=torch.bool)
    mask[list(edge_indices_to_remove)] = False
    remaining_edges = subset_edges[:, mask]
    return remaining_edges


def edge_delete_states(data, model, subset_nodes, subset_edges, b):

    edges = data.edge_index
    x = data.x

    subset_mask = torch.zeros(edges.shape[1], dtype=torch.bool)
    for i in range(subset_edges.shape[1]):
        subset_mask = subset_mask | ((edges[0] == subset_edges[0, i]) & (edges[1] == subset_edges[1, i]))
    edges_outside_subset = edges[:, ~subset_mask]

    num_edges_in_subset = subset_edges.shape[1]
    combinations_of_k_edges = list(combinations(range(num_edges_in_subset), b))

    all_predictions = []
    for comb in combinations_of_k_edges:
        remaining_edges = remove_edges(subset_edges, comb)
        combined_edges = torch.cat([remaining_edges, edges_outside_subset], dim=1)
        # out = model(x, combined_edges)
        # predictions = F.softmax(out, dim=1)
        predictions = model(x, combined_edges)
        subset_predictions = predictions[subset_nodes]
        all_predictions.append(subset_predictions)

    return all_predictions

    








def mo_alg(data, model, VT, L):

    subset, hop_nodes, hop_edges = get_nodes_edges_at_hops(VT, L, data.edge_index)
    print(hop_nodes)
    print(hop_edges)

    num_nodes = data.edge_index.max().item() + 1
    color_y = torch.full((num_nodes,), -1, dtype=torch.long)
    for hop, nodes in hop_nodes.items():
        color_y[nodes] = hop
    data.color_y = color_y

    subg, subset_nodes, sub_edge_index, mapping = get_L_hop_subg(VT, L, data)
    plot_hop_nodes(subg)

    new_node_indices = mapping
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subset_nodes.tolist())}

    for l in range(L, -1, -1):
        sub_edge_index = hop_edges[l]
        remapped_edge_index = torch.clone(sub_edge_index)
        remapped_edge_index[0] = torch.tensor([node_mapping[node] for node in sub_edge_index[0].tolist()])
        remapped_edge_index[1] = torch.tensor([node_mapping[node] for node in sub_edge_index[1].tolist()])
        # for b in range(remapped_edge_index.size(1)):
        for b in range(2):
            all_predictions = edge_delete_states(subg, model, new_node_indices, remapped_edge_index, b)
            for predicts in all_predictions:
                print(predicts)
                # f_plus = 
                # f_minus = 
                # concise = 

                # cost_vec = 
