import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data


def plot_L_hop_subg(edge_index, vt, L):

    # Get the L-hop subgraph
    subgraph_nodes, subgraph_edge_index, _, _ = k_hop_subgraph(
        node_idx=vt, num_hops=L, edge_index=edge_index
    )

    # print(subgraph_nodes)
    # print(subgraph_edge_index)
    subgraph_data = Data(edge_index=subgraph_edge_index)
    # Convert to a NetworkX graph for visualization
    G = nx.Graph()
    G.add_nodes_from(subgraph_nodes.tolist())
    G.add_edges_from(list(subgraph_edge_index.t().tolist()))
    # print(G)

    # Plot the subgraph using NetworkX and Matplotlib
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=20, font_color='black', font_weight='bold', edge_color='gray')
    plt.title(f"{L}-hop neighbor subgraph of node {vt}")
    plt.show()
