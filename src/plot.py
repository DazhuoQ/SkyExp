import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

def plot_hop_nodes(data):

    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Assign colors based on the node labels
    colors = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green']
    node_colors = [colors[label] for label in data.color_y]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_color='white', font_weight='bold')
    plt.title("VT induced L-hop subgraph")
    plt.savefig("./fig/VT_subg.pdf", format="pdf", bbox_inches="tight")
