import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Create a simple graph with 5 nodes and edges
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                           [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)

# Assign random node features and labels (3 classes)
x = torch.rand((5, 3))  # 5 nodes, 3 features per node
y = torch.tensor([0, 1, 2, 1, 0])  # Node labels (3 classes)

# Create a PyG graph
data = Data(x=x, edge_index=edge_index, y=y)

# Convert PyG graph to NetworkX graph
G = to_networkx(data, to_undirected=True)

# Assign colors based on the node labels
colors = ['red', 'blue', 'green']
node_colors = [colors[label] for label in data.y]

# Plot the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_color='white', font_weight='bold')
plt.title("PyG Graph with Node Colors Based on Labels (3 Classes)")
plt.show()
