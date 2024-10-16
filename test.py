import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                            [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
data = Data(edge_index=edge_index)

# Initialize the NeighborSampler
sampler = NeighborSampler(data.edge_index, sizes=[10], batch_size=1, shuffle=True)

# Specify the test node
test_node = 1

# Sample neighbors for the test node
for batch_size, n_id, adj in sampler.sample([test_node]):
    print(f"Sampled node IDs: {n_id.tolist()}")
    print(f"Adjacency: {adj}")

    # Extract neighbor indices (excluding the test node)
    neighbors = adj[1][0].tolist()  # the first entry of adj contains the neighbors
    print(f"Neighbors: {neighbors}")
