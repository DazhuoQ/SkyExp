import sys
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from src.model import get_model
from src.utils import load_config, dataset_func



def compute_fidelity(data, node_idx, edge_mask, ori_mask, model):

    original_edge_index = data.edge_index

    with torch.no_grad():
        y_original = F.softmax(model(data.x, original_edge_index[:, ori_mask]), dim=1)[node_idx]
        original_label = y_original.argmax()
        y_original = y_original[original_label]

    mask_edge_index = original_edge_index[:, edge_mask]

    with torch.no_grad():
        y_subgraph = F.softmax(model(data.x, mask_edge_index), dim=1)[node_idx]
        subgraph_label = y_subgraph.argmax()
        y_subgraph = y_subgraph[original_label]

    complementary_edge_index = original_edge_index[:, ~edge_mask]
    
    with torch.no_grad():
        y_complementary = F.softmax(model(data.x, complementary_edge_index), dim=1)[node_idx]
        complementary_label = y_complementary.argmax()
        y_complementary = y_complementary[original_label]

    factual = (subgraph_label == original_label)

    counterfactual = (complementary_label != original_label)

    fidelity_plus = y_original - y_complementary
    fidelity_plus = fidelity_plus.item()
    fidelity_minus = y_subgraph - y_original + 1
    fidelity_minus = fidelity_minus.item()

    return fidelity_plus, fidelity_minus, factual, counterfactual



# Replace 'your_folder_path' with the actual path to your folder
folder_path = './src/results/arxiv/ksx/'
k_sky_dict = defaultdict(list)

# Loop through all files in the folder
for filename in os.listdir(folder_path):

    # Check if the file is a .pt file
    if filename.endswith('.pt'):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}")

        name_parts = filename.split('.')
        name_without_extension = name_parts[0]

        parts = name_without_extension.split('_')
        k = int(parts[0])
        model_name = parts[1]
        L = int(parts[2])

        k_sky_lst = torch.load(file_path)

    # config = {
    #     'data_name': 'Cora',
    #     'data_size': 2708, 
    #     'input_dim': 1433,
    #     'hidden_dim': 16,
    #     'output_dim': 7,
    #     'num_test': 1,
    #     'random_seed': 42,
    #     'model_name': model_name,
    # }

    # config = {
    #     'data_name': 'FacebookPage',
    #     'data_size': 22470, 
    #     'input_dim': 128,
    #     'hidden_dim': 16,
    #     'output_dim': 4,
    #     'num_test': 1,
    #     'random_seed': 42,
    #     'model_name': model_name,
    # }

    # config = {
    #     'data_name': 'PubMed',
    #     'data_size': 19717, 
    #     'input_dim': 500,
    #     'hidden_dim': 16,
    #     'output_dim': 3,
    #     'num_test': 1,
    #     'random_seed': 42,
    #     'model_name': model_name,
    # }

    # config = {
    #     'data_name': 'AmazonComputers',
    #     'data_size': 13752, 
    #     'input_dim': 767,
    #     'hidden_dim': 16,
    #     'output_dim': 10,
    #     'num_test': 1,
    #     'random_seed': 42,
    #     'model_name': model_name,
    # }


    config = {
        'data_name': 'arxiv',
        'data_size': 169343, 
        'input_dim': 128,
        'hidden_dim': 16,
        'output_dim': 40,
        'num_test': 1,
        'random_seed': 42,
        'model_name': model_name,
    }


    data_name = config['data_name']
    random_seed = config['random_seed']

    # Get input graph
    data = dataset_func(config, random_seed)

    # Ready the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load('./models/{}_{}_model.pth'.format(data_name, model_name), weights_only=False, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    tot_lst = []
    for vt, k_sky in k_sky_lst:
        print(f'vt:{vt}')
        vt_lst = []
        for edge_mask in k_sky:
            _, _, _, original_edge_mask = k_hop_subgraph(vt, L, data.edge_index, relabel_nodes=False)
            fidelity_plus, fidelity_minus, _, _ = compute_fidelity(data, vt, edge_mask, original_edge_mask, model)
            conc = 1 - (math.log(edge_mask.sum().item()) / math.log(original_edge_mask.sum().item()))
            phi = [fidelity_plus, fidelity_minus, conc]
            vt_lst.append(phi)
        tot_lst.append(vt_lst)
 
    k_sky_dict[model_name] = tot_lst
out_dir = f'{folder_path}result_dict.pt'
torch.save(k_sky_dict, out_dir)

