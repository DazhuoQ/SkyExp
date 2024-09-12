import random
import os
import numpy as np
import yaml
import torch
from torch_geometric.datasets import Planetoid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dataset_func(config, random_seed):
    
    data_dir = "./datasets"
    data_name = config['data_name']
    data_size = config['data_size']
    num_class = config['output_dim']
    num_test = config['num_test']
    random_seed = config['random_seed']
    os.makedirs(data_dir, exist_ok=True)
    set_seed(random_seed)
    num_train_per_class = (data_size - num_test)//num_class
    data = Planetoid(root=data_dir, name=data_name, split='random', num_train_per_class=num_train_per_class, num_val=0, num_test=num_test)[0]
    return data

