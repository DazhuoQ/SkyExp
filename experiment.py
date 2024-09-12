import sys
from tqdm.std import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import *
import pandas as pd
from src.model import get_model
from src.mo_alg import mo_alg


def main(config_file, output_dir):
    # Load configuration
    config = load_config(config_file)
    data_name = config['data_name']
    model_name = config['model_name']
    random_seed = config['random_seed']
    L = config['L']
    exp_name = config['exp_name']
    vt_size = config['vt_size']
    
    
    # Get input graph
    data = dataset_func(config, random_seed)

    # Get the VT
    VT = torch.where(data.test_mask)[0]

    # Ready the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config)
    model.load_state_dict(torch.load('models/{}_{}_model.pth'.format(data_name, model_name), weights_only=True))
    model.eval()
    model.to(device)

    # log
    out = model(data)
    probs = F.softmax(out, dim=1)


    # experiments
    if exp_name == 'why':
        mo_alg(data, model, VT, L)


    # Save experiment settings
    print('Seed: '+str(config['random_seed']))
    print('Dataset: '+str(config['data_name']))
    print('Model: '+str(config['model_name']))
    print('Exp: '+str(config['exp_name']))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python experiment.py <config_file> <output_dir>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(config_file, output_dir)

