import numpy as np
import math
from collections import deque, defaultdict
import networkx as nx
import copy
from tqdm.std import tqdm
import random

import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from src.plot import plot_L_hop_subg


class ApxSXI:
    
    def __init__(self, G, model, VT, k, L, epsilon):
        self.G = G # original graph
        self.model = model  # gnn model
        self.k = k  # size of the skyline set
        self.VT = VT  # test nodes
        self.L = L  # num of gnn layers
        self.epsilon = epsilon
        self.k_sky_lst = []

        self.ipf_lst = []
        self.igd_lst = []
        self.ms_lst = []
        self.ipf = 0
        self.igd = np.inf


    def get_edge_sets_by_hop(self, vt):

        L = self.L
        edge_index = self.G.edge_index

        node_idx, edge_index_sub, _, original_edge_mask = k_hop_subgraph(vt, L, edge_index, relabel_nodes=False)
        _, _, _, ori_mask = k_hop_subgraph(vt, L+1, edge_index, relabel_nodes=False)
        selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
        subg_size = selected_edge_positions.size(0)
        
        hop_distances = {node.item(): float('inf') for node in node_idx}
        hop_distances[vt] = 0
        queue = deque([vt])
        
        while queue:
            current_node = queue.popleft()
            current_hop = hop_distances[current_node]
            
            for edge_idx in selected_edge_positions:
                src, dst = edge_index[:, edge_idx]
                if src.item() == current_node:
                    if hop_distances[dst.item()] == float('inf'):
                        hop_distances[dst.item()] = current_hop + 1
                        queue.append(dst.item())
                elif dst.item() == current_node:
                    if hop_distances[src.item()] == float('inf'):
                        hop_distances[src.item()] = current_hop + 1
                        queue.append(src.item())
        
        edges_by_hop = defaultdict(list)
        edge_masks_by_hop = {}
        for edge_idx in selected_edge_positions:
            src, dst = edge_index[:, edge_idx]
            src_hop = hop_distances[src.item()]
            dst_hop = hop_distances[dst.item()]
            
            edge_hop = min(src_hop, dst_hop) + 1
            edges_by_hop[edge_hop].append(edge_idx.item())

        for hop in range(1, L + 2):
            mask = original_edge_mask.clone()
            if hop in edges_by_hop:
                for future_hop in range(hop + 1, L + 2):
                    for edge_idx in edges_by_hop[future_hop]:
                        mask[edge_idx] = False
            edge_masks_by_hop[hop] = mask
        
        return edges_by_hop, edge_masks_by_hop, subg_size, ori_mask


    def compute_fidelity(self, node_idx, edge_mask, ori_mask):

        model = self.model
        data = self.G
        original_edge_index = self.G.edge_index

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


    def find_argmin_s(self, DRG):
        min_key = None
        min_size = float('inf')
        for key, value in DRG.items():
            current_size = len(value)
            if current_size < min_size:
                min_size = current_size
                min_key = key
        return min_key, min_size


    def update_sx(self, idx_s, DRG, s):
        idx_s = str(idx_s)
        if idx_s not in DRG:
            if len(DRG) < self.k:
                # DRG[idx_s].append(s)
                DRG[idx_s] = [s]
            else:
                s_overline, min_size = self.find_argmin_s(DRG)
                tot = sum(len(value) for value in DRG.values())
                if min_size < tot/self.k:
                    del DRG[s_overline]
                    DRG[idx_s].append(s)
        elif DRG[idx_s][0].sum().item() > s.sum().item():
            DRG[idx_s].insert(0,s)
        elif DRG[idx_s][0].sum().item() <= s.sum().item():
            DRG[idx_s].append(s)
        return DRG


    def generate_k_skylines(self):
        edge_index = self.G.edge_index
        # for vt in self.VT:
        for vt in tqdm(self.VT, desc='num VT'):
            vt = vt.item()
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L, edge_index, relabel_nodes=False)
            _, _, _, ori_mask = k_hop_subgraph(vt, self.L+1, edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)

            _, _, _, l2_edge_mask = k_hop_subgraph(vt, 2, edge_index, relabel_nodes=False)
            # fplus_ori, fminus_ori, factual_ori, counterfactual_ori = self.compute_fidelity(vt, l2_edge_mask, ori_mask)
            # print(f'l hop graph: factual_ori: {factual_ori}, counterfactual_ori: {counterfactual_ori}')

            # plot_L_hop_subg(edge_index, vt, self.L)

            DRG = defaultdict(list)
            k_sky = []
            cur_v = vt
            epoch = 10
            # if epoch < 10:
            #     epoch = 10
            visited = []
            s_0 = torch.zeros(edge_index.size(1), dtype=torch.bool)

            while epoch > 1:
                # print(f'epoch:{epoch}')
                # print(f'cur_v:{cur_v}')
                edge_size = s_0.sum().item() + 1
                t_star = (None, [0, 0, 0])
                if edge_size == 0:
                    fplus_0 = 0
                    fminus_0 = 0
                else:
                    fplus_0, fminus_0, _, _ = self.compute_fidelity(vt, s_0, ori_mask)

                source_edges = (edge_index[0] == cur_v)
                target_edges = (edge_index[1] == cur_v)
                # involving_edges = source_edges | target_edges
                involving_edges = target_edges
                edge_positions = involving_edges.nonzero(as_tuple=True)[0]

                # print(f'cur_v:{cur_v}, edge_positions:{edge_positions}')

                for edge_pos in edge_positions:
                    # print(f'position:{edge_pos}')
                    if edge_pos in visited:
                        continue
                    s = s_0.clone()
                    s[edge_pos] = True
                    fplus, fminus, factual, counterfactual = self.compute_fidelity(vt, s, original_edge_mask)
                    if not (factual or counterfactual):
                        # print('invalid')
                        continue
                    t = (edge_pos, [fplus-fplus_0, fminus-fminus_0, -1/subg_size])
                    p_s = [fplus, fminus, 1-(math.log(edge_size)/math.log(subg_size))]
                    idx_s = []
                    for i in range(len(p_s)-1):
                        if p_s[i] <= 0:
                            tmp = math.floor(math.log(1e-6,(1+self.epsilon)))
                        else:
                            tmp = math.floor(math.log(p_s[i],(1+self.epsilon)))
                        idx_s.append(tmp)
                    idx_s = "".join(str(i) for i in idx_s)
                    # print(f'idx_s:{idx_s}')
        
                    DRG = self.update_sx(idx_s, DRG, s)
                    if np.mean(t_star[1]) < np.mean(t[1]):
                        t_star = t
                
                epoch = epoch - 1

                # result = list(filter(lambda x: x not in visited, edge_positions))
                # e_star = random.choice(result)
                e_star = random.choice(edge_positions)
                # if t_star[0] == None:
                #     # result = list(filter(lambda x: x not in visited, edge_positions))
                #     # e_star = random.choice(result)
                #     e_star = random.choice(edge_positions)
                # else:
                #     e_star = t_star[0]
                
                visited.append(e_star)
                s_0[e_star] = True

                source_node = edge_index[0][e_star]
                target_node = edge_index[1][e_star]
                if source_node.item() == cur_v:
                    neighbor_u = target_node
                else:
                    neighbor_u = source_node
                cur_v = neighbor_u
                # print(f'epoch:{epoch}')

            k_sky = [DRG[key][0] for key in list(DRG.keys())]
            if len(k_sky) == 0:
                k_sky.append(l2_edge_mask)
            # print(f'k_sky:{k_sky}')
            self.k_sky_lst.append((vt, k_sky))


    def IPF(self):
        for vt, k_sky in self.k_sky_lst:
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L, self.G.edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)
            # print(f'vt: {vt} subg_size:{subg_size}')
            score_lst = []
            for mask in k_sky:
                fidelity_plus, fidelity_minus, _, _ = self.compute_fidelity(vt, mask, original_edge_mask)
                conc = 1 - (math.log(mask.sum().item()) / math.log(subg_size))
                # print(f'fidelity_plus:{fidelity_plus}, fidelity_minus:{fidelity_minus}, conc:{conc}')
                tmp = (fidelity_plus + fidelity_minus + conc)/3
                score_lst.append(tmp)
            score = np.mean(score_lst)
            self.ipf_lst.append((vt, score))
            self.ipf = np.mean([score for vt, score in self.ipf_lst])


    def IGD(self, r_container):

        def get_rank_lst(mask_list, num, ori_mask):
            phi_set = []
            for mask in mask_list:
                fidelity_plus, fidelity_minus, _, _ = self.compute_fidelity(vt, mask, ori_mask)
                conc = 1 - (math.log(mask.sum().item()) / math.log(subg_size))
                phi_set.append((fidelity_plus, fidelity_minus, conc))

            fplus_rank_lst = sorted(phi_set, key=lambda x: x[0], reverse=True)
            fminus_rank_lst = sorted(phi_set, key=lambda x: x[1], reverse=True)
            conc_rank_lst = sorted(phi_set, key=lambda x: x[2], reverse=True)

            fplus_top_k = [item[0] for item in fplus_rank_lst][:num]
            fminus_top_k = [item[1] for item in fminus_rank_lst][:num]
            conc_top_k = [item[2] for item in conc_rank_lst][:num]

            return fplus_top_k, fminus_top_k, conc_top_k

        k_sky_dict = dict(self.k_sky_lst)

        for vt, r_set in r_container:
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L, self.G.edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)

            fplus_top_k, fminus_top_k, conc_top_k = get_rank_lst(r_set, self.k, original_edge_mask)
            fplus_top_1, fminus_top_1, conc_top_1 = get_rank_lst(k_sky_dict[vt], 1, original_edge_mask)

            fplus_igd = np.mean([(abs(fplus_top_1[0] - score)) for score in fplus_top_k])
            fminus_igd = np.mean([(abs(fminus_top_1[0] - score)) for score in fminus_top_k])
            conc_igd = np.mean([(abs(conc_top_1[0] - score)) for score in conc_top_k])

            score = (fplus_igd + fminus_igd + conc_igd) / 3

            self.igd_lst.append((vt, score))
            self.igd = np.mean([score for vt, score in self.igd_lst])


    def MS(self, r_container):
        def get_best(mask_list, ori_mask):
            phi_set = []
            for mask in mask_list:
                fidelity_plus, fidelity_minus, _, _ = self.compute_fidelity(vt, mask, ori_mask)
                conc = 1 - (math.log(mask.sum().item()) / math.log(subg_size))
                phi_set.append((fidelity_plus, fidelity_minus, conc))

            fplus_rank_lst = sorted(phi_set, key=lambda x: x[0], reverse=True)
            fminus_rank_lst = sorted(phi_set, key=lambda x: x[1], reverse=True)
            conc_rank_lst = sorted(phi_set, key=lambda x: x[2], reverse=True)

            fplus_top_k = [item[0] for item in fplus_rank_lst][0]
            fminus_top_k = [item[1] for item in fminus_rank_lst][0]
            conc_top_k = [item[2] for item in conc_rank_lst][0]

            return fplus_top_k, fminus_top_k, conc_top_k

        k_sky_dict = dict(self.k_sky_lst)

        fplus_ms_lst = []
        fminus_ms_lst = []
        conc_ms_lst = []
        for vt, r_set in r_container:
            _, _, _, original_edge_mask = k_hop_subgraph(vt, self.L+1, self.G.edge_index, relabel_nodes=False)
            selected_edge_positions = torch.nonzero(original_edge_mask, as_tuple=False).squeeze()
            subg_size = selected_edge_positions.size(0)

            global_fplus, global_fminus, global_conc = get_best(r_set, original_edge_mask)
            our_fplus, our_fminus, our_conc = get_best(k_sky_dict[vt], original_edge_mask)

            if global_fplus == 0:
                global_fplus = 1
            if global_fminus == 0:
                global_fminus = 1
            if global_conc == 0:
                global_conc = 1

            fplus_ms = our_fplus/global_fplus
            fminus_ms = our_fminus/global_fminus
            conc_ms = our_conc/global_conc

            fplus_ms_lst.append(fplus_ms)
            fminus_ms_lst.append(fminus_ms)
            conc_ms_lst.append(conc_ms)

        self.ms_lst = [np.mean(fplus_ms_lst), np.mean(fminus_ms_lst), np.mean(conc_ms_lst)]




