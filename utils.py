import random
import numpy as np
import torch

import os
import pandas as pd

from construct_subgraph import construct_mini_batch_giant_graph, get_parallel_sampler, get_mini_batch

import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import tqdm

##############################################################################
##############################################################################
##############################################################################
# utility function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def row_norm(adj_t):
    if isinstance(adj_t, torch_sparse.SparseTensor):
        # adj_t = torch_sparse.fill_diag(adj, 1)
        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv.view(-1, 1))
        return adj_t

def sym_norm(adj):
    if isinstance(adj, torch_sparse.SparseTensor):
        adj_t = gcn_norm(adj, add_self_loops=False) 
        return adj_t

##############################################################################
##############################################################################
##############################################################################
# load data

def load_feat(d):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)

    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)

    return node_feats, edge_feats    

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def node_cls_info(args):
    # load node label information
    ldf = pd.read_csv('DATA/{}/labels.csv'.format(args.data))
    node_role = torch.tensor(ldf['ext_roll'].values, dtype=torch.int32)
    node_labels = torch.tensor(ldf['label'].values, dtype=torch.int32)
    
    return ldf, node_role, node_labels

##############################################################################
##############################################################################
##############################################################################

@torch.no_grad()
def get_node_embeds(model, edge_feats, g, df, args):
    # for each node, sample its neighbors with the most recent neighbors (sorted) 
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)

    loader = df.groupby(df.index // args.batch_size)
    pbar = tqdm(total=len(loader))
    pbar.set_description('Compute node embeddings ...')

    ###################################################
    all_embds = []

    sampler.reset()
    for _, rows in loader:
        # root_nodes = [edge_src_node, edge_dst_node, random_neg_nodes] of size 3 * batch_size
        root_nodes = np.array(rows.node.values, dtype=np.int32)
        ts = np.array(rows.time.values, dtype=np.float32)

        # get subgraph data
        inputs = sampler, root_nodes, ts, args.sampled_num_hops, 0
        subgraph_data = get_mini_batch(*inputs)
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()

        # get mini-batch inds
        all_inds, has_temporal_neighbors = [], []

        # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
        all_edge_indptr = subgraph_data['all_edge_indptr']
        for i in range(len(all_edge_indptr)-1):
            num_edges = all_edge_indptr[i+1] - all_edge_indptr[i]
            all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
            has_temporal_neighbors.append(num_edges>0)

        ###################################################
        inputs = [
            subgraph_edge_feats.to(args.device), 
            subgraph_edts.to(args.device), 
            len(has_temporal_neighbors), 
            torch.tensor(all_inds).long()
        ]

        cur_embs = model(*inputs).clone().detach().cpu()
        all_embds.append(cur_embs)

        pbar.update(1)
    pbar.close()
    
    all_embds = torch.cat(all_embds, dim=0)
    return all_embds

###################################################
# compute hits@K score
def hits_at_K(y_pred_pos, y_pred_neg, K=50):
    y_pred_pos, y_pred_neg = y_pred_pos.flatten().detach(), y_pred_neg.flatten().detach()
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return hitsK