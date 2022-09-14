from tqdm import tqdm
import torch

import copy
import json

import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, add_self_loops

from construct_subgraph import construct_mini_batch_giant_graph, print_subgraph_data
from construct_subgraph import get_parallel_sampler, get_mini_batch

from link_pred_train_utils import compute_sign_feats
from utils import row_norm, sym_norm
    
@torch.no_grad()
def link_pred_eval(model, args, g, df, node_feats, edge_feats):
    model.eval()
    
    ###################################################
    all_results = {
        'test_R@1' : [],
        'test_R@5' : [],
        'test_R@10' : [],
        'test_mrr': [],
    }
    
    ###################################################
    sampler, neg_link_sampler = get_parallel_sampler(g, args.num_neighbors)
    
    cur_df = df[args.val_edge_end:]
    neg_samples = 100
    cur_inds = args.val_edge_end

    pbar = tqdm(total=len(cur_df))
    pbar.set_description('Compute R@K and MRR ...')        
        
    ###################################################
    
    for _, rows in cur_df.iterrows():
        
        # construct graph
        pos_pair = np.array([rows.src, rows.dst])
        
        root_nodes = np.concatenate([
            pos_pair, 
            neg_link_sampler.sample(neg_samples)]
        ).astype(np.int32)
        
        ts = np.tile(rows.time, neg_samples + 2).astype(np.float32)
        
        subgraph_data = get_mini_batch(sampler, root_nodes, ts, args.sampled_num_hops)
        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()

        if args.use_graph_structure:
            num_subgraphs = len(subgraph_data)
            num_of_df_links = 1
            subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, root_nodes, args)
            cur_inds += num_of_df_links
        else:
            subgraph_node_feats = None

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
        
        # forward + backward
        has_temporal_neighbors = [True for _ in range(len(has_temporal_neighbors))] # ignore all mask ???
        y_pred_pos, y_pred_neg = model.predict(inputs, has_temporal_neighbors, neg_samples, subgraph_node_feats)
        y_pred_pos, y_pred_neg = y_pred_pos.flatten().cpu().numpy(), y_pred_neg.flatten().cpu().numpy()
        
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum()
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum()
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        
        all_results['test_R@1'].append((optimistic_rank <= 1).astype(np.float32))
        all_results['test_R@5'].append((optimistic_rank <= 5).astype(np.float32))
        all_results['test_R@10'].append((optimistic_rank <= 10).astype(np.float32))
        all_results['test_mrr'].append(1./ranking_list.astype(np.float32))

        pbar.update(1)
    pbar.close()    
    
    all_results['test_R@1'] = np.mean(all_results['test_R@1'])
    all_results['test_R@5'] = np.mean(all_results['test_R@5'])
    all_results['test_R@10'] = np.mean(all_results['test_R@10'])
    all_results['test_mrr'] = np.mean(all_results['test_mrr'])
    
    print(all_results)
    return all_results


