from tqdm import tqdm
import torch

import time
import copy
import json

import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, add_self_loops

from data_process_utils import pre_compute_subgraphs, get_random_inds, get_subgraph_sampler
from construct_subgraph import construct_mini_batch_giant_graph, print_subgraph_data

from utils import row_norm, sym_norm

def run(model, optimizer, args, subgraphs, df, node_feats, edge_feats, mode):
    time_aggre = 0
    
    ###################################################
    # setup modes
    if mode == 'train':
        model.train()
        cur_df = df[:args.train_edge_end]
        neg_samples = args.neg_samples
        cached_neg_samples = args.extra_neg_samples
        cur_inds = 0

    elif mode == 'valid':
        model.eval()
        cur_df = df[args.train_edge_end:args.val_edge_end]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.train_edge_end

    elif mode == 'test':
        model.eval()
        cur_df = df[args.val_edge_end:]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description('%s mode with negative samples %d ...'%(mode, neg_samples))        
        
    ###################################################
    # compute + training + fetch all scores
    all_ap = []
    all_auc = []
    
    for ind in range(len(train_loader)):
        ###################################################
        if args.use_cached_subgraph == False and mode == 'train':
            subgraph_data_list = subgraphs.all_root_nodes[ind]
            mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
            subgraph_data = subgraphs.mini_batch(ind, mini_batch_inds)
        else: # valid + test
            subgraph_data_list = subgraphs[ind]
            mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
            subgraph_data = [subgraph_data_list[i] for i in mini_batch_inds]

        subgraph_data = construct_mini_batch_giant_graph(subgraph_data, args.max_edges)
        # print_subgraph_data(subgraph_data) # for debugging

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()

        if args.use_graph_structure:
            num_subgraphs = len(mini_batch_inds)
            num_of_df_links = len(subgraph_data_list) // (cached_neg_samples+2)   
            subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
            
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
        has_temporal_neighbors = [True for _ in range(len(has_temporal_neighbors))] # not using it
        
        start_time = time.time()
        loss, ap, auc = model(inputs, has_temporal_neighbors, neg_samples, subgraph_node_feats)
        if mode == 'train' and optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_aggre += (time.time() - start_time)
        
        all_ap.append(ap)
        all_auc.append(auc)

        ###################################################
        # cur_inds changes every epoch
        

        pbar.update(1)
    pbar.close()    
    
    ap, auc = sum(all_ap)/len(all_ap), sum(all_auc)/len(all_auc)
    print('%s mode with time %.4f, average precision %.4f, auc score %.4f, loss %.4f'%(mode, time_aggre, ap, auc, loss.item()))
    return ap, auc, loss.item()

def link_pred_train(model, args, g, df, node_feats, edge_feats):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    ###################################################
    # get cached data
    if args.use_cached_subgraph:
        train_subgraphs = pre_compute_subgraphs(args, g, df, mode='train')
    else:
        train_subgraphs = get_subgraph_sampler(args, g, df, mode='train')
        
    valid_subgraphs = pre_compute_subgraphs(args, g, df, mode='valid')
    test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' )
    
    ###################################################
    all_results = {
        'train_ap': [],
        'valid_ap': [],
        'test_ap' : [],
        'train_auc': [],
        'valid_auc': [],
        'test_auc' : [],
        'train_loss': [],
        'valid_loss': [],
        'test_loss': [],
    }

    best_ap, best_auc = 0, 0
    
    for epoch in range(args.epochs):
        print('>>> Epoch ', epoch+1)
        train_ap, train_auc, train_loss = run(model, optimizer, args, train_subgraphs, df, 
                                              node_feats, edge_feats, mode='train')
        with torch.no_grad():
            # second variable (optimizer) is only required for training
            valid_ap, valid_auc, valid_loss = run(copy.deepcopy(model), None, args, valid_subgraphs, df, 
                                                  node_feats, edge_feats, mode='valid')
            # second variable (optimizer) is only required for training
            test_ap,  test_auc,  test_loss  = run(copy.deepcopy(model), None, args, test_subgraphs,  df, 
                                                  node_feats, edge_feats, mode='test')  

        if valid_ap > best_ap:
            best_ap_model = copy.deepcopy(model).cpu() 
            best_ap = valid_ap
            best_epoch = epoch
            best_test_ap, best_test_auc = test_ap, test_auc

        if valid_auc > best_auc:
            best_auc = valid_auc

        if epoch > best_epoch + 20:
            break

        all_results['train_ap'].append(train_ap)
        all_results['valid_ap'].append(valid_ap)
        all_results['test_ap'].append(test_ap)
        
        all_results['valid_auc'].append(valid_auc)
        all_results['train_auc'].append(train_auc)
        all_results['test_auc'].append(test_auc)
        
        all_results['train_loss'].append(train_loss)
        all_results['valid_loss'].append(valid_loss)
        all_results['test_loss'].append(test_loss)        

    print('average precision %.4f, auc score %.4f'%(best_test_ap, best_test_auc))
    
    all_results['final_test_ap'] = best_test_ap
    all_results['final_test_auc'] = best_test_auc

    json.dump(all_results, open(args.link_pred_result_fn, 'w'))
    return best_ap_model


def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    num_duplicate = len(root_nodes) // num_links 
    num_nodes = node_feats.shape[0]

    root_inds = torch.arange(len(root_nodes)).view(num_duplicate, -1)
    root_inds = [arr.flatten() for arr in root_inds.chunk(1, dim=1)]

    output_feats = torch.zeros((len(root_nodes), node_feats.size(1))).to(args.device)
    i = start_i

    for _root_ind in root_inds:

        if i == 0 or args.structure_hops == 0:
            sign_feats = node_feats.clone()
        else:
            prev_i = max(0, i - args.structure_time_gap)
            cur_df = df[prev_i: i] # get adj's row, col indices (as undirected)
            src = torch.from_numpy(cur_df.src.values)
            dst = torch.from_numpy(cur_df.dst.values)
            edge_index = torch.stack([
                torch.cat([src, dst]), 
                torch.cat([dst, src])
            ])

            edge_index, edge_cnt = torch.unique(edge_index, dim=1, return_counts=True) 

            mask = edge_index[0]!=edge_index[1] # ignore self-loops

            adj = SparseTensor(
                # value = edge_cnt[mask].float(), # take number of edges into consideration
                value = torch.ones_like(edge_cnt[mask]).float(),
                row = edge_index[0][mask].long(),
                col = edge_index[1][mask].long(),
                sparse_sizes=(num_nodes, num_nodes)
            )
            adj_norm = row_norm(adj).to(args.device)

            sign_feats = [node_feats]
            for _ in range(args.structure_hops):
                sign_feats.append(adj_norm@sign_feats[-1])
            sign_feats = torch.sum(torch.stack(sign_feats), dim=0)

        output_feats[_root_ind] = sign_feats[root_nodes[_root_ind]]

        i += len(_root_ind) // num_duplicate

    return output_feats

@torch.no_grad()
def fetch_all_predict(model, optimizer, args, subgraphs, df, node_feats, edge_feats, mode):

    ###################################################
    # setup modes
    if mode == 'train':
        model.train()
        cur_df = df[:args.train_edge_end]
        neg_samples = args.neg_samples
        cached_neg_samples = args.extra_neg_samples
        cur_inds = 0

    elif mode == 'valid':
        model.eval()
        cur_df = df[args.train_edge_end:args.val_edge_end]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.train_edge_end

    elif mode == 'test':
        model.eval()
        cur_df = df[args.val_edge_end:]
        neg_samples = 1
        cached_neg_samples = 1
        cur_inds = args.val_edge_end

    train_loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(train_loader))
    pbar.set_description('%s mode with negative samples %d ...'%(mode, neg_samples))        
        
    ###################################################
    # compute + training + fetch all scores
    all_pos = []
    all_neg = []
    
    for ind in range(len(train_loader)):
        ###################################################
        subgraph_data_list = subgraphs[ind]
        
        mini_batch_inds = get_random_inds(len(subgraph_data_list), cached_neg_samples, neg_samples)
        subgraph_data = construct_mini_batch_giant_graph([subgraph_data_list[i] for i in mini_batch_inds], args.max_edges)
        # print_subgraph_data(subgraph_data) # for debugging

        # raw edge feats 
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()

        if args.use_graph_structure:
            num_of_df_links = len(subgraph_data_list) // (cached_neg_samples+2)   
            subgraph_node_feats = compute_sign_feats(node_feats, df, cur_inds, num_of_df_links, subgraph_data['root_nodes'], args)
            
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
        pos_predict, neg_predict = model.predict(inputs, has_temporal_neighbors, neg_samples, subgraph_node_feats)
        all_pos.append(pos_predict.detach().cpu())
        all_neg.append(neg_predict.detach().cpu())

        pbar.update(1)
    pbar.close()    
    
    all_pos = torch.cat(all_pos)
    all_neg = torch.cat(all_neg)
    return all_pos, all_neg
