import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

from torch_scatter import scatter
from torch_geometric.nn import TransformerConv

################################################################################################
################################################################################################
################################################################################################
"""
Module: Time-encoder
"""

from model import FeatEncode



################################################################################################
################################################################################################
################################################################################################
"""
Module: Self-attention-mixer
"""
class MixerBlock(nn.Module):
    def __init__(self, hidden_channels, heads, dropout=0):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.conv = TransformerConv(hidden_channels, hidden_channels//heads, heads)
        self.dropout = dropout

    def gat_conv(self, x, adj):
        x = self.layernorm(x)
        x = self.conv(x, adj)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, adj):
        return x + self.conv(x, adj)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.conv.reset_parameters()

class SAMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=1,
                ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)

        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(MixerBlock(hidden_channels, heads, dropout))

        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()

        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

        
    def forward_fully_connected(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        device = edge_feats.device
        x = self.feat_encoder(edge_feats, edge_ts)

        # if two nodes belongs to the same subgraph, then it has same batch_inds
        batch_inds = torch.arange(batch_size).repeat_interleave(self.per_graph_size).to(device)
        batch_inds = batch_inds[inds] # select the activate nodes

        unique_batch_inds = batch_inds.unique() # activate nodes
        
        row = batch_inds.view(-1, 1).repeat(1, len(batch_inds))
        col = batch_inds.view(1, -1).repeat(len(batch_inds), 1)
        edge_inds = torch.stack(torch.where(row == col))
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x, edge_inds)
        x = self.layernorm(x)
        # x = scatter(x, batch_inds, dim=0, reduce="sum")[unique_batch_inds] / self.per_graph_size
        x = scatter(x, batch_inds, dim=0, reduce="mean")[unique_batch_inds]
        x = self.mlp_head(x)
        
        out = torch.zeros((batch_size, x.size(1))).to(device)
        out[unique_batch_inds] = out[unique_batch_inds] + x
        return out
    
    def forward_connected_to_root(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        device = edge_feats.device
        x = self.feat_encoder(edge_feats, edge_ts)

        # if two nodes belongs to the same subgraph, then it has same batch_inds
        batch_inds = torch.arange(batch_size).repeat_interleave(self.per_graph_size).to(device)
        batch_inds = batch_inds[inds] # select the activate nodes

        unique_batch_inds = batch_inds.unique() # activate nodes
        
        row = batch_inds 
        col = torch.arange(len(row)).to(device)
        edge_inds = torch.stack([row, col])
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x, edge_inds)
        x = self.layernorm(x)
        # x = scatter(x, batch_inds, dim=0, reduce="mean")[unique_batch_inds]
        x = scatter(x, batch_inds, dim=0, reduce="sum")[unique_batch_inds] / self.per_graph_size
        x = self.mlp_head(x)
        
        out = torch.zeros((batch_size, x.size(1))).to(device)
        out[unique_batch_inds] = out[unique_batch_inds] + x
        return out
    
    def forward(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        
        return self.forward_fully_connected(edge_feats, edge_ts, batch_size, inds)
        # return self.forward_connected_to_root(edge_feats, edge_ts, batch_size, inds)
    
################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""

from model import EdgePredictor_per_node
    
class Mixer_per_node(nn.Module):
    """
    Wrapper of MLPMixer and EdgePredictor
    """
    def __init__(self, gat_mixer_configs, edge_predictor_configs):
        super(Mixer_per_node, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = SAMixer(**gat_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        
        self.creterion = nn.BCEWithLogitsLoss(reduction='none')
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, model_inputs, has_temporal_neighbors, neg_samples, node_feats):
        pos_mask, neg_mask = self.pos_neg_mask(has_temporal_neighbors, neg_samples)
        
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        
        loss_pos = self.creterion(pred_pos, torch.ones_like(pred_pos))[pos_mask].mean()
        loss_neg = self.creterion(pred_neg, torch.zeros_like(pred_neg))[neg_mask].mean()
        
        # compute roc and precision score
        
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().detach()
        y_true = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0).cpu().detach()
        acc = average_precision_score(y_true, y_pred)
        if neg_samples > 1:
            auc = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0)
            auc = 1 / (auc+1)
        else:
            auc = roc_auc_score(y_true, y_pred)

        return loss_pos + loss_neg, acc, auc 
    
    def pos_neg_mask(self, mask, neg_samples):
        num_edge = len(mask) // (neg_samples + 2)
        src_mask = mask[:num_edge]
        pos_dst_mask = mask[num_edge:2 * num_edge]
        neg_dst_mask = mask[2 * num_edge:]

        pos_mask = [(i and j) for i,j in zip(src_mask, pos_dst_mask)]
        neg_mask = [(i and j) for i,j in zip(src_mask * neg_samples, neg_dst_mask)]
        return pos_mask, neg_mask
    
    
################################################################################################
################################################################################################
################################################################################################

"""
Module: Node classifier
"""


from model import NodeClassificationModel