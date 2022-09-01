import numpy as np
from sampler_core import ParallelSampler

    
##############################################################################
##############################################################################
##############################################################################
# get sampler
class NegLinkSampler:
    """
    From https://github.com/amazon-research/tgl/blob/main/sampler.py
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)
    
def get_parallel_sampler(g, num_neighbors=10):
    """
    Function wrapper of the C++ sampler (https://github.com/amazon-research/tgl/blob/main/sampler_core.cpp)
    Sample the 1-hop most recent neighbors of each node
    """

    configs = [
        g['indptr'],       # indptr --> fixed: data info
        g['indices'],      # indices --> fixed: data info
        g['eid'],          # eid --> fixed: data info
        g['ts'],           # ts --> fixed: data info
        10, # num_thread_per_worker --> change this based on machine's setup
        1,  # num_workers --> change this based on machine's setup
        1,  # num_layers --> change this based on machine's setup
        [num_neighbors],   # num_neighbors --> hyper-parameters. Reddit 10, WIKI 30
        True,  # recent --> fixed: never touch
        False, # prop_time --> never touch
        1,     # num_history --> fixed: never touch
        0      # window_duration --> fixed: never touch
    ]
    
    sampler = ParallelSampler(*configs)       
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    return sampler, neg_link_sampler
    
##############################################################################
##############################################################################
##############################################################################
# sampling

def get_mini_batch(sampler, root_nodes, ts, num_hops): # neg_samples is not used
    """
    Call function fetch_subgraph()
    Return: Subgraph of each node. 
    """
    all_graphs = []
    
    for root_node, root_time in zip(root_nodes, ts):
        all_graphs.append(fetch_subgraph(sampler, root_node, root_time, num_hops))

    return all_graphs

def fetch_subgraph(sampler, root_node, root_time, num_hops):
    """
    Sample a subgraph for each node or node pair
    """
    all_row_col_times_nodes_eid = []

    # suppose sampling for both a single node and a node pair (two side of a link)
    if isinstance(root_node, list):
        nodes, ts = [i for i in root_node], [root_time for i in root_node]
    else:
        nodes, ts = [root_node], [root_time]
    
    # fetch all nodes+edges
    for _ in range(num_hops):
        sampler.sample(nodes, ts)
        ret = sampler.get_ret() # 1-hop recent neighbors
        row, col, eid = ret[0].row(), ret[0].col(), ret[0].eid()
        nodes, ts = ret[0].nodes(), ret[0].ts().astype(np.float32)
        row_col_times_nodes_eid = np.stack([ts[row], nodes[row], ts[col], nodes[col], eid]).T
        all_row_col_times_nodes_eid.append(row_col_times_nodes_eid)
    all_row_col_times_nodes_eid = np.concatenate(all_row_col_times_nodes_eid, axis=0)

    # remove duplicate edges and sort according to the root node time (descending)
    all_row_col_times_nodes_eid = np.unique(all_row_col_times_nodes_eid, axis=0)[::-1]
    all_row_col_times_nodes = all_row_col_times_nodes_eid[:, :-1]
    eid = all_row_col_times_nodes_eid[:, -1]

    # remove duplicate (node+time) and sorted by time decending order
    all_row_col_times_nodes = np.array_split(all_row_col_times_nodes, 2, axis=1)
    times_nodes = np.concatenate(all_row_col_times_nodes, axis=0)
    times_nodes = np.unique(times_nodes, axis=0)[::-1]
    
    # each (node, time) pair identifies a node
    node_2_ind = dict()
    for ind, (time, node) in enumerate(times_nodes):
        node_2_ind[(time, node)] = ind

    # translate the nodes into new index
    row = np.zeros(len(eid), dtype=np.int32)
    col = np.zeros(len(eid), dtype=np.int32)
    for i, ((t1, n1), (t2, n2)) in enumerate(zip(*all_row_col_times_nodes)):
        row[i] = node_2_ind[(t1, n1)]
        col[i] = node_2_ind[(t2, n2)]
        
    # fetch get time + node information
    eid = eid.astype(np.int32)
    ts = times_nodes[:,0].astype(np.float32)
    nodes = times_nodes[:,1].astype(np.int32)
    dts = root_time - ts # make sure the root node time is 0
    
    return {
        # edge info: sorted with descending row (src) node temporal order
        'row': row, 
        'col': col, 
        'eid': eid, 
        # node info
        'nodes': nodes , # sorted by the ascending order of node's dts (root_node's dts = 0)
        'dts': dts,
        # graph info
        'num_nodes': len(nodes),
        'num_edges': len(eid),
        # root info
        'root_node': root_node,
        'root_time': root_time,
    }


def construct_mini_batch_giant_graph(all_graphs, max_num_edges):
    """
    Take the subgraph computed by fetch_subgraph() and combine it into a giant graph
    Return: the new indices of the graph
    """
    
    all_rows, all_cols, all_eids, all_nodes, all_dts = [], [], [], [], []
    
    cumsum_edges = 0
    all_edge_indptr = [0]
    
    cumsum_nodes = 0
    all_node_indptr = [0]
    
    all_root_nodes = []
    all_root_times = []
    for all_graph in all_graphs:
        # record inds
        num_nodes = all_graph['num_nodes']
        num_edges = min(all_graph['num_edges'], max_num_edges)
        
        # add graph information
        all_rows.append(all_graph['row'][:num_edges] + cumsum_nodes)
        all_cols.append(all_graph['col'][:num_edges] + cumsum_nodes)
        all_eids.append(all_graph['eid'][:num_edges])
        
        all_nodes.append(all_graph['nodes'])
        all_dts.append(all_graph['dts'])

        # update cumsum
        cumsum_nodes += num_nodes
        all_node_indptr.append(cumsum_nodes)
        
        cumsum_edges += num_edges
        all_edge_indptr.append(cumsum_edges)
        
        # add root nodes
        all_root_nodes.append(all_graph['root_node'])
        all_root_times.append(all_graph['root_time'])
    # for each edges
    all_rows = np.concatenate(all_rows).astype(np.int32)
    all_cols = np.concatenate(all_cols).astype(np.int32)
    all_eids = np.concatenate(all_eids).astype(np.int32)
    all_edge_indptr = np.array(all_edge_indptr).astype(np.int32)
    
    # for each nodes
    all_nodes = np.concatenate(all_nodes).astype(np.int32)
    all_dts = np.concatenate(all_dts).astype(np.float32)
    all_node_indptr = np.array(all_node_indptr).astype(np.int32)
        
    return {
        # for edges
        'row': all_rows, 
        'col': all_cols, 
        'eid': all_eids, 
        'edts': all_dts[all_cols] - all_dts[all_rows],
        # number of subgraphs + 1
        'all_node_indptr': all_node_indptr,
        'all_edge_indptr': all_edge_indptr,
        # for nodes
        'nodes': all_nodes, 
        'dts': all_dts, 
        # general information
        'all_num_nodes': cumsum_nodes,
        'all_num_edges': cumsum_edges,
        # root nodes
        'root_nodes': np.array(all_root_nodes, dtype=np.int32), 
        'root_times': np.array(all_root_times, dtype=np.float32), 
    }

##############################################################################
##############################################################################
##############################################################################

def print_subgraph_data(subgraph_data):
    """
    Used to double check see if the sampled graph is as expected
    """
    for key, vals in subgraph_data.items():
        if isinstance(vals, np.ndarray):
            print(key, vals.shape)
        else:
            print(key, vals)