import numpy as np
import pandas as pd
import torch

df = pd.read_csv('../GDELT/edges.csv')
select = np.arange(0, len(df), 100)

new_df = {
    'Unnamed: 0': np.arange(len(select)),
    'src': df.src.values[select],
    'dst': df.dst.values[select],
    'time': df.time.values[select],
    'int_roll': df.int_roll.values[select],
    'ext_roll': df.ext_roll.values[select],
}

# create edges.csv
new_df = pd.DataFrame(data=new_df)
new_df.to_csv('./edges.csv', index=False)

# create edge features
edge_feats = torch.load('../GDELT/edge_features.pt')
torch.save(edge_feats[select], './edge_features.pt')


# create node features
node_feats = torch.load('../GDELT/node_features.pt')
torch.save(node_feats, './node_features.pt')