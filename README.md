Packages need including pytorch-geometric, pytorch, pybind11.
The code are tested under cuda113 and cuda116 environment.


Step 1: Compile C++ sampler
```
python setup.py build_ext --inplace
```

Step 2: Preprocess data (from https://github.com/amazon-research/tgl)
```
python gen_graph.py --data REDDIT --add_reverse
```

Step 3: Run experiment
```
python train.py --data GDELT_lite --num_neighbors 30   | tee gdelt_lite.txt
python train.py --data REDDIT     --num_neighbors 10   | tee reddit.txt
python train.py --data WIKI       --num_neighbors 30   | tee wiki.txt
python train.py --data MOOC       --num_neighbors 20   | tee mooc.txt
python train.py --data LASTFM     --num_neighbors 10   | tee lastfm.txt
```

Model arch with hyper-parameters including `time_dims, hidden_dims, node_feat_dims, edge_feat_dims`.
```
Mixer_per_node(
  (base_model): MLPMixer(
    (feat_encoder): FeatEncode(
      (time_encoder): TimeEncode(
        (w): Linear(in_features=1, out_features=time_dims, bias=True)
      )
      (feat_encoder): Linear(in_features=time_dims+edge_feat_dims, out_features=hidden_dims, bias=True)
    )
    (layernorm): LayerNorm((hidden_dims,), eps=1e-05, elementwise_affine=True)
    (mlp_head): Linear(in_features=hidden_dims, out_features=hidden_dims, bias=True)
    (mixer_blocks): ModuleList(
      (0): MixerBlock(
        (token_layernorm): LayerNorm((hidden_dims,), eps=1e-05, elementwise_affine=True)
        (token_forward): FeedForward(
          (linear_0): Linear(in_features=hidden_dims, out_features=0.5 * hidden_dims, bias=True)
          (linear_1): Linear(in_features=0.5 * hidden_dims, out_features=hidden_dims, bias=True)
        )
        (channel_layernorm): LayerNorm((hidden_dims,), eps=1e-05, elementwise_affine=True)
        (channel_forward): FeedForward(
          (linear_0): Linear(in_features=hidden_dims, out_features=4 * hidden_dims, bias=True)
          (linear_1): Linear(in_features=4 * hidden_dims, out_features=hidden_dims, bias=True)
        )
      )
    )
  )
  (edge_predictor): EdgePredictor_per_node(
    (src_fc): Linear(in_features=node_feat_dims+hidden_dims, out_features=hidden_dims, bias=True)
    (dst_fc): Linear(in_features=node_feat_dims+hidden_dims, out_features=hidden_dims, bias=True)
    (out_fc): Linear(in_features=hidden_dims, out_features=1, bias=True)
  )
  (creterion): BCEWithLogitsLoss()
)
```