# Usage

## 1. LIBERO Experiments

Please refer to [scipts files](libero_exp/scripts) of LIBERO.

To start an experiment, please choose:
- `Benchmark` from `['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']`
- `Policy` from `['bc_policy', 'bc_ib_policy']`
- `Backbone` from `['mlp', 'rnn', 'transformer', 'vilt']`

### Training 

```bash
# for bc_policy
#                                       benchmark        policy      backbone     train_ratio seed
bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_policy' 'transformer' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_policy' 'vilt' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_policy' 'rnn' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_policy' 'mlp' 0.9 0

# for bc_ib_policy
bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_ib_policy' 'transformer' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_ib_policy' 'vilt' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_ib_policy' 'rnn' 0.9 0
bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_ib_policy' 'mlp' 0.9 0
```

### Evaluation

```bash
#                                       diectory of checkpoint            only evalute on final checkpoint
bash libero_exp/scripts/eval_libero.sh 'outputs/libero/bc_policy/vilt/libero_goal/1130_1137_seed0' False
```

## 2. CortexBench Experiments

Please refer to [scipts files](cortexbench_exp/scripts) of CortexBench.

To start an experiment, please choose:
- `Policy` from `['bc_policy', 'bc_ib_policy']`
- `Backbone` from `['ResNet', 'ViT', 'R3M', 'MVP', 'VC1', 'Voltron', 'MPI']`
- `Fusion Method` from `['spatial_fuse', 'temporal_fuse']`
- `Fine-tuning Method` from `['full_ft', 'partial_ft']`

### Training 

```bash
# for bc_policy
#                                        benchmark    policy   fine-tuning and fusion backbone seed
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_policy full_ft_temporal_fuse ResNet 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_policy full_ft_spatial_fuse ResNet 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_policy partial_ft_temporal_fuse VC1 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_policy partial_ft_spatial_fuse VC1 0

# for bc_ib_policy
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_ib_policy full_ft_temporal_fuse ResNet 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_ib_policy full_ft_spatial_fuse ResNet 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_ib_policy partial_ft_temporal_fuse VC1 0
bash cortexbench_exp/scripts/main_cortex_metaworld.sh bc_ib_policy partial_ft_spatial_fuse VC1 0
```

### Evaluation

```bash
bash cortexbench_exp/scripts/eval_cortex_metaworld.sh
```

