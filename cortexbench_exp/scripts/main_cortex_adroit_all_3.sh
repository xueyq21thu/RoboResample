#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

cd /baishuanghao/code/BC-IB
source activate bcib

POLiCY_NAME=$1
CONFIG_NAME=$2
ENV_NAME=adroit
MODEL_TYPE=$3
SEED=$4

MINE=0.1
MI=1e-4

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_policy full_ft_spatial_fuse ResNet
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_policy full_ft_temporal_fuse ResNet
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_policy partial_ft_temporal_fuse VC1
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_policy partial_ft_spatial_fuse VC1

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_ib_policy full_ft_spatial_fuse ResNet
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_ib_policy full_ft_temporal_fuse ResNet
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_adroit_all_3.sh bc_ib_policy partial_ft_spatial_fuse VC1

TASK_NAMES=(pen-v0 relocate-v0)
# TASK_NAMES=(pen-v0)
# TASK_NAMES=(relocate-v0)

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    for TASK_NAME in "${TASK_NAMES[@]}"; do
        python train_cortex.py \
            --config-path=cortexbench_exp/configs/${ENV_NAME}/${POLiCY_NAME} \
            --config-name=${CONFIG_NAME} \
            env.env_name=${ENV_NAME} \
            env.task_name=${TASK_NAME} \
            train.seed=${SEED} \
            policy.embedding_type=${MODEL_TYPE} \
            mine_mi_loss_scale=${MINE} \
            mi_loss_scale=${MI} 
    done
done
    