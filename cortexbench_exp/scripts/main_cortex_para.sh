#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl

cd /baishuanghao/code/BC-IB
source activate bcib

POLiCY_NAME=$1
CONFIG_NAME=$2
ENV_NAME=$3
TASK_NAME=$4
MODEL_TYPE=$5

MINE=0.1

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_para.sh bc_policy partial_ft_temporal_fuse dmcontrol dmc_walker_stand-v1 VC1
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_para.sh bc_policy partial_ft_spatial_fuse dmcontrol dmc_walker_stand-v1 VC1 

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_para.sh bc_ib_policy partial_ft_spatial_fuse dmcontrol dmc_walker_stand-v1 VC1

MI_VALUES=(0.0005 0.002 0.003)
for MI in "${MI_VALUES[@]}"; do
    python train_cortex.py \
        --config-path=cortexbench_exp/configs/${ENV_NAME}/${POLiCY_NAME} \
        --config-name=${CONFIG_NAME} \
        env.env_name=${ENV_NAME} \
        env.task_name=${TASK_NAME} \
        policy.embedding_type=${MODEL_TYPE} \
        mine_mi_loss_scale=${MINE} \
        mi_loss_scale=${MI} 
done
    