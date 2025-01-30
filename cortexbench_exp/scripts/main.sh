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
MODEL_NAME=$6


# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex.sh bc_policy full_ft dmcontrol dmc_walker_stand-v1 VC1 vc1_vitb

python train_cortex.py \
    --config-path=cortexbench_exp/configs/dmcontrol/bc_policy \
    --config-name=full_ft \
    env.env_name=dmcontrol \
    env.task_name=dmc_walker_stand-v1 \
    policy.embedding_type=VC1 \
    policy.embedding=vc1_vitb \
    