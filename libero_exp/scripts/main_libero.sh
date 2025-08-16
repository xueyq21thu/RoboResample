#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl

# source activate bcib

ENV_NAME=$1             # ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
POLICY_NAME=$2          # ['bc_policy', 'bc_ib_policy']
CONFIG_NAME=$3          # backbone_name: ['mlp', 'rnn', 'transformer', 'vilt']
TRAIN_RATIO=$4          
SEED=$5

MI=1e-3
MINE=0.1

# bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_policy' 'transformer' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_policy' 'vilt' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_policy' 'rnn' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_policy' 'mlp' 0.9 0

# bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_ib_policy' 'transformer' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_ib_policy' 'vilt' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_ib_policy' 'rnn' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_ib_policy' 'mlp' 0.9 0

MUJOCO_GL=egl python train_libero.py \
    --config-path=libero_exp/configs/${POLICY_NAME} \
    --config-name=${CONFIG_NAME} \
    data.env_name=${ENV_NAME} \
    train.seed=${SEED} \
    data.train_ratio=${TRAIN_RATIO} \
    train.mine_mi_loss_scale=${MINE} \
    train.mi_loss_scale=${MI} 
    # data.dataset_path=/mnt/disk_2/guanxing/xueyq21thu/libero
