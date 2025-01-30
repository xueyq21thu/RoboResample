#!/bin/bash

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd /baishuanghao/code/BC-IB
# source activate /root/miniconda3/envs/libero
source activate bcib

ENV_NAME=$1             # ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"]
POLICY_NAME=$2          # ['bc_policy', 'bc_ib_policy']
CONFIG_NAME=$3          # backbone_name: ['rnn', 'transformer', 'vilt']
TRAIN_RATIO=$4

MI=1e-3
MINE=0.1

# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_spatial' 'bc_policy' 'transformer' 0.2
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_object' 'bc_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_goal' 'bc_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_10' 'bc_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_90' 'bc_policy' 'transformer' 0.2 

# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_spatial' 'bc_ib_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_object' 'bc_ib_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_goal' 'bc_ib_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_10' 'bc_ib_policy' 'transformer' 0.2 
# /baishuanghao/code/BC-IB/libero_exp/scripts/main_libero_all_4.sh 'libero_90' 'bc_ib_policy' 'transformer' 0.2 

SEEDS=(1 2)
for SEED in "${SEEDS[@]}"; do
    python train_libero.py \
        --config-path=libero_exp/configs/${POLICY_NAME} \
        --config-name=${CONFIG_NAME} \
        data.env_name=${ENV_NAME} \
        train.seed=${SEED} \
        data.train_ratio=${TRAIN_RATIO} \
        mine_mi_loss_scale=${MINE} \
        mi_loss_scale=${MI}
done
