#!/bin/bash

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd /baishuanghao/code/BC-IB
# source activate /root/miniconda3/envs/libero
source activate bcib

LOAD_PATH=$1
EVAL_ALL=$2

IFS='/' read -ra parts <<< "$LOAD_PATH"

POLICY_NAME="${parts[6]}"       # ['bc_policy', 'bc_ib_policy']
CONFIG_NAME="${parts[7]}"       # backbone_name: ['rnn_eval', 'transformer_eval', 'vilt_eval']
ENV_NAME="${parts[8]}"          # ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"]
SEED_AND_NUM="${parts[10]}"
SEED=$(echo "$SEED_AND_NUM" | sed -n 's/.*seed\([0-9]*\)/\1/p')

# /baishuanghao/code/BC-IB/libero_exp/scripts/eval_libero.sh '/baishuanghao/code/BC-IB/outputs/libero/bc_policy/vilt/libero_goal/1130_1137_seed0' False

python eval_libero.py \
    --config-path=libero_exp/configs/${POLICY_NAME} \
    --config-name=${CONFIG_NAME}_eval \
    data.env_name=${ENV_NAME} \
    train.seed=${SEED} \
    eval.load_path=${LOAD_PATH} \
    eval_all=${EVAL_ALL}
