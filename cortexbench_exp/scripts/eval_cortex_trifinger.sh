#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

source activate bcib

EVAL_ALL=True
LOAD_PATHS=(
/baishuanghao/code2/BC-IB/outputs/cortexbench/full_ft_spatial_fuse/bc_ib_policy/ResNet/trifinger/move/demo100_scale0.005_0.1/0201_1745_seed0    # the directory of testing model
)

# bash cortexbench_exp/scripts/eval_cortex_trifinger.sh

for LOAD_PATH in "${LOAD_PATHS[@]}"; do
    python eval_cortex.py \
        --config-path=${LOAD_PATH} \
        --config-name=config.yaml \
        eval.load_path=${LOAD_PATH} \
        eval.eval_all=${EVAL_ALL}
done