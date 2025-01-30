#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

cd /baishuanghao/code/BC-IB
source activate bcib

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/eval_cortex_trifinger_2.sh True

EVAL_ALL=$1

LOAD_PATHS=(
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/move/demo100_scale0.005_0.1/0118_1131_seed0'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/move/demo100_scale0.005_0.1/0118_1139_seed1'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/move/demo100_scale0.005_0.1/0118_1145_seed2'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/reach/demo100_scale0.005_0.1/0118_1137_seed0'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/reach/demo100_scale0.005_0.1/0118_1144_seed1'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/partial_ft_temporal_fuse/bc_ib_policy/VC1/trifinger/reach/demo100_scale0.005_0.1/0118_1151_seed2'
)

for LOAD_PATH in "${LOAD_PATHS[@]}"; do
    IFS='/' read -ra parts <<< "$LOAD_PATH"

    CONFIG_NAME="${parts[6]}"
    POLiCY_NAME="${parts[7]}"
    MODEL_TYPE="${parts[8]}"
    ENV_NAME="${parts[9]}"
    TASK_NAME="${parts[10]}"
    SEED_AND_NUM="${parts[12]}"
    SEED=$(echo "$SEED_AND_NUM" | sed -n 's/.*seed\([0-9]*\)/\1/p')

    python eval_cortex.py \
        --config-path=cortexbench_exp/configs/${ENV_NAME}/${POLiCY_NAME} \
        --config-name=${CONFIG_NAME}_eval \
        env.env_name=${ENV_NAME} \
        env.task_name=${TASK_NAME} \
        train.seed=${SEED} \
        policy.embedding_type=${MODEL_TYPE} \
        eval.load_path=${LOAD_PATH} \
        eval.eval_all=${EVAL_ALL}
done    