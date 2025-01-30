#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export MUJOCO_GL=egl

cd /baishuanghao/code/BC-IB
source activate bcib

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/eval_cortex_dmc_6.sh True

EVAL_ALL=$1

LOAD_PATHS=(
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_reacher_easy-v1/demo100/0117_0211_seed2'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_stand-v1/demo100/0116_1930_seed0'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_stand-v1/demo100/0116_2311_seed1'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_stand-v1/demo100/0117_0259_seed2'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_walk-v1/demo100/0116_2015_seed0'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_walk-v1/demo100/0116_2357_seed1'
    '/baishuanghao/code/BC-IB/outputs/cortexbench/full_ft_temporal_fuse/bc_policy/ResNet/dmcontrol/dmc_walker_walk-v1/demo100/0116_2357_seed1'
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