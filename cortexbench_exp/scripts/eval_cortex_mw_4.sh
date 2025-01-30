#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export MUJOCO_GL=egl  
export CUDA_VISIBLE_DEVICES=0

cd /baishuanghao/code/BC-IB
source activate bcib

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/eval_cortex_mw_4.sh True

EVAL_ALL=$1

LOAD_PATHS=(
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