#!/bin/bash

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd /baishuanghao/code/BC-IB
# source activate /root/miniconda3/envs/libero
source activate bcib

EVAL_ALL=$1

# /baishuanghao/code/BC-IB/libero_exp/scripts/eval_libero_all.sh False
# /baishuanghao/code/BC-IB/libero_exp/scripts/eval_libero_all.sh True

LOAD_PATHS=(
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_goal/ratio0.9_scale0.0001_0.1/0123_2208_seed0'
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_goal/ratio0.9_scale0.0001_0.1/0124_0814_seed1'
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_goal/ratio0.9_scale0.0001_0.1/0124_1818_seed2'
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_object/ratio0.9_scale0.0001_0.1/0123_2207_seed0'
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_object/ratio0.9_scale0.0001_0.1/0124_0954_seed1'
    '/baishuanghao/code/BC-IB/outputs/libero/bc_ib_policy/vilt/libero_object/ratio0.9_scale0.0001_0.1/0124_2159_seed2'
)

for LOAD_PATH in "${LOAD_PATHS[@]}"; do
    IFS='/' read -ra parts <<< "$LOAD_PATH"

    POLICY_NAME="${parts[6]}"       # ['bc_policy', 'bc_ib_policy']
    CONFIG_NAME="${parts[7]}"       # backbone_name: ['rnn_eval', 'transformer_eval', 'vilt_eval']
    ENV_NAME="${parts[8]}"          # ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"]
    SEED_AND_NUM="${parts[10]}"
    SEED=$(echo "$SEED_AND_NUM" | sed -n 's/.*seed\([0-9]*\)/\1/p')

    python eval_libero.py \
        --config-path=libero_exp/configs/${POLICY_NAME} \
        --config-name=${CONFIG_NAME}_eval \
        data.env_name=${ENV_NAME} \
        train.seed=${SEED} \
        eval.load_path=${LOAD_PATH} \
        eval_all=${EVAL_ALL}
done
