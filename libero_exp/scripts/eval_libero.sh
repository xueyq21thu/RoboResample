#!/bin/bash

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

source activate bcib

LOAD_PATH=$1
EVAL_ALL=$2

# bash libero_exp/scripts/eval_libero.sh 'outputs/libero/bc_policy/vilt/libero_goal/1130_1137_seed0' False

python eval_libero.py \
    --config-path=${LOAD_PATH} \
    --config-name=config.yaml \
    eval.load_path=${LOAD_PATH} \
    eval.eval_all=${EVAL_ALL}
