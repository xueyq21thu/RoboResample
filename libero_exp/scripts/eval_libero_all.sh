#!/bin/bash

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

source activate bcib

EVAL_ALL=$1

# bash libero_exp/scripts/eval_libero_all.sh False
# bash libero_exp/scripts/eval_libero_all.sh True

LOAD_PATHS=(
)

for LOAD_PATH in "${LOAD_PATHS[@]}"; do
    python eval_libero.py \
        --config-path=${LOAD_PATH} \
        --config-name=config.yaml \
        eval.load_path=${LOAD_PATH} \
        eval.eval_all=${EVAL_ALL}
done
