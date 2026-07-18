#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-9}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/airl.yml" "baselines/gail.yml" "baselines/bc.yml")
# BASELINE_CONFIGS=("baselines/bc.yml")
BASELINE_CONFIG_IDX=$((IDX % ${#BASELINE_CONFIGS[@]}))
BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
DATASET_IDX=$((IDX / ${#BASELINE_CONFIGS[@]}))
MODEL_IDX=$DATASET_IDX

echo "BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
echo "BASELINE_CONFIG: $BASELINE_CONFIG"
echo "MODEL_IDX: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/baselines.yml \
    --dataset-config datasets/reach_mirrored_circling.yml \
    --obs-dataset-config datasets/reach_mirrored_circling.yml \
    --orl-dataset-config datasets/reach_mirrored_circling.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/reach_baselines_n10_eval.yml \
    --baselines-config $BASELINE_CONFIG \
    --model-idx $MODEL_IDX \
