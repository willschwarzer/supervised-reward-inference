#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-18}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/airl.yml" "baselines/gail.yml")
BASELINE_CONFIG_IDX=$((IDX % ${#BASELINE_CONFIGS[@]}))
BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
DATASET_IDX=$((IDX / ${#BASELINE_CONFIGS[@]}))

echo "BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
echo "BASELINE_CONFIG: $BASELINE_CONFIG"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_baselines.yml \
    --train-config train/baselines.yml \
    --dataset-config datasets/reach_mirrored_circling.yml \
    --obs-dataset-config datasets/reach_mirrored_circling.yml \
    --orl-dataset-config datasets/reach_mirrored_circling.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/reach_baselines_n10.yml \
    --baselines-config $BASELINE_CONFIG \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \