#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-19}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/airl_pickplace.yml" "baselines/gail_pickplace.yml")

# Calculate the total number of combinations
TOTAL_BASELINE_CONFIGS=${#BASELINE_CONFIGS[@]}

# Calculate indices
BASELINE_CONFIG_IDX=$((IDX % TOTAL_BASELINE_CONFIGS))
DATASET_IDX=$((IDX / TOTAL_BASELINE_CONFIGS))
MODEL_IDX=$DATASET_IDX

# Select the baseline config based on the calculated index
BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}

echo "Selected BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
echo "Selected BASELINE_CONFIG: $BASELINE_CONFIG"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_baselines.yml \
    --train-config train/baselines.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/pickplace_baselines.yml \
    --baselines-config $BASELINE_CONFIG \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \