#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-29}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/airl_pickplace.yml" "baselines/gail_pickplace.yml" "baselines/bc_pickplace.yml")

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
echo "MODEL_IDX: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/n10_baselines.yml \
    --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/pickplace_baselines_n10_eval.yml \
    --baselines-config $BASELINE_CONFIG \
    --model-idxs $MODEL_IDX \
