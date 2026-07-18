#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-59}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/bc.yml")
ADJUSTMENTS=(1.0 0.6 0.2 -0.2 -0.6 -1.0)

# Calculate the total number of combinations
TOTAL_BASELINE_CONFIGS=${#BASELINE_CONFIGS[@]}
TOTAL_ADJUSTMENTS=${#ADJUSTMENTS[@]}
TOTAL_COMBINATIONS=$((TOTAL_BASELINE_CONFIGS * TOTAL_ADJUSTMENTS))

BASELINE_CONFIG_IDX=$((IDX % TOTAL_BASELINE_CONFIGS))
ADJUSTMENTS_IDX=$(((IDX / TOTAL_BASELINE_CONFIGS) % TOTAL_ADJUSTMENTS))
DATASET_IDX=$((IDX / TOTAL_COMBINATIONS))
MODEL_IDX=$DATASET_IDX

BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
ADJUSTMENT=${ADJUSTMENTS[$ADJUSTMENTS_IDX]}

# echo "NOISE_COEFF: $NOISE_COEFF"
echo "ADJUSTMENTS_IDX: $ADJUSTMENTS_IDX"
echo "ADJUSTMENT: $ADJUSTMENT"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_baselines.yml \
    --train-config train/n1_baselines.yml \
    --dataset-config datasets/reach_goal_pos_adjustment.yml \
    --obs-dataset-config datasets/reach_noise.yml \
    --orl-dataset-config datasets/reach_noise.yml \
    --model-config model/default.yml \
    --rl-config rl/default.yml \
    --inference-config inference/reach_baselines_n1.yml \
    --baselines-config $BASELINE_CONFIG \
    --dataset-args "goal_pos_adjustment_factor=$ADJUSTMENT" \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \



    # --train-config train/random_obs_selection_even_more_obs.yml \