#!/bin/bash

# Define the grid search values
GRID_VALUES=(0.05)

# Calculate the number of grid values
NUM_GRID_VALUES=${#GRID_VALUES[@]}

# Calculate the dataset index and grid index based on SLURM_ARRAY_TASK_ID
IDX=${SLURM_ARRAY_TASK_ID:-9}
DATASET_IDX=$((IDX / NUM_GRID_VALUES))
GRID_IDX=$((IDX % NUM_GRID_VALUES))

# Get the grid value for start_prop_obs_1 and end_prop_obs_1
PROP_OBS=${GRID_VALUES[$GRID_IDX]}

echo "IDX: $IDX"
echo "start_prop_obs_1 and end_prop_obs_1: $PROP_OBS"
MODEL_IDX=$DATASET_IDX
echo "model-idx: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/pickplace_more_obs_skip.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/pickplace_single_only_rl.yml \
    --model-idxs $MODEL_IDX \
    --train-args "start_prop_obs_1=$PROP_OBS, end_prop_obs_1=0.2" \

    # --dataset-config datasets/default_reach.yml \
    # --obs-dataset-config datasets/default_push.yml \
    # --orl-dataset-config datasets/default_push.yml \
    # --train-config train/random_obs_selection_default.yml \