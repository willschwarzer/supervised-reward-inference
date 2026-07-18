#!/bin/bash

# Define the grid search values
GRID_VALUES=(0.05)

# Calculate the number of grid values
NUM_GRID_VALUES=${#GRID_VALUES[@]}

# Calculate the dataset index and grid index based on SLURM_ARRAY_TASK_ID
IDX=${SLURM_ARRAY_TASK_ID:-29}
DATASET_IDX=$((IDX / NUM_GRID_VALUES))
GRID_IDX=$((IDX % NUM_GRID_VALUES))

# Get the grid value for start_prop_obs_1 and end_prop_obs_1
PROP_OBS=${GRID_VALUES[$GRID_IDX]}

echo "IDX: $IDX"
echo "dataset-idx: $DATASET_IDX"
# echo "start_prop_obs_1 and end_prop_obs_1: $PROP_OBS"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/pickplace_more_obs.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/goals.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/default_skip_all.yml \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \
    --obs-dataset-2-idx $DATASET_IDX \
    --train-args "num_epochs=500" \