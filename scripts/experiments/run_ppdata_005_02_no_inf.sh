#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-179}
echo "IDX: $IDX"

TRAIN_SPLITS_ARR=(0.05 0.2 0.8)
NUM_OBS_ARR=(100 1000 10000)
TRAIN_SPLITS_IDX=$((IDX % ${#TRAIN_SPLITS_ARR[@]}))
TRAIN_SPLIT=${TRAIN_SPLITS_ARR[$TRAIN_SPLITS_IDX]}
NUM_OBS_IDX=$(((IDX / ${#TRAIN_SPLITS_ARR[@]}) % ${#NUM_OBS_ARR[@]}))
NUM_OBS=${NUM_OBS_ARR[$NUM_OBS_IDX]}
NUM_CONFS=$((${#TRAIN_SPLITS_ARR[@]} * ${#NUM_OBS_ARR[@]}))
DATASET_IDX=$((IDX / $NUM_CONFS))

echo "TRAIN_SPLIT_IDX: $TRAIN_SPLITS_IDX"
echo "TRAIN_SPLIT: $TRAIN_SPLIT"
echo "NUM_OBS_IDX: $NUM_OBS_IDX"
echo "NUM_OBS: $NUM_OBS"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/n10_pp.yml \
    --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/default.yml \
    --rl-config rl/default.yml \
    --inference-config inference/default_skip_all.yml \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \
    --obs-dataset-2-idx $DATASET_IDX \
    --train-args "train_split=$TRAIN_SPLIT, num_obs=$NUM_OBS, start_prop_obs_1=0.05, end_prop_obs_1=0.2" \
