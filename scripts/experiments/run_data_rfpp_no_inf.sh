#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-83}
echo "IDX: $IDX"

# NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
# NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
TRAIN_CONFIGS=("train/08_100.yml" "train/08_1000.yml" "train/08_10000.yml" "train/02_100.yml" "train/02_1000.yml" "train/02_10000.yml" "train/005_100.yml" "train/005_1000.yml" "train/005_10000.yml")
TRAIN_CONFIG_IDX=$((IDX % ${#TRAIN_CONFIGS[@]}))
TRAIN_CONFIG=${TRAIN_CONFIGS[$TRAIN_CONFIG_IDX]}
DATASET_IDX=$((IDX / ${#TRAIN_CONFIGS[@]}))

echo "NOISE_COEFF: $NOISE_COEFF"
echo "DATASET_IDX: $DATASET_IDX"
echo "TRAIN_CONFIG_IDX: $TRAIN_CONFIG_IDX"
echo "TRAIN_CONFIG: $TRAIN_CONFIG"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config $TRAIN_CONFIG \
    --dataset-config datasets/reach_mirrored_circling.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/default.yml \
    --rl-config rl/default.yml \
    --inference-config inference/default_skip_all.yml \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX
