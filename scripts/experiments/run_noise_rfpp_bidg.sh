#!/bin/bash

# IDX=${SLURM_ARRAY_TASK_ID:-0}
# echo "IDX: $IDX"
# NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # expect idx to be 0-10
# echo "NOISE_COEFF: $NOISE_COEFF"

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"

NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
NOISE_COEFF=${NOISE_COEFFS[$NOISE_COEFF_IDX]}
DATASET_IDX=$((IDX / ${#NOISE_COEFFS[@]}))

echo "NOISE_COEFF: $NOISE_COEFF"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/default.yml \
    --dataset-config datasets/reach_noise.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/goals.yml \
    --rl-config rl/default.yml \
    --inference-config inference/default_skip_all.yml \
    --dataset-args "noise_coeff=$NOISE_COEFF" \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \
    --train-args "num_epochs=500" \
