#!/bin/bash

# IDX=${SLURM_ARRAY_TASK_ID:-0}
# echo "IDX: $IDX"
# NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # expect idx to be 0-10
# echo "NOISE_COEFF: $NOISE_COEFF"

IDX=${SLURM_ARRAY_TASK_ID:-42}
echo "IDX: $IDX"

BASELINE_CONFIGS=("baselines/airl.yml" "baselines/gail.yml" "baselines/bc.yml")
NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)

# Calculate the total number of combinations
TOTAL_BASELINE_CONFIGS=${#BASELINE_CONFIGS[@]}
TOTAL_NOISE_COEFFS=${#NOISE_COEFFS[@]}
TOTAL_COMBINATIONS=$((TOTAL_BASELINE_CONFIGS * TOTAL_NOISE_COEFFS))

# Calculate indices
BASELINE_CONFIG_IDX=$((IDX % TOTAL_BASELINE_CONFIGS))
NOISE_COEFF_IDX=$(((IDX / TOTAL_BASELINE_CONFIGS) % TOTAL_NOISE_COEFFS))
DATASET_IDX=$((IDX / TOTAL_COMBINATIONS))
MODEL_IDX=$DATASET_IDX

# Select the baseline config and noise coefficient based on the calculated indices
BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
NOISE_COEFF=${NOISE_COEFFS[$NOISE_COEFF_IDX]}

echo "Selected BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
echo "Selected BASELINE_CONFIG: $BASELINE_CONFIG"
echo "Selected NOISE_COEFF_IDX: $NOISE_COEFF_IDX"
echo "Selected NOISE_COEFF: $NOISE_COEFF"
echo "MODEL_IDX: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/baselines.yml \
    --dataset-config datasets/reach_noise.yml \
    --obs-dataset-config datasets/reach_noise.yml \
    --orl-dataset-config datasets/reach_noise.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/reach_baselines_eval.yml \
    --baselines-config $BASELINE_CONFIG \
    --dataset-args "noise_coeff=$NOISE_COEFF" \
    --model-idxs $MODEL_IDX



    # --train-config train/random_obs_selection_even_more_obs.yml \