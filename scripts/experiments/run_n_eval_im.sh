#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"

NOISE_COEFF=0.87
INFERENCE_CONFIGS=("inference/reach_baselines_n1_eval.yml" "inference/reach_baselines_n10_eval.yml" "inference/reach_baselines_eval.yml")
BASELINE_CONFIGS=("baselines/airl.yml" "baselines/gail.yml" "baselines/bc.yml")

# Calculate the total number of combinations
TOTAL_BASELINE_CONFIGS=${#BASELINE_CONFIGS[@]}
TOTAL_INFERENCE_CONFIGS=${#INFERENCE_CONFIGS[@]}
TOTAL_COMBINATIONS=$((TOTAL_BASELINE_CONFIGS * TOTAL_INFERENCE_CONFIGS))

# Calculate indices
BASELINE_CONFIG_IDX=$((IDX % TOTAL_BASELINE_CONFIGS))
INFERENCE_CONFIG_IDX=$(((IDX / TOTAL_BASELINE_CONFIGS) % TOTAL_INFERENCE_CONFIGS))
DATASET_IDX=$((IDX / TOTAL_COMBINATIONS))
MODEL_IDX=$DATASET_IDX

# Select the baseline config and inference config based on the calculated indices
BASELINE_CONFIG=${BASELINE_CONFIGS[$BASELINE_CONFIG_IDX]}
INFERENCE_CONFIG=${INFERENCE_CONFIGS[$INFERENCE_CONFIG_IDX]}

echo "Selected BASELINE_CONFIG_IDX: $BASELINE_CONFIG_IDX"
echo "Selected BASELINE_CONFIG: $BASELINE_CONFIG"
echo "Selected INFERENCE_CONFIG_IDX: $INFERENCE_CONFIG_IDX"
echo "Selected INFERENCE_CONFIG: $INFERENCE_CONFIG"
echo "MODEL_IDX: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/baselines.yml \
    --dataset-config datasets/reach_noise_fixed.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config $INFERENCE_CONFIG \
    --baselines-config $BASELINE_CONFIG \
    --dataset-args "noise_coeff=$NOISE_COEFF" \
    --model-idx $MODEL_IDX \
