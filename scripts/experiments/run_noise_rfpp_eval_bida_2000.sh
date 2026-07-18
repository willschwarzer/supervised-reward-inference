#!/bin/bash

# IDX=${SLURM_ARRAY_TASK_ID:-0}
# echo "IDX: $IDX"
# NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # expect idx to be 0-10
# echo "NOISE_COEFF: $NOISE_COEFF"

IDX=${SLURM_ARRAY_TASK_ID:-209}
# IDX=5
echo "IDX: $IDX"

NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
NOISE_COEFF=${NOISE_COEFFS[$NOISE_COEFF_IDX]}
DATASET_IDX=$((IDX / ${#NOISE_COEFFS[@]}))
MODEL_IDX=$DATASET_IDX

echo "NOISE_COEFF: $NOISE_COEFF"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/skip.yml \
    --dataset-config datasets/reach_noise.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/actions.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/eval_10_envs.yml \
    --dataset-args "noise_coeff=$NOISE_COEFF" \
    --model-idxs $MODEL_IDX \
    --train-args "num_epochs=2000" \
    --model-args "load_model=True" \
    --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
    --rl-args "no_task_rep=False"



    # --train-config train/random_obs_selection_even_more_obs.yml \