#!/bin/bash

# IDX=${SLURM_ARRAY_TASK_ID:-0}
# echo "IDX: $IDX"
# NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # expect idx to be 0-10
# echo "NOISE_COEFF: $NOISE_COEFF"

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"

ADJUSTMENTS=(1.0 0.6 0.2 -0.2 -0.6 -1.0)
ADJUSTMENTS_IDX=$((IDX % ${#ADJUSTMENTS[@]}))
ADJUSTMENT=${ADJUSTMENTS[$ADJUSTMENTS_IDX]}
DATASET_IDX=$((IDX / ${#ADJUSTMENTS[@]}))

# echo "NOISE_COEFF: $NOISE_COEFF"
echo "ADJUSTMENTS_IDX: $ADJUSTMENTS_IDX"
echo "ADJUSTMENT: $ADJUSTMENT"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/n1_skip.yml \
    --dataset-config datasets/reach_goal_pos_adjustment.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/goals.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/eval_20_envs.yml \
    --dataset-args "goal_pos_adjustment_factor=$ADJUSTMENT" \
    --model-idxs $DATASET_IDX \
    --train-args "num_epochs=500" \
    --model-args "load_model=True" \
    --inference-args "batch_size=2, episodes=50" \
    --rl-args "no_task_rep=False"



    # --train-config train/random_obs_selection_even_more_obs.yml \