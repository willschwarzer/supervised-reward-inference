#!/bin/bash

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
    --general-config general/default_experiment.yml \
    --train-config train/n1_skip.yml \
    --dataset-config datasets/reach_goal_pos_adjustment.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/default_only_rl.yml \
    --dataset-args "goal_pos_adjustment_factor=$ADJUSTMENT" \
    --model-idxs $DATASET_IDX \



    # --train-config train/random_obs_selection_even_more_obs.yml \