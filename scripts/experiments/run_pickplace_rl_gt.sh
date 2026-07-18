#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"
MAX_DATASET_IDX=30
DATASET_IDX=$((IDX % MAX_DATASET_IDX))
echo "dataset-idx: $DATASET_IDX"
MODEL_IDX=$DATASET_IDX

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/pickplace_more_obs_skip.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/load_default.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal_gt_reward.yml \
    --inference-config inference/pickplace_single_only_rl.yml \

    # --dataset-config datasets/default_reach.yml \
    # --obs-dataset-config datasets/default_push.yml \
    # --orl-dataset-config datasets/default_push.yml \
    # --train-config train/random_obs_selection_default.yml \