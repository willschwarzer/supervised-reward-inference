#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-9}
echo "IDX: $IDX"
DATASET_IDX=$IDX
echo "dataset-idx: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/pickplace_more_obs.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/default.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/default_skip_all.yml \
    --dataset-idx $DATASET_IDX \
    --obs-dataset-idx $DATASET_IDX \
    --obs-dataset-2-idx $DATASET_IDX \

    # --dataset-config datasets/default_reach.yml \
    # --obs-dataset-config datasets/default_push.yml \
    # --orl-dataset-config datasets/default_push.yml \
    # --train-config train/random_obs_selection_default.yml \