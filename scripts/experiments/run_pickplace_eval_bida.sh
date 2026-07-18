#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-29}
echo "IDX: $IDX"
MAX_DATASET_IDX=10
DATASET_IDX=$((IDX % MAX_DATASET_IDX))
MODEL_IDX=$DATASET_IDX
echo "MODEL_IDX: $MODEL_IDX"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/pickplace_more_obs_skip.yml \
    --dataset-config datasets/reach_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/actions.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/eval_20_envs_pp_multitask.yml \
    --model-idxs $MODEL_IDX \
    --train-args "num_epochs=500" \
    --model-args "load_model=True" \
    --inference-args "batch_size=2, episodes=50, include_extra_reward_info=True" \
    --rl-args "no_task_rep=False"

    # --dataset-config datasets/default_reach.yml \
    # --obs-dataset-config datasets/default_push.yml \
    # --orl-dataset-config datasets/default_push.yml \
    # --train-config train/random_obs_selection_default.yml \