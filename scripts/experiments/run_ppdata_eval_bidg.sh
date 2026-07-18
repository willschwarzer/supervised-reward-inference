#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"

# NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
# NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
# TRAIN_CONFIGS=("train/08_100.yml" "train/08_1000.yml" "train/08_10000.yml" "train/02_100.yml" "train/02_1000.yml" "train/02_10000.yml" "train/005_100.yml" "train/005_1000.yml" "train/005_10000.yml")
# TRAIN_CONFIGS=("train/08_100_pp.yml" "train/08_1000_pp.yml" "train/08_10000_pp.yml" "train/02_100_pp.yml" "train/02_1000_pp.yml" "train/02_10000_pp.yml" "train/005_100_pp.yml" "train/005_1000_pp.yml" "train/005_10000_pp.yml")
TRAIN_SPLITS_ARR=(0.05 0.2 0.8)
NUM_OBS_ARR=(100 1000 10000)
TRAIN_SPLITS_IDX=$((IDX % ${#TRAIN_SPLITS_ARR[@]}))
TRAIN_SPLIT=${TRAIN_SPLITS_ARR[$TRAIN_SPLITS_IDX]}
NUM_OBS_IDX=$(((IDX / ${#TRAIN_SPLITS_ARR[@]}) % ${#NUM_OBS_ARR[@]}))
NUM_OBS=${NUM_OBS_ARR[$NUM_OBS_IDX]}
NUM_CONFS=$((${#TRAIN_SPLITS_ARR[@]} * ${#NUM_OBS_ARR[@]}))
DATASET_IDX=$((IDX / $NUM_CONFS))
MODEL_IDX=$DATASET_IDX

echo "MODEL IDX: $DATASET_IDX"


python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config train/n10_pp_skip.yml \
    --dataset-config datasets/reach_mirrored_circling_avoid_obj.yml \
    --obs-dataset-config datasets/default_pickplace.yml \
    --obs-dataset-config-2 datasets/reach_for_pickplace.yml \
    --model-config model/goals.yml \
    --rl-config rl/ppo_quarter_legacy_paper_esr_no_goal.yml \
    --inference-config inference/eval_20_envs_pp_multitask.yml \
    --model-idxs $MODEL_IDX \
    --train-args "train_split=$TRAIN_SPLIT, num_obs=$NUM_OBS, num_epochs=200" \
    --model-args "load_model=True" \
    --inference-args "batch_size=2, episodes=50" \
    --rl-args "no_task_rep=False"
