#!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-19}
echo "IDX: $IDX"

# NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
# NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
NOISE_COEFF=0.87
TRAIN_CONFIGS=("train/n1_skip.yml" "train/n10_skip.yml" "train/skip.yml")
TRAIN_CONFIG_IDX=$((IDX % ${#TRAIN_CONFIGS[@]}))
TRAIN_CONFIG=${TRAIN_CONFIGS[$TRAIN_CONFIG_IDX]}
DATASET_IDX=$((IDX / ${#TRAIN_CONFIGS[@]}))
MODEL_IDX=$DATASET_IDX

# defaults for optional arg-buckets
GENERAL_ARGS=""
TRAIN_ARGS="num_epochs=500"
DATASET_ARGS="noise_coeff=$NOISE_COEFF"
OBS_DATASET_ARGS=""
MODEL_ARGS="load_model=True"
RL_ARGS="no_task_rep=False"
INFERENCE_ARGS="batch_size=2, episodes=50, include_extra_reward_info=True"

# parse any --*-args flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --general-args)       GENERAL_ARGS="$GENERAL_ARGS${GENERAL_ARGS:+, }$2";         shift 2;;
    --train-args)         TRAIN_ARGS="$TRAIN_ARGS${TRAIN_ARGS:+, }$2";             shift 2;;
    --dataset-args)       DATASET_ARGS="$DATASET_ARGS${DATASET_ARGS:+, }$2";         shift 2;;
    --obs-dataset-args)   OBS_DATASET_ARGS="$OBS_DATASET_ARGS${OBS_DATASET_ARGS:+, }$2"; shift 2;;
    --model-args)         MODEL_ARGS="$MODEL_ARGS${MODEL_ARGS:+, }$2";             shift 2;;
    --rl-args)            RL_ARGS="$RL_ARGS${RL_ARGS:+, }$2";                   shift 2;;
    --inference-args)     INFERENCE_ARGS="$INFERENCE_ARGS${INFERENCE_ARGS:+, }$2";     shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

echo "NOISE_COEFF: $NOISE_COEFF"
echo "MODEL_IDX: $MODEL_IDX"
echo "TRAIN_CONFIG_IDX: $TRAIN_CONFIG_IDX"
echo "TRAIN_CONFIG: $TRAIN_CONFIG"

python -m sri.run_experiment \
    --general-config general/default_eval.yml \
    --train-config $TRAIN_CONFIG \
    --dataset-config datasets/reach_noise_fixed.yml \
    --obs-dataset-config datasets/reach_for_pickplace.yml \
    --model-config model/actions.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m.yml \
    --inference-config inference/eval_20_envs.yml \
    --model-idxs $MODEL_IDX \
    ${GENERAL_ARGS:+--general-args    "$GENERAL_ARGS"} \
    ${TRAIN_ARGS:+--train-args      "$TRAIN_ARGS"} \
    ${DATASET_ARGS:+--dataset-args    "$DATASET_ARGS"} \
    ${OBS_DATASET_ARGS:+--obs-dataset-args "$OBS_DATASET_ARGS"} \
    ${MODEL_ARGS:+--model-args      "$MODEL_ARGS"} \
    ${RL_ARGS:+--rl-args         "$RL_ARGS"} \
    ${INFERENCE_ARGS:+--inference-args  "$INFERENCE_ARGS"} \
