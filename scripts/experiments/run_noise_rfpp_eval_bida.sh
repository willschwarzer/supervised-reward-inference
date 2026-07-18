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
    --general-args)       GENERAL_ARGS="$GENERAL_ARGS $2";         shift 2;;
    --train-args)         TRAIN_ARGS="$TRAIN_ARGS $2";             shift 2;;
    --dataset-args)       DATASET_ARGS="$DATASET_ARGS $2";         shift 2;;
    --obs-dataset-args)   OBS_DATASET_ARGS="$OBS_DATASET_ARGS $2"; shift 2;;
    --model-args)         MODEL_ARGS="$MODEL_ARGS $2";             shift 2;;
    --rl-args)            RL_ARGS="$RL_ARGS $2";                   shift 2;;
    --inference-args)     INFERENCE_ARGS="$INFERENCE_ARGS $2";     shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

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
    ${GENERAL_ARGS:+--general-args    "$GENERAL_ARGS"} \
    ${TRAIN_ARGS:+--train-args      "$TRAIN_ARGS"} \
    ${DATASET_ARGS:+--dataset-args    "$DATASET_ARGS"} \
    ${OBS_DATASET_ARGS:+--obs-dataset-args "$OBS_DATASET_ARGS"} \
    ${MODEL_ARGS:+--model-args      "$MODEL_ARGS"} \
    ${RL_ARGS:+--rl-args         "$RL_ARGS"} \
    ${INFERENCE_ARGS:+--inference-args  "$INFERENCE_ARGS"} \



    # --train-config train/random_obs_selection_even_more_obs.yml \