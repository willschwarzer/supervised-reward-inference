#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --array=0-119
#SBATCH --job-name=lb-polg12t
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

set -euo pipefail

MODEL_TYPE="${MODEL_TYPE:-unet}"
if [[ "${MODEL_TYPE}" != "cnn" && "${MODEL_TYPE}" != "unet" ]]; then
  echo "MODEL_TYPE must be cnn or unet, got '${MODEL_TYPE}'" >&2
  exit 2
fi

mkdir -p /home/submission_user/work/logs/learning_biases_bridge
cd .
module load conda/latest

AGENT_NAMES=(optimal naive sophisticated myopic overconfident underconfident)
NUM_AGENTS=${#AGENT_NAMES[@]}
NUM_BETA=2
NUM_CONDITIONS=$((NUM_AGENTS * NUM_BETA)) # 12
NUM_TRIALS="${NUM_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-1000}"
TASK_ID=${SLURM_ARRAY_TASK_ID}

MAX_TASKS=$((NUM_TRIALS * NUM_CONDITIONS))
if [[ "${TASK_ID}" -ge "${MAX_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is out of range for NUM_TRIALS=${NUM_TRIALS}" >&2
  exit 3
fi

TRIAL_IDX=$((TASK_ID / NUM_CONDITIONS))
COND_IDX=$((TASK_ID % NUM_CONDITIONS))
AGENT_IDX=$((COND_IDX % NUM_AGENTS))
BETA_IDX=$((COND_IDX / NUM_AGENTS))
AGENT_NAME="${AGENT_NAMES[$AGENT_IDX]}"
SEED=$((BASE_SEED + TRIAL_IDX))

if [[ "${BETA_IDX}" -eq 0 ]]; then
  BETA_LABEL="default"
  EXPORT_EXTRA_ARGS_LOCAL=""
elif [[ "${BETA_IDX}" -eq 1 ]]; then
  BETA_LABEL="none"
  EXPORT_EXTRA_ARGS_LOCAL="--beta none"
else
  echo "Unexpected BETA_IDX=${BETA_IDX} for TASK_ID=${TASK_ID}" >&2
  exit 4
fi

export SRI_ENV="${SRI_ENV:-meta-world}"
export LB_ENV="${LB_ENV:-pemirl_tf1}"
export EVAL_ENV="${EVAL_ENV:-pemirl_tf1}"

export AGENTS="${AGENT_NAME}"
export SEEDS="${SEED}"
export EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS_LOCAL}"
export SHAH_EXTRA_ARGS="${SHAH_EXTRA_ARGS:---batchsize 20 --verbosity 0}"
export SRI_EVAL_EXTRA_ARGS="${SRI_EVAL_EXTRA_ARGS:-}"
export SRI_DUAL_CHECKPOINT_EVAL="${SRI_DUAL_CHECKPOINT_EVAL:-1}"
export SRI_TRAIN_EXTRA_ARGS="${SRI_TRAIN_EXTRA_ARGS:---dem-encoder-type ${MODEL_TYPE} --demonstration-source policy --epochs 100 --regularization-type none}"

ROOT_BASE="${ROOT_BASE:-artifacts/learning_biases_bridge_polgrid12_trials${NUM_TRIALS}_${MODEL_TYPE}_job${SLURM_ARRAY_JOB_ID}}"
export BASE_DIR="${BASE_DIR:-${ROOT_BASE}/${BETA_LABEL}_${AGENT_NAME}/trial_${TRIAL_IDX}}"

# Default: keep this run local-file centric to avoid accidental W&B spam (120 jobs).
export WANDB_ENABLE="${WANDB_ENABLE:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_GROUP="${WANDB_GROUP:-}"
export WANDB_TAGS="${WANDB_TAGS:-}"
export WANDB_TRAIN_ENABLE="${WANDB_TRAIN_ENABLE:-0}"
export WANDB_TRAIN_PROJECT="${WANDB_TRAIN_PROJECT:-${WANDB_PROJECT}}"
export WANDB_TRAIN_ENTITY="${WANDB_TRAIN_ENTITY:-${WANDB_ENTITY}}"
export WANDB_TRAIN_GROUP="${WANDB_TRAIN_GROUP:-}"
export WANDB_TRAIN_TAGS="${WANDB_TRAIN_TAGS:-}"
export WANDB_TRAIN_MODE="${WANDB_TRAIN_MODE:-}"

echo "[polgrid12-trials] task=${TASK_ID} trial=${TRIAL_IDX}/${NUM_TRIALS} cond=${COND_IDX}/${NUM_CONDITIONS} model=${MODEL_TYPE}"
echo "[polgrid12-trials] agent=${AGENT_NAME} beta=${BETA_LABEL} seed=${SEED} BASE_DIR=${BASE_DIR}"
echo "[polgrid12-trials] AGENTS=${AGENTS} EXPORT_EXTRA_ARGS='${EXPORT_EXTRA_ARGS}'"
echo "[polgrid12-trials] SRI_TRAIN_EXTRA_ARGS=${SRI_TRAIN_EXTRA_ARGS}"
echo "[polgrid12-trials] SRI_DUAL_CHECKPOINT_EVAL=${SRI_DUAL_CHECKPOINT_EVAL}"

bash scripts/learning_biases_bridge/run_matrix.sh

echo "[polgrid12-trials] done task=${TASK_ID} trial=${TRIAL_IDX} agent=${AGENT_NAME} beta=${BETA_LABEL}"
