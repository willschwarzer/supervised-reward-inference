#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --array=0-11
#SBATCH --job-name=lb-polg12
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

set -euo pipefail

MODEL_TYPE="${MODEL_TYPE:?Set MODEL_TYPE=cnn or MODEL_TYPE=unet}"
if [[ "${MODEL_TYPE}" != "cnn" && "${MODEL_TYPE}" != "unet" ]]; then
  echo "MODEL_TYPE must be cnn or unet, got '${MODEL_TYPE}'" >&2
  exit 2
fi

mkdir -p /home/submission_user/work/logs/learning_biases_bridge
cd .
module load conda/latest

AGENT_NAMES=(optimal naive sophisticated myopic overconfident underconfident)
NUM_AGENTS=${#AGENT_NAMES[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
AGENT_IDX=$((TASK_ID % NUM_AGENTS))
BETA_IDX=$((TASK_ID / NUM_AGENTS))
AGENT_NAME="${AGENT_NAMES[$AGENT_IDX]}"

if [[ "${BETA_IDX}" -eq 0 ]]; then
  BETA_LABEL="default"
  EXPORT_EXTRA_ARGS_LOCAL=""
elif [[ "${BETA_IDX}" -eq 1 ]]; then
  BETA_LABEL="none"
  EXPORT_EXTRA_ARGS_LOCAL="--beta none"
else
  echo "Unexpected BETA_IDX=${BETA_IDX} for TASK_ID=${TASK_ID}" >&2
  exit 3
fi

export SRI_ENV="${SRI_ENV:-meta-world}"
export LB_ENV="${LB_ENV:-pemirl_tf1}"
export EVAL_ENV="${EVAL_ENV:-pemirl_tf1}"

export AGENTS="${AGENT_NAME}"
export SEEDS="${SEEDS:-0}"
export EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS_LOCAL}"
export SHAH_EXTRA_ARGS="${SHAH_EXTRA_ARGS:---batchsize 20 --verbosity 0}"
export SRI_EVAL_EXTRA_ARGS="${SRI_EVAL_EXTRA_ARGS:-}"
export SRI_DUAL_CHECKPOINT_EVAL="${SRI_DUAL_CHECKPOINT_EVAL:-1}"
export SRI_TRAIN_EXTRA_ARGS="${SRI_TRAIN_EXTRA_ARGS:---dem-encoder-type ${MODEL_TYPE} --demonstration-source policy --epochs 100 --regularization-type none}"

ARRAY_ROOT="artifacts/learning_biases_bridge_polgrid12_${MODEL_TYPE}_job${SLURM_ARRAY_JOB_ID}"
export BASE_DIR="${BASE_DIR:-${ARRAY_ROOT}/${BETA_LABEL}_${AGENT_NAME}}"

export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-submission-learning-biases}"
export WANDB_GROUP="${WANDB_GROUP:-lb-polgrid12-${MODEL_TYPE}-${SLURM_ARRAY_JOB_ID}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-lb-polgrid12-${MODEL_TYPE}-${BETA_LABEL}-${AGENT_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-learning_biases,bridge,sri,policy,${MODEL_TYPE},grid12,array}"
export WANDB_TRAIN_ENABLE="${WANDB_TRAIN_ENABLE:-1}"
export WANDB_TRAIN_PROJECT="${WANDB_TRAIN_PROJECT:-${WANDB_PROJECT}}"
export WANDB_TRAIN_ENTITY="${WANDB_TRAIN_ENTITY:-${WANDB_ENTITY}}"
export WANDB_TRAIN_GROUP="${WANDB_TRAIN_GROUP:-lb-polgrid12-sri-${MODEL_TYPE}-${SLURM_ARRAY_JOB_ID}}"
export WANDB_TRAIN_TAGS="${WANDB_TRAIN_TAGS:-learning_biases,bridge,sri_loss,policy,${MODEL_TYPE},grid12,array}"

echo "[polgrid12] task=${TASK_ID} model=${MODEL_TYPE} agent=${AGENT_NAME} beta=${BETA_LABEL}"
echo "[polgrid12] BASE_DIR=${BASE_DIR}"
echo "[polgrid12] AGENTS=${AGENTS} EXPORT_EXTRA_ARGS='${EXPORT_EXTRA_ARGS}'"
echo "[polgrid12] SRI_TRAIN_EXTRA_ARGS=${SRI_TRAIN_EXTRA_ARGS}"
echo "[polgrid12] SRI_DUAL_CHECKPOINT_EVAL=${SRI_DUAL_CHECKPOINT_EVAL}"

bash scripts/learning_biases_bridge/run_matrix.sh

echo "[polgrid12] done task=${TASK_ID} model=${MODEL_TYPE} agent=${AGENT_NAME} beta=${BETA_LABEL}"
