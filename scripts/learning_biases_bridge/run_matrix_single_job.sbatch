#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

set -euo pipefail

mkdir -p /home/submission_user/work/logs/learning_biases_bridge

cd .
module load conda/latest

export SRI_ENV="${SRI_ENV:-meta-world}"
export LB_ENV="${LB_ENV:-pemirl_tf1}"
export EVAL_ENV="${EVAL_ENV:-pemirl_tf1}"

# Defaults: one seed across all bridge agents with protocol split sizes.
export AGENTS="${AGENTS:-optimal naive sophisticated myopic overconfident underconfident}"
export SEEDS="${SEEDS:-0}"
export BASE_DIR="${BASE_DIR:-artifacts/learning_biases_bridge_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}}"

# Keep defaults for method hyperparameters unless you override these externally.
export EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
export SHAH_EXTRA_ARGS="${SHAH_EXTRA_ARGS:---batchsize 20 --verbosity 0}"
export SRI_TRAIN_EXTRA_ARGS="${SRI_TRAIN_EXTRA_ARGS:---dem-encoder-type transformer}"
export SRI_EVAL_EXTRA_ARGS="${SRI_EVAL_EXTRA_ARGS:-}"

# W&B logging: this project name is new and will be created on first sync.
export WANDB_ENABLE="${WANDB_ENABLE:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-submission-learning-biases}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-lb-bridge-${SLURM_JOB_ID}}"
export WANDB_GROUP="${WANDB_GROUP:-lb-bridge-sbatch}"
export WANDB_TAGS="${WANDB_TAGS:-learning_biases,bridge,policy,transformer}"
export WANDB_TRAIN_ENABLE="${WANDB_TRAIN_ENABLE:-1}"
export WANDB_TRAIN_PROJECT="${WANDB_TRAIN_PROJECT:-${WANDB_PROJECT}}"
export WANDB_TRAIN_ENTITY="${WANDB_TRAIN_ENTITY:-${WANDB_ENTITY}}"
export WANDB_TRAIN_GROUP="${WANDB_TRAIN_GROUP:-lb-bridge-sri-${SLURM_JOB_ID}}"
export WANDB_TRAIN_TAGS="${WANDB_TRAIN_TAGS:-learning_biases,bridge,sri_loss,transformer}"

echo "Running bridge matrix with BASE_DIR=${BASE_DIR}"
echo "Using envs: SRI_ENV=${SRI_ENV} LB_ENV=${LB_ENV} EVAL_ENV=${EVAL_ENV}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME}"

bash scripts/learning_biases_bridge/run_matrix.sh
