#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=02:00:00
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --array=0-359
#SBATCH --job-name=lb-polreeval
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

set -euo pipefail

mkdir -p /home/submission_user/work/logs/learning_biases_bridge
cd .
module load conda/latest

ROOT="${ROOT:-artifacts/learning_biases_bridge_polgrid12_trials30_unet_job53334767}"
EVAL_ENV="${EVAL_ENV:-pemirl_tf1}"
GAMMA="${GAMMA:-0.95}"
EPISODE_LENGTH="${EPISODE_LENGTH:-20}"
RUN_LAST_CHECKPOINT_EVAL="${RUN_LAST_CHECKPOINT_EVAL:-1}"
LAST_PRED_FILE="${LAST_PRED_FILE:-sri_pred_reward_vec_last.npy}"
LAST_OUT_SUBDIR="${LAST_OUT_SUBDIR:-last_checkpoint_eval}"

mapfile -t PRED_FILES < <(find "${ROOT}" -path '*/sri_pred_reward_vec.npy' | sort)
NUM_RUNS="${#PRED_FILES[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [[ "${NUM_RUNS}" -eq 0 ]]; then
  echo "No SRI prediction files found under ROOT=${ROOT}" >&2
  exit 2
fi

if [[ "${TASK_ID}" -ge "${NUM_RUNS}" ]]; then
  echo "TASK_ID=${TASK_ID} is out of range for ROOT=${ROOT} (NUM_RUNS=${NUM_RUNS})" >&2
  exit 3
fi

pred_path="${PRED_FILES[$TASK_ID]}"
run_dir="$(dirname "${pred_path}")"
shah_metrics_path="${run_dir}/shah_metrics.json"

echo "[policy-reeval] task=${TASK_ID}/${NUM_RUNS} ROOT=${ROOT}"
echo "[policy-reeval] run_dir=${run_dir}"
echo "[policy-reeval] pred_path=${pred_path}"
echo "[policy-reeval] gamma=${GAMMA} episode_length=${EPISODE_LENGTH}"

eval_cmd=(
  python -m sri.learning_biases_bridge.evaluate_sri_policy
  --dataset "${run_dir}"
  --pred "${pred_path}"
  --out "${run_dir}"
  --gamma "${GAMMA}"
  --episode-length "${EPISODE_LENGTH}"
)
if [[ -f "${shah_metrics_path}" ]]; then
  eval_cmd+=(--shah-metrics "${shah_metrics_path}")
fi
conda run -n "${EVAL_ENV}" "${eval_cmd[@]}"

if [[ "${RUN_LAST_CHECKPOINT_EVAL}" == "1" ]]; then
  last_pred_path="${run_dir}/${LAST_PRED_FILE}"
  if [[ -f "${last_pred_path}" ]]; then
    last_out_dir="${run_dir}/${LAST_OUT_SUBDIR}"
    echo "[policy-reeval] last_pred_path=${last_pred_path}"
    last_cmd=(
      python -m sri.learning_biases_bridge.evaluate_sri_policy
      --dataset "${run_dir}"
      --pred "${last_pred_path}"
      --out "${last_out_dir}"
      --gamma "${GAMMA}"
      --episode-length "${EPISODE_LENGTH}"
    )
    if [[ -f "${shah_metrics_path}" ]]; then
      last_cmd+=(--shah-metrics "${shah_metrics_path}")
    fi
    conda run -n "${EVAL_ENV}" "${last_cmd[@]}"
  else
    echo "[policy-reeval] skipping last-checkpoint eval; missing ${last_pred_path}"
  fi
fi

echo "[policy-reeval] done task=${TASK_ID}"
