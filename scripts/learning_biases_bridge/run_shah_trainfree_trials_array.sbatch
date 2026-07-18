#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --array=0-359%30
#SBATCH --job-name=lb-shah360
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

set -euo pipefail

mkdir -p /home/submission_user/work/logs/learning_biases_bridge
cd .
module load conda/latest

METHODS=(boltzmann_planner optimal_planner joint_with_init)
AGENT_NAMES=(optimal naive sophisticated myopic overconfident underconfident)
NUM_METHODS=${#METHODS[@]}
NUM_AGENTS=${#AGENT_NAMES[@]}
NUM_BETA=2
NUM_CONDITIONS=$((NUM_AGENTS * NUM_BETA)) # 12
NUM_TRIALS="${NUM_TRIALS:-10}"
BASE_SEED="${BASE_SEED:-1000}"
TASK_ID=${SLURM_ARRAY_TASK_ID}

MAX_TASKS=$((NUM_METHODS * NUM_CONDITIONS * NUM_TRIALS))
if [[ "${TASK_ID}" -ge "${MAX_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} out of range for NUM_TRIALS=${NUM_TRIALS}" >&2
  exit 2
fi

PER_TRIAL=$((NUM_METHODS * NUM_CONDITIONS)) # 36
TRIAL_IDX=$((TASK_ID / PER_TRIAL))
REM=$((TASK_ID % PER_TRIAL))
METHOD_IDX=$((REM / NUM_CONDITIONS))
COND_IDX=$((REM % NUM_CONDITIONS))
AGENT_IDX=$((COND_IDX % NUM_AGENTS))
BETA_IDX=$((COND_IDX / NUM_AGENTS))
METHOD="${METHODS[$METHOD_IDX]}"
AGENT="${AGENT_NAMES[$AGENT_IDX]}"
SEED=$((BASE_SEED + TRIAL_IDX))

if [[ "${BETA_IDX}" -eq 0 ]]; then
  BETA_LABEL="default"
  EXPORT_BETA_ARGS=""
elif [[ "${BETA_IDX}" -eq 1 ]]; then
  BETA_LABEL="none"
  EXPORT_BETA_ARGS="--beta none"
else
  echo "Unexpected BETA_IDX=${BETA_IDX}" >&2
  exit 3
fi

ROOT_BASE="${ROOT_BASE:-artifacts/learning_biases_bridge_shah_trainfree_trials${NUM_TRIALS}_job${SLURM_ARRAY_JOB_ID}}"
METHOD_BASE="${ROOT_BASE}/${METHOD}"
LB_ENV="${LB_ENV:-pemirl_tf1}"
SHAH_EXTRA_ARGS="${SHAH_EXTRA_ARGS:---batchsize 20 --verbosity 0 --use-gpu --gamma 0.95}"
SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
MAX_GENERATION_RETRIES="${MAX_GENERATION_RETRIES:-20}"
REWARD_INFER_SIZE="${REWARD_INFER_SIZE:-8000}"

run_cmd() {
  local env_name="$1"
  shift
  local cmd="$*"
  conda run -n "${env_name}" bash -lc "${cmd}"
}

echo "[shah360] task=${TASK_ID} trial=${TRIAL_IDX}/${NUM_TRIALS} method=${METHOD} cond=${COND_IDX}/${NUM_CONDITIONS} agent=${AGENT} beta=${BETA_LABEL} seed=${SEED}"
echo "[shah360] root=${ROOT_BASE}"

run_dir="${METHOD_BASE}/${BETA_LABEL}_${AGENT}/${SEED}"
out_dir="${run_dir}/shah_methods/${METHOD}"
metrics_path="${out_dir}/shah_metrics.json"

if [[ "${SKIP_IF_DONE}" == "1" && -f "${metrics_path}" ]]; then
  echo "[shah360] skip existing ${METHOD} ${BETA_LABEL}_${AGENT} seed=${SEED}"
  exit 0
fi

mkdir -p "${run_dir}" "${out_dir}"

echo "[shah360] export ${BETA_LABEL}_${AGENT} seed=${SEED}"
run_cmd "${LB_ENV}" \
  "python -m learning_biases.bridge_export_dataset --agent '${AGENT}' --seed ${SEED} --max-generation-retries ${MAX_GENERATION_RETRIES} --reward-infer-size ${REWARD_INFER_SIZE} --out '${run_dir}' ${EXPORT_BETA_ARGS}"

echo "[shah360] run method=${METHOD} on ${BETA_LABEL}_${AGENT} seed=${SEED}"
run_cmd "${LB_ENV}" \
  "python -m learning_biases.bridge_run_shah_given_rewards --dataset '${run_dir}' --out '${out_dir}' --seed ${SEED} --method '${METHOD}' ${SHAH_EXTRA_ARGS}"

echo "[shah360] done task=${TASK_ID} method=${METHOD} trial=${TRIAL_IDX} agent=${AGENT} beta=${BETA_LABEL}"
