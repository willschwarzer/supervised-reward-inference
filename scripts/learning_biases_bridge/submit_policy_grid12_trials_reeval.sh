#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/learning_biases_bridge_polgrid12_trials30_unet_job53334767}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARRAY_SCRIPT="${SCRIPT_DIR}/run_policy_grid12_trials_reeval_array.sbatch"
EVAL_ENV="${EVAL_ENV:-pemirl_tf1}"
COLLATE_ENV="${COLLATE_ENV:-meta-world}"
GAMMA="${GAMMA:-0.95}"
EPISODE_LENGTH="${EPISODE_LENGTH:-20}"
RUN_LAST_CHECKPOINT_EVAL="${RUN_LAST_CHECKPOINT_EVAL:-1}"
ARRAY_LIMIT="${ARRAY_LIMIT:-}"

cd .

num_runs="$(find "${ROOT}" -path '*/sri_pred_reward_vec.npy' | wc -l | tr -d ' ')"
if [[ "${num_runs}" == "0" ]]; then
  echo "No SRI prediction files found under ROOT=${ROOT}" >&2
  exit 2
fi

array_spec="0-$((num_runs - 1))"
if [[ -n "${ARRAY_LIMIT}" ]]; then
  array_spec="${array_spec}%${ARRAY_LIMIT}"
fi

eval_job_id="$(
  sbatch --parsable \
    --array="${array_spec}" \
    --export=ALL,ROOT="${ROOT}",EVAL_ENV="${EVAL_ENV}",GAMMA="${GAMMA}",EPISODE_LENGTH="${EPISODE_LENGTH}",RUN_LAST_CHECKPOINT_EVAL="${RUN_LAST_CHECKPOINT_EVAL}" \
    "${ARRAY_SCRIPT}"
)"

collate_wrap="bash -lc 'set -euo pipefail; cd .; module load conda/latest; conda run -n \"${COLLATE_ENV}\" python -m sri.learning_biases_bridge.collate_trial_matrix --root \"${ROOT}\" --out \"${ROOT}\"'"

collate_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${eval_job_id}" \
    --job-name=lb-polcol \
    --partition=cpu,cpu-preempt \
    -A YOUR_ACCOUNT_GROUP \
    -n 1 \
    -c 2 \
    --mem=8G \
    --time=00:30:00 \
    --output=/home/submission_user/work/logs/learning_biases_bridge/%j.out \
    --error=/home/submission_user/work/logs/learning_biases_bridge/%j.err \
    --wrap "${collate_wrap}"
)"

echo "Submitted policy re-eval array: ${eval_job_id}"
echo "Submitted dependent collation job: ${collate_job_id}"
echo "ROOT=${ROOT}"
echo "GAMMA=${GAMMA}"
echo "ARRAY=${array_spec}"
