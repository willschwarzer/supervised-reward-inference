#!/usr/bin/env bash
set -euo pipefail

DEFAULT_AGENTS=(optimal naive sophisticated myopic overconfident underconfident)
if [ -n "${AGENTS:-}" ]; then
  read -r -a AGENT_LIST <<<"${AGENTS}"
else
  AGENT_LIST=("${DEFAULT_AGENTS[@]}")
fi
SEEDS="${SEEDS:-0}"
BASE_DIR="${BASE_DIR:-artifacts/learning_biases_bridge}"
SRI_ENV="${SRI_ENV:-meta-world}"
LB_ENV="${LB_ENV:-pemirl_tf1}"
EVAL_ENV="${EVAL_ENV:-${LB_ENV}}"
EXPORT_EXTRA_ARGS="${EXPORT_EXTRA_ARGS:-}"
SHAH_EXTRA_ARGS="${SHAH_EXTRA_ARGS:-}"
SHAH_METHODS="${SHAH_METHODS:-given_rewards}"
SRI_TRAIN_EXTRA_ARGS="${SRI_TRAIN_EXTRA_ARGS:-}"
SRI_EVAL_EXTRA_ARGS="${SRI_EVAL_EXTRA_ARGS:-}"
SRI_DUAL_CHECKPOINT_EVAL="${SRI_DUAL_CHECKPOINT_EVAL:-0}"
SRI_LAST_PRED_FILE="${SRI_LAST_PRED_FILE:-sri_pred_reward_vec_last.npy}"
SRI_LAST_OUT_SUBDIR="${SRI_LAST_OUT_SUBDIR:-last_checkpoint_eval}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_TRAIN_ENABLE="${WANDB_TRAIN_ENABLE:-0}"
WANDB_TRAIN_PROJECT="${WANDB_TRAIN_PROJECT:-${WANDB_PROJECT}}"
WANDB_TRAIN_ENTITY="${WANDB_TRAIN_ENTITY:-${WANDB_ENTITY}}"
WANDB_TRAIN_GROUP="${WANDB_TRAIN_GROUP:-}"
WANDB_TRAIN_TAGS="${WANDB_TRAIN_TAGS:-${WANDB_TAGS}}"
WANDB_TRAIN_MODE="${WANDB_TRAIN_MODE:-}"

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load conda/latest >/dev/null 2>&1 || true
fi

run_cmd() {
  local env_name="$1"
  shift
  local cmd="$*"

  if command -v conda >/dev/null 2>&1; then
    conda run -n "${env_name}" bash -lc "${cmd}"
  else
    echo "[warn] conda not found; running in current shell: ${cmd}"
    bash -lc "${cmd}"
  fi
}

mkdir -p "${BASE_DIR}"

read -r -a SHAH_METHOD_LIST <<<"${SHAH_METHODS}"
if [ "${#SHAH_METHOD_LIST[@]}" -eq 0 ]; then
  SHAH_METHOD_LIST=(given_rewards)
fi

for agent in "${AGENT_LIST[@]}"; do
  for seed in ${SEEDS}; do
    run_dir="${BASE_DIR}/${agent}/${seed}"
    mkdir -p "${run_dir}"

    echo "=== [${agent}] [seed=${seed}] export dataset ==="
    run_cmd "${LB_ENV}" \
      "python -m learning_biases.bridge_export_dataset --agent ${agent} --seed ${seed} --out '${run_dir}' ${EXPORT_EXTRA_ARGS}"

    echo "=== [${agent}] [seed=${seed}] run Shah methods: ${SHAH_METHOD_LIST[*]} ==="
    shah_metrics_for_comparison=""
    for shah_method in "${SHAH_METHOD_LIST[@]}"; do
      shah_out="${run_dir}/shah_methods/${shah_method}"
      mkdir -p "${shah_out}"
      run_cmd "${LB_ENV}" \
        "python -m learning_biases.bridge_run_shah_given_rewards --dataset '${run_dir}' --out '${shah_out}' --seed ${seed} --method ${shah_method} ${SHAH_EXTRA_ARGS}"
      if [ -f "${shah_out}/shah_metrics.json" ] && [ -z "${shah_metrics_for_comparison}" ]; then
        shah_metrics_for_comparison="${shah_out}/shah_metrics.json"
      fi
    done

    given_rewards_metrics="${run_dir}/shah_methods/given_rewards/shah_metrics.json"
    given_rewards_pred="${run_dir}/shah_methods/given_rewards/shah_inferred_rewards.npy"
    if [ -f "${given_rewards_metrics}" ]; then
      shah_metrics_for_comparison="${given_rewards_metrics}"
      cp "${given_rewards_metrics}" "${run_dir}/shah_metrics.json"
      if [ -f "${given_rewards_pred}" ]; then
        cp "${given_rewards_pred}" "${run_dir}/shah_inferred_rewards.npy"
      fi
    elif [ -n "${shah_metrics_for_comparison}" ]; then
      cp "${shah_metrics_for_comparison}" "${run_dir}/shah_metrics.json"
    fi

    echo "=== [${agent}] [seed=${seed}] train SRI-policy ==="
    train_args="${SRI_TRAIN_EXTRA_ARGS}"
    if [ "${SRI_DUAL_CHECKPOINT_EVAL}" = "1" ]; then
      if [[ "${train_args}" != *"--save-last-pred"* ]]; then
        train_args="${train_args} --save-last-pred"
      fi
    fi
    train_wb_args=""
    if [ "${WANDB_TRAIN_ENABLE}" = "1" ] || [ -n "${WANDB_TRAIN_PROJECT}" ]; then
      if [ -z "${WANDB_TRAIN_PROJECT}" ]; then
        echo "[warn] WANDB_TRAIN enabled but WANDB_TRAIN_PROJECT is empty; skipping train-time W&B."
      else
        run_name_prefix="${WANDB_RUN_NAME:-lb-bridge}"
        train_wb_args="${train_wb_args} --wandb-project '${WANDB_TRAIN_PROJECT}'"
        train_wb_args="${train_wb_args} --wandb-run-name '${run_name_prefix}-${agent}-seed${seed}-sri'"
        if [ -n "${WANDB_TRAIN_ENTITY}" ]; then
          train_wb_args="${train_wb_args} --wandb-entity '${WANDB_TRAIN_ENTITY}'"
        fi
        if [ -n "${WANDB_TRAIN_GROUP}" ]; then
          train_wb_args="${train_wb_args} --wandb-group '${WANDB_TRAIN_GROUP}'"
        fi
        if [ -n "${WANDB_TRAIN_TAGS}" ]; then
          train_wb_args="${train_wb_args} --wandb-tags '${WANDB_TRAIN_TAGS}'"
        fi
        if [ -n "${WANDB_TRAIN_MODE}" ]; then
          train_wb_args="${train_wb_args} --wandb-mode '${WANDB_TRAIN_MODE}'"
        fi
      fi
    fi
    run_cmd "${SRI_ENV}" \
      "python -m sri.learning_biases_bridge.train_sri_policy --dataset '${run_dir}' --out '${run_dir}' --seed ${seed} ${train_args} ${train_wb_args}"

    echo "=== [${agent}] [seed=${seed}] evaluate SRI-policy ==="
    if [ -n "${shah_metrics_for_comparison}" ] && [ -f "${shah_metrics_for_comparison}" ]; then
      run_cmd "${EVAL_ENV}" \
        "python -m sri.learning_biases_bridge.evaluate_sri_policy --dataset '${run_dir}' --pred '${run_dir}/sri_pred_reward_vec.npy' --out '${run_dir}' --shah-metrics '${shah_metrics_for_comparison}' ${SRI_EVAL_EXTRA_ARGS}"
    else
      run_cmd "${EVAL_ENV}" \
        "python -m sri.learning_biases_bridge.evaluate_sri_policy --dataset '${run_dir}' --pred '${run_dir}/sri_pred_reward_vec.npy' --out '${run_dir}' ${SRI_EVAL_EXTRA_ARGS}"
    fi

    if [ "${SRI_DUAL_CHECKPOINT_EVAL}" = "1" ]; then
      last_pred_path="${run_dir}/${SRI_LAST_PRED_FILE}"
      last_out_dir="${run_dir}/${SRI_LAST_OUT_SUBDIR}"
      if [ -f "${last_pred_path}" ]; then
        echo "=== [${agent}] [seed=${seed}] evaluate SRI-policy (last checkpoint) ==="
        if [ -n "${shah_metrics_for_comparison}" ] && [ -f "${shah_metrics_for_comparison}" ]; then
          run_cmd "${EVAL_ENV}" \
            "python -m sri.learning_biases_bridge.evaluate_sri_policy --dataset '${run_dir}' --pred '${last_pred_path}' --out '${last_out_dir}' --shah-metrics '${shah_metrics_for_comparison}' ${SRI_EVAL_EXTRA_ARGS}"
        else
          run_cmd "${EVAL_ENV}" \
            "python -m sri.learning_biases_bridge.evaluate_sri_policy --dataset '${run_dir}' --pred '${last_pred_path}' --out '${last_out_dir}' ${SRI_EVAL_EXTRA_ARGS}"
        fi
      else
        echo "[warn] dual checkpoint eval requested but missing ${last_pred_path}"
      fi
    fi
  done
done

echo "=== aggregate results ==="
run_cmd "${SRI_ENV}" \
  "python -m sri.learning_biases_bridge.aggregate_results --base-dir '${BASE_DIR}' --out '${BASE_DIR}'"

if [ "${WANDB_ENABLE}" = "1" ] || [ -n "${WANDB_PROJECT}" ]; then
  if [ -z "${WANDB_PROJECT}" ]; then
    echo "[warn] WANDB enabled but WANDB_PROJECT is empty; skipping W&B logging."
  else
    echo "=== log to W&B (${WANDB_PROJECT}) ==="
    WB_ARGS="--summary '${BASE_DIR}/summary.json' --project '${WANDB_PROJECT}' --base-dir '${BASE_DIR}'"
    if [ -n "${WANDB_ENTITY}" ]; then
      WB_ARGS="${WB_ARGS} --entity '${WANDB_ENTITY}'"
    fi
    if [ -n "${WANDB_RUN_NAME}" ]; then
      WB_ARGS="${WB_ARGS} --run-name '${WANDB_RUN_NAME}'"
    fi
    if [ -n "${WANDB_GROUP}" ]; then
      WB_ARGS="${WB_ARGS} --group '${WANDB_GROUP}'"
    fi
    if [ -n "${WANDB_TAGS}" ]; then
      WB_ARGS="${WB_ARGS} --tags '${WANDB_TAGS}'"
    fi
    run_cmd "${SRI_ENV}" \
      "python -m sri.learning_biases_bridge.log_wandb_summary ${WB_ARGS}"
  fi
fi

echo "Done. Summary at ${BASE_DIR}/summary.{json,csv}"
