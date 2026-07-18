#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
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

export BASE_DIR="${BASE_DIR:-artifacts/learning_biases_bridge_sanity_alignment}"
export CLEAN_DIR="${CLEAN_DIR:-${BASE_DIR}/clean}"
export SHUF_DIR="${SHUF_DIR:-${BASE_DIR}/shuffled}"
export RUNS_DIR="${RUNS_DIR:-${BASE_DIR}/runs}"

export AGENT="${AGENT:-optimal}"
export DATASET_SEED="${DATASET_SEED:-17}"
export TRAIN_SIZE="${TRAIN_SIZE:-400}"
export VAL_SIZE="${VAL_SIZE:-200}"
export INFER_SIZE="${INFER_SIZE:-100}"
export MAX_GEN_RETRIES="${MAX_GEN_RETRIES:-20}"
export DATA_BETA="${DATA_BETA:-1.0}"
export DATA_NOISE="${DATA_NOISE:-0.2}"
export FORCE_REEXPORT="${FORCE_REEXPORT:-0}"

export SRI_EPOCHS="${SRI_EPOCHS:-40}"
export SRI_BATCH_SIZE="${SRI_BATCH_SIZE:-128}"
export SRI_EVAL_BATCH_SIZE="${SRI_EVAL_BATCH_SIZE:-256}"
export SRI_MODEL_SEED="${SRI_MODEL_SEED:-0}"
export SRI_ENCODER="${SRI_ENCODER:-transformer}"

export SHUFFLE_SEED="${SHUFFLE_SEED:-123}"

echo "== Alignment sanity sweep =="
echo "BASE_DIR=${BASE_DIR}"
echo "Agent=${AGENT} beta=${DATA_BETA} noise=${DATA_NOISE} dataset_seed=${DATASET_SEED} sizes=${TRAIN_SIZE}/${VAL_SIZE}/${INFER_SIZE}"
echo "SRI encoder=${SRI_ENCODER} epochs=${SRI_EPOCHS}"

manifest_path="${CLEAN_DIR}/manifest.json"
rebuild_clean=1
if [ -f "${manifest_path}" ]; then
  if [ "${FORCE_REEXPORT}" = "1" ]; then
    echo "[1/6] FORCE_REEXPORT=1, removing existing clean dataset at ${CLEAN_DIR}"
    rm -rf "${CLEAN_DIR}"
  elif MANIFEST_PATH="${manifest_path}" \
    AGENT="${AGENT}" \
    DATASET_SEED="${DATASET_SEED}" \
    TRAIN_SIZE="${TRAIN_SIZE}" \
    VAL_SIZE="${VAL_SIZE}" \
    INFER_SIZE="${INFER_SIZE}" \
    MAX_GEN_RETRIES="${MAX_GEN_RETRIES}" \
    DATA_BETA="${DATA_BETA}" \
    DATA_NOISE="${DATA_NOISE}" \
    python3 -c '
import json
import math
import os
import sys

path = os.environ["MANIFEST_PATH"]
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
cfg = payload.get("config", {})

checks = [
    ("agent", str(cfg.get("agent")), os.environ["AGENT"]),
    ("seed", int(cfg.get("seed")), int(os.environ["DATASET_SEED"])),
    ("planner_train_size", int(cfg.get("planner_train_size")), int(os.environ["TRAIN_SIZE"])),
    ("planner_val_size", int(cfg.get("planner_val_size")), int(os.environ["VAL_SIZE"])),
    ("reward_infer_size", int(cfg.get("reward_infer_size")), int(os.environ["INFER_SIZE"])),
    ("max_generation_retries", int(cfg.get("max_generation_retries")), int(os.environ["MAX_GEN_RETRIES"])),
]

for name, actual, expected in checks:
    if actual != expected:
        print(f"[1/6] Existing dataset mismatch for {name}: {actual} != {expected}")
        sys.exit(1)

float_checks = [
    ("beta", float(cfg.get("beta")), float(os.environ["DATA_BETA"])),
    ("noise", float(cfg.get("noise")), float(os.environ["DATA_NOISE"])),
]
for name, actual, expected in float_checks:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        print(f"[1/6] Existing dataset mismatch for {name}: {actual} != {expected}")
        sys.exit(1)

print("[1/6] Existing clean dataset matches requested generation settings.")
sys.exit(0)
'; then
    rebuild_clean=0
  else
    echo "[1/6] Existing clean dataset settings differ; rebuilding ${CLEAN_DIR}"
    rm -rf "${CLEAN_DIR}"
  fi
fi

if [ "${rebuild_clean}" = "1" ]; then
  echo "[1/6] Exporting clean dataset"
  conda run -n "${LB_ENV}" python -m learning_biases.bridge_export_dataset \
    --agent "${AGENT}" \
    --seed "${DATASET_SEED}" \
    --beta "${DATA_BETA}" \
    --noise "${DATA_NOISE}" \
    --planner-train-size "${TRAIN_SIZE}" \
    --planner-val-size "${VAL_SIZE}" \
    --reward-infer-size "${INFER_SIZE}" \
    --max-generation-retries "${MAX_GEN_RETRIES}" \
    --out "${CLEAN_DIR}"
else
  echo "[1/6] Reusing existing clean dataset at ${CLEAN_DIR}"
fi

echo "[2/6] Building shuffled-label variant (train/val rewards only)"
rm -rf \
  "${SHUF_DIR}" \
  "${RUNS_DIR}/aligned" \
  "${RUNS_DIR}/shuffled" \
  "${RUNS_DIR}/aligned_eval" \
  "${RUNS_DIR}/shuffled_eval" \
  "${BASE_DIR}/alignment_check.json"
mkdir -p "${RUNS_DIR}"
conda run -n "${SRI_ENV}" python -m sri.learning_biases_bridge.build_shuffled_dataset \
  --clean-dir "${CLEAN_DIR}" \
  --out-dir "${SHUF_DIR}" \
  --shuffle-seed "${SHUFFLE_SEED}"

echo "[3/6] Train aligned-label model"
conda run -n "${SRI_ENV}" python -m sri.learning_biases_bridge.train_sri_policy \
  --dataset "${CLEAN_DIR}" \
  --out "${RUNS_DIR}/aligned" \
  --dem-encoder-type "${SRI_ENCODER}" \
  --epochs "${SRI_EPOCHS}" \
  --batch-size "${SRI_BATCH_SIZE}" \
  --eval-batch-size "${SRI_EVAL_BATCH_SIZE}" \
  --seed "${SRI_MODEL_SEED}"

echo "[4/6] Train shuffled-label model"
conda run -n "${SRI_ENV}" python -m sri.learning_biases_bridge.train_sri_policy \
  --dataset "${SHUF_DIR}" \
  --out "${RUNS_DIR}/shuffled" \
  --dem-encoder-type "${SRI_ENCODER}" \
  --epochs "${SRI_EPOCHS}" \
  --batch-size "${SRI_BATCH_SIZE}" \
  --eval-batch-size "${SRI_EVAL_BATCH_SIZE}" \
  --seed "${SRI_MODEL_SEED}"

echo "[5/6] Evaluate both models on reward_infer"
# Uses legacy learning_biases evaluator, which depends on TensorFlow in LB_ENV.
conda run -n "${LB_ENV}" python -m sri.learning_biases_bridge.evaluate_sri_policy \
  --dataset "${CLEAN_DIR}" \
  --pred "${RUNS_DIR}/aligned/sri_pred_reward_vec.npy" \
  --out "${RUNS_DIR}/aligned_eval"

conda run -n "${LB_ENV}" python -m sri.learning_biases_bridge.evaluate_sri_policy \
  --dataset "${SHUF_DIR}" \
  --pred "${RUNS_DIR}/shuffled/sri_pred_reward_vec.npy" \
  --out "${RUNS_DIR}/shuffled_eval"

echo "[6/6] Write summary"
conda run -n "${SRI_ENV}" python -m sri.learning_biases_bridge.write_alignment_summary \
  --runs-dir "${RUNS_DIR}" \
  --dataset-dir "${CLEAN_DIR}" \
  --out "${BASE_DIR}/alignment_check.json"

echo "Alignment sanity sweep complete."
