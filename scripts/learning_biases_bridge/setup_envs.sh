#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally conservative and can be edited per-cluster.
# It creates:
#   1) meta-world       (Torch/SRI env)
#   2) learning-biases-tf1 (TF1 env for Shah baseline)

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load conda/latest >/dev/null 2>&1 || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please load conda and rerun."
  exit 1
fi

conda create -y -n meta-world python=3.10
conda run -n meta-world pip install -r requirements.txt

conda create -y -n learning-biases-tf1 python=3.8
conda run -n learning-biases-tf1 pip install -r learning_biases/requirements.txt

echo "Created envs: meta-world, learning-biases-tf1"
echo "If you already have pemirl_tf1 with working TF1, you can use LB_ENV=pemirl_tf1 in run_matrix.sh"
