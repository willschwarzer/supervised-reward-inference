#!/bin/bash

# IDX=${SLURM_ARRAY_TASK_ID:-0}
# echo "IDX: $IDX"
# NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # expect idx to be 0-10
# echo "NOISE_COEFF: $NOISE_COEFF"

IDX=${SLURM_ARRAY_TASK_ID:-0}
# IDX=5
echo "IDX: $IDX"

DATASET_IDX=$IDX
MODEL_IDX=$IDX
NOISE_COEFF=0.0

echo "NOISE_COEFF: $NOISE_COEFF"
echo "DATASET_IDX: $DATASET_IDX"

python -m sri.run_experiment \
    --general-config general/default_experiment.yml \
    --train-config train/skip.yml \
    --dataset-config datasets/reach_noise.yml \
    --obs-dataset-config datasets/reach_noise.yml \
    --orl-dataset-config datasets/reach_noise.yml \
    --model-config model/load_default.yml \
    --rl-config rl/tqc_quarter_legacy_paper_esr_5m_gt_reward.yml \
    --inference-config inference/default_only_rl.yml \
    --dataset-idx 0 \
    --model-idxs 0



    # --train-config train/random_obs_selection_even_more_obs.yml \