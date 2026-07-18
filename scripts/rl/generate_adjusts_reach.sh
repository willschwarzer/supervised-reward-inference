# #!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-5}
echo "IDX: $IDX"

ADJUSTMENTS=(1.0 0.6 0.2 -0.2 -0.6 -1.0)
ADJUSTMENTS_IDX=$((IDX % ${#ADJUSTMENTS[@]}))
ADJUSTMENT=${ADJUSTMENTS[$ADJUSTMENTS_IDX]}

echo "ADJUSTMENT: $ADJUSTMENT"

python -m sri.rl.generate \
    --dataset-config datasets/reach_goal_pos_adjustment.yml \
    --general-config general/default_generation.yml \
    --noise-coeff 0.0 \
    --goal-pos-adjustment-factor $ADJUSTMENT

# IDX=$1
# echo "IDX: $IDX"
# # NOISE_COEFF=$(echo "$IDX * 0.1" | bc)
# # # expect idx to be 0-10
# # echo "NOISE_COEFF: $NOISE_COEFF"
# NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)


# python -m sri.rl.generate \
#     --dataset-config datasets/reach_noise.yml \
#     --general-config general/default_generation.yml \
#     --noise-coeff $NOISE_COEFF \
