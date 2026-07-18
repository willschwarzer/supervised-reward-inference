# #!/bin/bash

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "IDX: $IDX"

NOISE_COEFFS=(0.0 0.35 0.60 0.76 0.87 0.95 1.0)
NOISE_COEFF_IDX=$((IDX % ${#NOISE_COEFFS[@]}))
NOISE_COEFF=${NOISE_COEFFS[$NOISE_COEFF_IDX]}

echo "NOISE_COEFF: $NOISE_COEFF"

python -m sri.rl.generate \
    --dataset-config datasets/reach_noise.yml \
    --general-config general/default_generation.yml \
    --noise-coeff $NOISE_COEFF

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
