#!/bin/bash
#SBATCH -n 1
#SBATCH -A YOUR_ACCOUNT
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATC --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-29

python -m sri.rl.generate \
    --dataset-config datasets/default_pickplace.yml \
    --general-config general/default_generation.yml \
