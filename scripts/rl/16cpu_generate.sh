#!/bin/bash
#SBATCH -n 1
#SBATCH -A YOUR_ACCOUNT
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBAT --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBAT --mail-type=END,FAIL

# conda activate rings
# bash scripts/rl/generate.sh
bash scripts/rl/generate_pickplace_1000.sh
# bash scripts/rl/generate_reach_avoid_obj_1000.sh
