#!/bin/bash
#SBATCH -n 1
#SBATCH -A YOUR_ACCOUNT
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH --time=0-24:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --array=1-3
#SBAT --mail-type=END,FAIL

# conda activate rings
bash scripts/rl/train.sh
