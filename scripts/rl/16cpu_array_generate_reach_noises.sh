#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATC --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH -A YOUR_ACCOUNT
#SBAT --mail-type=END,FAIL
#SBATCH --array=0-209

bash scripts/rl/generate_noises_reach.sh