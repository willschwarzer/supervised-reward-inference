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
#SBAT --mail-type=END,FAIL
#SBATCH --array=0-59

bash scripts/rl/generate_adjusts_reach.sh
