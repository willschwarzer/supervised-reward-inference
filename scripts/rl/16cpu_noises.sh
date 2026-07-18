#!/bin/bash
#SBATCH -n 1
#SBATCH -A YOUR_ACCOUNT
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATC --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBAT --mail-type=END,FAIL
#SBATCH --array=0-10

# conda activate rings
bash scripts/rl/generate_noises_reach.sh $SLURM_ARRAY_TASK_ID
