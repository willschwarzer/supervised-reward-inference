#!/bin/bash

# Launch the first gen job

GEN_SCRIPT_1="scripts/rl/16cpu_array_generate_reach_avoid_obj.sh"
NUM_JOBS_GEN_1=30

gen_job1_id=$(sbatch --array=0-$(($NUM_JOBS_GEN_1-1))%15 $GEN_SCRIPT_1 | awk '{print $4}')

SRI_SCRIPT="scripts/experiments/run_pickplace_005_02_no_inf.sh"
NUM_JOBS_SRI=30

# Launch the SRI job with dependencies on the gen jobs
# sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 <<EOF
sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 --dependency=afterok:$gen_job1_id <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

bash $SRI_SCRIPT
EOF
)

# Extract the job ID of the SRI job
sri_job_id=$(echo $sri_job_id | awk '{print $4}')

RL_SCRIPT="scripts/experiments/run_pickplace_rl_005_02.sh"
NUM_JOBS_RL=$NUM_JOBS_SRI
# NUM_JOBS_RL=90
# Launch the second job with a dependency on the SRI job
sbatch --array=0-$(($NUM_JOBS_RL-1))%30 --dependency=afterok:$sri_job_id <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

bash $RL_SCRIPT
EOF
