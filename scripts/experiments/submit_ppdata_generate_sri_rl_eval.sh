#!/bin/bash

# GEN_SCRIPT_1="scripts/rl/16cpu_array_generate_pickplace.sh"
NUM_JOBS_GEN_1=20

# # Launch the first gen job

# gen_job1_id=$(sbatch --array=0-$(($NUM_JOBS_GEN_1-1))%10 $GEN_SCRIPT_1 | awk '{print $4}')

# GEN_SCRIPT_2="scripts/rl/16cpu_array_generate_reach_circling_avoid_obj.sh"
# NUM_JOBS_GEN_2=$NUM_JOBS_GEN_1

# # Launch the second gen job
# gen_job2_id=$(sbatch --array=0-$(($NUM_JOBS_GEN_2-1))%10 $GEN_SCRIPT_2 | awk '{print $4}')

# GEN_SCRIPT_3="scripts/rl/16cpu_array_generate_reach_for_pickplace.sh"
# NUM_JOBS_GEN_3=$NUM_JOBS_GEN_1

# # Launch the third gen job
# gen_job3_id=$(sbatch --array=0-$(($NUM_JOBS_GEN_3-1))%10 $GEN_SCRIPT_3 | awk '{print $4}')

# SRI_SCRIPT="scripts/experiments/run_02_ppdata_no_inf.sh"
SRI_SCRIPT="scripts/experiments/run_ppdata_005_02_no_inf.sh"
# NUM_JOBS_SRI=270
NUM_JOBS_SRI=$(($NUM_JOBS_GEN_1*9))

# Launch the SRI job with dependencies on the gen jobs
# sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 --dependency=afterok:$gen_job1_id:$gen_job2_id:$gen_job3_id <<EOF
sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 <<EOF
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

RL_SCRIPT="scripts/experiments/run_ppdata_rl_005_02.sh"
NUM_JOBS_RL=$NUM_JOBS_SRI
# Launch the second job with a dependency on the SRI job
rl_job_id=$(sbatch --array=0-$(($NUM_JOBS_RL-1))%30 --dependency=afterok:$sri_job_id <<EOF
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
)

# Extract the job ID of the RL job
rl_job_id=$(echo $rl_job_id | awk '{print $4}')

NUM_JOBS_EVAL=$NUM_JOBS_RL

EVAL_SCRIPT="scripts/experiments/run_ppdata_eval_rl.sh"
sbatch --array=0-$(($NUM_JOBS_EVAL-1))%30 --dependency=afterok:$rl_job_id <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=0-18:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL
bash $EVAL_SCRIPT
EOF