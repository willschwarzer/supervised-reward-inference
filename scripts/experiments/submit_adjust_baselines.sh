#!/bin/bash

NUM_JOBS_ADV=360
# ADV_SCRIPT="scripts/experiments/run_adjust_adv.sh"
# adv_job_id=$(sbatch --array=0-$(($NUM_JOBS_ADV-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP
# #SBATCH --gres=gpu:1
# #SBATCH --time=0-18:00:00
# #SBATCH --partition=gpu
# #SBATCH --constraint=sm_80
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL

# bash $ADV_SCRIPT
# EOF
# )

# # Extract the job ID of the adv job
# adv_job_id=$(echo $adv_job_id | awk '{print $4}')

NUM_JOBS_BC=180

# BC_SCRIPT="scripts/experiments/run_adjust_bc.sh"

# bc_job_id=$(sbatch --array=0-$(($NUM_JOBS_BC-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP
# #SBATCH --time=0-08:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL

# bash $BC_SCRIPT
# EOF
# )

# # Extract the job ID of the BC job
# bc_job_id=$(echo $bc_job_id | awk '{print $4}')

NUM_JOBS_EVAL=$(($NUM_JOBS_ADV + $NUM_JOBS_BC))

EVAL_SCRIPT="scripts/experiments/run_adjust_eval_im.sh"

# Launch the second job with a dependency on the SRI job
# sbatch --array=0-$(($NUM_JOBS_EVAL-1))%30 --dependency=afterok:$adv_job_id:$bc_job_id <<EOF
sbatch --array=0-$(($NUM_JOBS_EVAL-1))%30 <<EOF
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

bash $EVAL_SCRIPT
EOF
