# #!/bin/bash

# NUM_JOBS_REACH=30
# REACH_SCRIPT="scripts/experiments/run_reach_rl_gt.sh"
# reach_job_id=$(sbatch --array=0-$(($NUM_JOBS_REACH-1))%30 <<EOF
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

# bash $REACH_SCRIPT
# EOF
# )

# reach_job_id=$(echo $reach_job_id | awk '{print $4}')

# NUM_JOBS_PICKPLACE=30

# PICKPLACE_SCRIPT="scripts/experiments/run_pickplace_rl_gt.sh"

# pickplace_job_id=$(sbatch --array=0-$(($NUM_JOBS_PICKPLACE-1))%30 <<EOF
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

# bash $PICKPLACE_SCRIPT
# EOF
# )

# pickplace_job_id=$(echo $pickplace_job_id | awk '{print $4}')

# NUM_JOBS_REACH_EVAL=$NUM_JOBS_REACH
NUM_JOBS_REACH_EVAL=30

REACH_EVAL_SCRIPT="scripts/experiments/run_reach_eval_gt.sh"

# Launch the second job with a dependency on the SRI job
# sbatch --array=0-$(($NUM_JOBS_REACH_EVAL-1))%30 --dependency=afterok:$reach_job_id <<EOF
sbatch --array=0-$(($NUM_JOBS_REACH_EVAL-1))%30 <<EOF
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

bash $REACH_EVAL_SCRIPT
EOF

# NUM_JOBS_PICKPLACE_EVAL=$NUM_JOBS_PICKPLACE
NUM_JOBS_PICKPLACE_EVAL=30

PICKPLACE_EVAL_SCRIPT="scripts/experiments/run_pickplace_eval_gt.sh"

# Launch the second job with a dependency on the SRI job
# sbatch --array=0-$(($NUM_JOBS_PICKPLACE_EVAL-1))%30 --dependency=afterok:$pickplace_job_id <<EOF
sbatch --array=0-$(($NUM_JOBS_PICKPLACE_EVAL-1))%30 <<EOF
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

bash $PICKPLACE_EVAL_SCRIPT
EOF