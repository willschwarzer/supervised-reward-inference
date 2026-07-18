# #!/bin/bash

# NUM_JOBS_NOISE=210
# # noise, adjust, n, data, pickplace, ppdata
# # Launch the SRI job with dependencies on the gen jobs
# NOISE_EVAL_SCRIPT="scripts/experiments/run_noise_eval_rl.sh"
# # sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 <<EOF
# noise_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_NOISE-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP

# #SBATCH --time=0-18:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL

# bash $NOISE_EVAL_SCRIPT
# EOF
# )

# # Extract the job ID of the noise eval job, for no particular reason ^^
# noise_eval_job_id=$(echo $sri_job_id | awk '{print $4}')

# NUM_JOBS_ADJUST=180
# ADJUST_EVAL_SCRIPT="scripts/experiments/run_adjust_eval_rl.sh"

# adjust_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_ADJUST-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP

# #SBATCH --time=0-18:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL
# bash $ADJUST_EVAL_SCRIPT
# EOF
# )

# # Extract the job ID of the adjust eval job for no particular reason ^^
# adjust_eval_job_id=$(echo $sri_job_id | awk '{print $4}')

NUM_JOBS_N=90
N_EVAL_SCRIPT="scripts/experiments/run_n_eval_rl.sh"

n_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_N-1))%30 <<EOF
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
bash $N_EVAL_SCRIPT
EOF
)

# Extract the job ID of the n eval job for no particular reason ^^
n_eval_job_id=$(echo $sri_job_id | awk '{print $4}')

# NUM_JOBS_DATA=270
# DATA_EVAL_SCRIPT="scripts/experiments/run_data_eval_rl.sh"

# data_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_DATA-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP

# #SBATCH --time=0-18:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL
# bash $DATA_EVAL_SCRIPT
# EOF
# )

# # Extract the job ID of the data eval job for no particular reason ^^
# data_eval_job_id=$(echo $sri_job_id | awk '{print $4}')

# NUM_JOBS_PICKPLACE=30
# PICKPLACE_EVAL_SCRIPT="scripts/experiments/run_pickplace_eval_rl.sh"

# pickplace_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_PICKPLACE-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP

# #SBATCH --time=0-18:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL
# bash $PICKPLACE_EVAL_SCRIPT
# EOF
# )

# # Extract the job ID of the pickplace eval job for no particular reason ^^
# pickplace_eval_job_id=$(echo $sri_job_id | awk '{print $4}')

# NUM_JOBS_PPDATA=270
# PPDATA_EVAL_SCRIPT="scripts/experiments/run_ppdata_eval_rl.sh"

# ppdata_eval_job_id=$(sbatch --array=0-$(($NUM_JOBS_PPDATA-1))%30 <<EOF
# #!/bin/bash
# #SBATCH -n 1
# #SBATCH -c 8
# #SBATCH --mem=40G
# #SBATCH -A YOUR_ACCOUNT_GROUP

# #SBATCH --time=0-18:00:00
# #SBATCH --partition=cpu
# #SBATCH --output=/home/submission_user/work/logs/paper/n/%A_%a.out
# #SBATCH --error=/home/submission_user/work/logs/paper/n/%A_%a.err
# #SBATCH --mail-user=user@example.com
# #SBATCH --mail-type=FAIL
# bash $PPDATA_EVAL_SCRIPT
# EOF
# )

# # Extract the job ID of the ppdata eval job for no particular reason ^^
# ppdata_eval_job_id=$(echo $sri_job_id | awk '{print $4}')
