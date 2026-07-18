#!/bin/bash

# Parse command line arguments
BATCH_TYPE=${1:-0}  # Default to 0 (all batches) if not provided

# Validate batch type argument
if [[ ! $BATCH_TYPE =~ ^[0-3]$ ]]; then
    echo "Error: Batch type must be a number between 0-3"
    echo "Usage: $0 [BATCH_TYPE]"
    echo "  0: Run all batches (default)"
    echo "  1: Run only SRI batch"
    echo "  2: Run only SGI batch"
    echo "  3: Run only SAI batch"
    exit 1
fi

MODEL_ARGS="--model-args dem_encoder_type=transformer"

SRI_SCRIPT="scripts/experiments/run_n_rfpp_no_inf.sh"
NUM_JOBS_SRI=90
# NUM_JOBS_SRI=3

# Only run SRI batch if batch type is 0 (all) or 1 (SRI only)
if [[ $BATCH_TYPE == 0 || $BATCH_TYPE == 1 ]]; then
    # Launch the SRI job with dependencies on the gen jobs
    # sri_job_id=$(sbatch --array=0-$(($NUM_JOBS_SRI-1))%30 <<EOF
    sri_job_id=$(sbatch <<EOF
#!/bin/bash
#SBATCH --array=2,17,38,41,44,47,50,56,62,65,68,71,74,77,80,83,86,88,89
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80,vram40
#SBATCH --exclude=gpu031
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

bash $SRI_SCRIPT $MODEL_ARGS
EOF
    )

    # Extract the job ID of the SRI job
    sri_job_id=$(echo $sri_job_id | awk '{print $4}')

    RL_SCRIPT="scripts/experiments/run_n_rfpp_rl.sh"
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
#SBATCH --exclude=gpu031
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL

bash $RL_SCRIPT $MODEL_ARGS
EOF
    )

    rl_job_id=$(echo $rl_job_id | awk '{print $4}')

    EVAL_SCRIPT="scripts/experiments/run_n_eval_rl.sh"
    NUM_JOBS_EVAL=$NUM_JOBS_RL
    # Launch the third job with a dependency on the second job
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
bash $EVAL_SCRIPT $MODEL_ARGS
EOF
fi

# Only run SGI batch if batch type is 0 (all) or 2 (SGI only)
if [[ $BATCH_TYPE == 0 || $BATCH_TYPE == 2 ]]; then
    # SGI
    SGI_SCRIPT="scripts/experiments/run_n_rfpp_bidg.sh"
    NUM_JOBS_SGI=$NUM_JOBS_SRI
    sgi_job_id=$(sbatch --array=0-$(($NUM_JOBS_SGI-1))%30 <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --exclude=gpu031
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL
bash $SGI_SCRIPT $MODEL_ARGS
EOF
    )
    sgi_job_id=$(echo $sgi_job_id | awk '{print $4}')

    SGI_EVAL_SCRIPT="scripts/experiments/run_n_rfpp_eval_bidg.sh"
    NUM_JOBS_SGI_EVAL=$NUM_JOBS_SGI
    # Launch the second job with a dependency on the SGI job
    sbatch --array=0-$(($NUM_JOBS_SGI_EVAL-1))%30 --dependency=afterok:$sgi_job_id <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=0-18:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL
bash $SGI_EVAL_SCRIPT $MODEL_ARGS
EOF
fi

# Only run SAI batch if batch type is 0 (all) or 3 (SAI only)
if [[ $BATCH_TYPE == 0 || $BATCH_TYPE == 3 ]]; then
    # SAI
    SAI_SCRIPT="scripts/experiments/run_n_rfpp_bida.sh"
    NUM_JOBS_SAI=$NUM_JOBS_SGI
    sai_job_id=$(sbatch --array=0-$(($NUM_JOBS_SAI-1))%30 <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-18:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --exclude=gpu031
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL
bash $SAI_SCRIPT $MODEL_ARGS
EOF
    )
    sai_job_id=$(echo $sai_job_id | awk '{print $4}')
    SAI_EVAL_SCRIPT="scripts/experiments/run_n_rfpp_eval_bida.sh"
    NUM_JOBS_SAI_EVAL=$NUM_JOBS_SAI
    # Launch the second job with a dependency on the SAI job
    sbatch --array=0-$(($NUM_JOBS_SAI_EVAL-1))%30 --dependency=afterok:$sai_job_id <<EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --time=0-18:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL
bash $SAI_EVAL_SCRIPT $MODEL_ARGS
EOF
fi