#!/bin/bash
#
# run_all.sh
# Usage: run_all.sh [switch]
#   switch: comma-separated list of blocks to run
#     1 = PPData
#     2 = PickPlace
#     3 = Data
#     4 = N
#     5 = Adjust
#     6 = Noise
#   default (no arg): 1,2,3,4,5,6

SWITCH=${1:-"1,2,3,4,5,6"}

# helper to turn "1,2,3" into an array
IFS=',' read -r -a BLOCKS <<< "$SWITCH"

for blk in "${BLOCKS[@]}"; do
  case "$blk" in

    1)
      PPDATA_SCRIPT="scripts/experiments/run_ppdata_bida.sh"
      PPDATA_NUM_JOBS=270
      sbatch --array=0-$(($PPDATA_NUM_JOBS-1))%30 \
             --dependency=afterok:34056127:34056128 <<EOF
#!/bin/bash
#SBATCH -J ppdata_bida
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $PPDATA_SCRIPT
EOF
      ;;

    2)
      PICKPLACE_SCRIPT="scripts/experiments/run_pickplace_bida.sh"
      PICKPLACE_NUM_JOBS=30
      # sbatch --array=0-$(($PICKPLACE_NUM_JOBS-1))%30 \
            #  --dependency=afterok:34056127:34056128 <<EOF
      sbatch <<EOF
#!/bin/bash
#SBATCH -J pickplace_bida
#SBATCH --dependency=afterok:34183552 
#SBATCH --array=0-9
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $PICKPLACE_SCRIPT
EOF
      ;;

    3)
      DATA_SCRIPT="scripts/experiments/run_data_rfpp_bida.sh"
      DATA_NUM_JOBS=270
    #   sbatch --array=0-$(($DATA_NUM_JOBS-1))%30 <<EOF
    sbatch <<EOF
#!/bin/bash
#SBATCH -J data_bida
##SBATCH --dependency=afterok:34107372
#SBATCH --array=0-80
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $DATA_SCRIPT
EOF
      ;;

    4)
      N_SCRIPT="scripts/experiments/run_n_rfpp_bida.sh"
      N_NUM_JOBS=90
      sbatch --array=0-$(($N_NUM_JOBS-1))%30 <<EOF
#!/bin/bash
#SBATCH -J n_bida
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $N_SCRIPT
EOF
      ;;

    5)
      ADJUST_SCRIPT="scripts/experiments/run_adjust_rfpp_bida.sh"
      sbatch <<EOF
#!/bin/bash
#SBATCH -J adjust_bida
##SBATCH --array=1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,49,50,52,53,55,56,57
#SBATCH --array=0-79
##SBATCH --dependency=afterok:34107287
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $ADJUST_SCRIPT
EOF
      ;;

    6)
      NOISE_SCRIPT="scripts/experiments/run_noise_rfpp_bida.sh"
      NOISE_NUM_JOBS=210
      sbatch --array=0-$(($NOISE_NUM_JOBS-1))%30 <<EOF
#!/bin/bash
#SBATCH -J noise_bida
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -A YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=sm_80
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=FAIL,TIME_LIMIT
bash $NOISE_SCRIPT
EOF
      ;;

    *)
      echo "Warning: unknown block \"$blk\"; skipping."
      ;;
  esac
done
echo "All jobs submitted."