#!/bin/bash
python -m sri.rl.generate \
    --dataset-config datasets/default_reach.yml \
    --general-config general/default_generation.yml \
    --load-goals-from-datasets-with-config datasets/default_reach.yml \
    --dataset-idx 4 \
    # --num-tasks-per-env 100 \
    # --num-rollouts-per-task 100 \
    # --env pick-place \
    # --num-envs 16 \
    # --goal-x-bounds -0.3 0.3 \
    # --goal-y-bounds 0.3 0.7 \
    # --goal-z-bounds 0.05 0.3 \
    # --horizon 250 \
    # --extra-reward-info \
    # --hand-speed 30 \
    # --render \
    # --random-hand-starts \
    # --render \
    # horizon of 300 is appropriate for pick-place
    # --render \
    # --circle-radius 0.1 \
    # --circle-around \
    # --mirror-goal \
    # --wandb-run-name resilient-cloud-79 \
    # --wandb-run-name lucky-leaf-116 \
    # --chai-rollouts \
    # --gripped-start \
    # --wandb-run-name brisk-dawn-24 \