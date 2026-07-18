from sb3_contrib import TQC
from stable_baselines3 import SAC, PPO
# from stable_baselines3.common.evaluation import evaluate_policy

from train import make_env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
# no need to import hidden, you can just set env._partially_observable = True
from metaworld.policies.sawyer_push_v2_policy import SawyerPushV2Policy
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
import pickle
import numpy as np
import argparse
import os
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import imageio
import wandb

def inference(goal_model, dataloader, policy, args):
    # maybe at some point this will also take a GCRL model and do inference with that
    # but for now we're just doing oracle inference

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.inference_env]
    if args.inference_batch_size == 1:
        env = DummyVecEnv([make_env(env_cls, args)])
        env = VecMonitor(env)
    else:
        env = SubprocVecEnv([make_env(env_cls, args) for _ in range(args.inference_batch_size)])
        env = VecMonitor(env)
        print(f"Using SubprocVecEnv with {env.num_envs} environments")

    # set environment horizon
    env.set_attr("max_path_length", args.horizon)

    if policy is not None:
        raise NotImplementedError("RL inference not implemented yet")
    else:
        # this means we're using explicit control
        model_class = {"reach-v2-goal-observable": SawyerReachV2Policy, 
                    "push-v2-goal-observable": SawyerPushV2Policy, 
                    "pick-place-v2-goal-observable": SawyerPickPlaceV2Policy}.get(args.env)
        model = model_class() # no args, we're just doing the ground-truth task


    succeded_at_end = np.zeros(len(dataloader), args.inference_batch_size, args.inference_episodes, dtype=np.bool)
    succeded_at_any_point = np.zeros(len(dataloader), args.inference_batch_size, args.inference_episodes, dtype=np.bool)

    env.set_attr("_freeze_rand_vec", False) # need new object positions for each rollout

    with tqdm(dataloader, unit="batch") as tepoch:
        for dem_batch, _, goal_batch in tepoch:
            obs = env.reset()
            pred_goals = goal_model.predict(dem_batch)
            for ep in tqdm(range(0, args.num_rollouts_per_task)):
                # set true goals
                for env_idx in range(env.num_envs):
                    env.env_method("_set_target_pos", goal_batch[env_idx], indices=[env_idx])
                # set observed (predicted) goals
                obs[:, -3:] = pred_goals
                for step in range(args.horizon):
                    if policy is not None:
                        raise NotImplementedError("RL inference not implemented yet")
                        action, _ = policy.predict(obs, deterministic=True)
                    else:
                        oracle_actions = []
                        for env_idx in range(env.num_envs):
                            oracle_actions.append(model.get_action(obs[env_idx], step))
                        action = np.array(oracle_actions)
                    next_obs, _, dones, infos = env.step(action)
                    # next_obs[:, -3:] = pred_goals # ugh I am the worst programmer to ever program in the history of programming
                    for env_idx in range(env.num_envs):
                        succeded_at_any_point[env_idx, ep] = succeded_at_any_point[env_idx, ep] or infos[env_idx]["success"]
                    if step < args.horizon - 1:
                        assert not dones.any(), "Environment should not be done before max_steps"
                    elif step == args.horizon - 1:
                        assert dones.all(), "Environment should be done at max_steps"
                        for env_idx in range(env.num_envs):
                            succeded_at_end[env_idx, ep] = infos[env_idx]["success"]
                    obs = next_obs
                # want to know which goals failed
                # for env_idx in range(env.num_envs):
                #     if not succeded_at_end[env_idx, ep]:
                #         print(f"Goal {goal_batch[env_idx]} failed")
                #         imageio.imsave(f"runs/{wandb.run.name}/goal_{goal_batch[env_idx]}_failed.png", env.render(env_idx))
                #     else:
                #         print(f"Goal {goal_batch[env_idx]} succeeded")
                #         imageio.imsave(f"runs/{wandb.run.name}/goal_{goal_batch[env_idx]}_succeeded.png", env.render(env_idx))
    return succeded_at_end, succeded_at_any_point