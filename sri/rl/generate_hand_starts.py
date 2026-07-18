from sb3_contrib import TQC
from stable_baselines3 import SAC, PPO

# from stable_baselines3.common.evaluation import evaluate_policy

from sri.rl.train import make_env
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
from imitation.data import rollout

# from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import serialize
from imitation.data.types import Trajectory
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import imageio
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-positions-per-dim",
        type=int,
        default=10,
        help="Number of positions to sample per dimension, resulting in n^3 total saves. Default: 10",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel. Default: 1",
    )
    parser.add_argument(
        "--goal-x-bounds",
        nargs=2,
        type=float,
        default=[-0.1, 0.1],
        help="Bounds for the x-coordinate of the goal. Default is [-0.1, 0.1]. \
                            We do not recommend changing this beyond -0.65 to 0.65, as that appears to be the table boundary.",
    )
    parser.add_argument(
        "--goal-y-bounds",
        nargs=2,
        type=float,
        default=[0.8, 0.9],
        help="Bounds for the y-coordinate of the goal. Default is [0.8, 0.9]. \
                            We do not recommend changing this beyond 0.25 to 0.95, as that appears to be the table boundary.",
    )
    parser.add_argument(
        "--goal-z-bounds",
        nargs=2,
        type=float,
        default=[0.1, 0.4],
        help="Bounds for the z-coordinate of the goal. Default is [0.1, 0.4].",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether or not to render the environment. Default: False",
    )
    args = parser.parse_args()
    if args.render:
        print(
            "###########################################\n\
              WARNING: RENDERING IS ON. THIS WILL BE SLOW.\n\
              ############################################"
        )
    args.env = "reach-v2-goal-observable"
    # args.reinit_goals = False # need to add this for making envs
    # args.record_grips = False # ditto
    # args.gripped_start = False # ditto
    args.horizon = 1000
    if args.goal_z_bounds[0] < 0.07:
        print(
            "Warning: lower bound for z-coordinate of goal is less than 0.07, which may cause unreachable goals."
        )
    # we reinit goals every rollouts_per_task rollouts though
    assert (
        args.num_positions_per_dim**3 % args.num_envs == 0
    ), "Total number of hand positions must be divisible by num_envs"
    return args


def main():
    args = parse_args()

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.env]
    if args.num_envs == 1:
        # env = make_env(env_cls, args)()
        # env = DummyVecEnv([lambda: env])
        env = DummyVecEnv([make_env(env_cls, args)])
        env = VecMonitor(env)
    else:
        env = SubprocVecEnv([make_env(env_cls, args) for _ in range(args.num_envs)])
        env = VecMonitor(env)
        print(f"Using SubprocVecEnv with {env.num_envs} environments")

    # this means we're using explicit control
    model = SawyerReachV2Policy()
    # print(env_args)

    # initialize wandb
    wandb.init(project="metaworld_hand_starts")
    wandb.config.update(args)

    # Evaluation loop for VecEnv
    num_tasks = args.num_positions_per_dim**3
    goal_size = 3
    goals = np.zeros((num_tasks, goal_size))
    x_vals = np.linspace(args.goal_x_bounds[0], args.goal_x_bounds[1], args.num_positions_per_dim)
    y_vals = np.linspace(args.goal_y_bounds[0], args.goal_y_bounds[1], args.num_positions_per_dim)
    z_vals = np.linspace(args.goal_z_bounds[0], args.goal_z_bounds[1], args.num_positions_per_dim)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                goals[i * args.num_positions_per_dim**2 + j * args.num_positions_per_dim + k] = [
                    x,
                    y,
                    z,
                ]
    threshold = 0.005

    horizon = 1000
    env.set_attr("max_path_length", horizon)
    qpos_list = []

    for task_idx in tqdm(range(0, num_tasks, env.num_envs)):
        obs = env.reset()
        # set the goal for this rollout
        for env_idx in range(env.num_envs):
            env.env_method("_set_target_pos", goals[task_idx + env_idx], indices=[env_idx])
        obs[:, -3:] = goals[task_idx : task_idx + env.num_envs]
        if args.render:
            imgs = []
        for step in range(horizon):
            oracle_actions = []
            for env_idx in range(env.num_envs):
                oracle_actions.append(model.get_action(obs[env_idx], step))
            action = np.array(oracle_actions)
            next_obs, _, dones, _ = env.step(action)
            if args.render:
                img_array = env.render()
                imgs.append(img_array)
            if step < horizon - 1:
                assert not dones.any(), "Environment should not be done before max_steps"
            elif step == horizon - 1:
                assert dones.all(), "Environment should be done at max_steps"
            # if step < horizon - 1:
            #     assert np.all(next_obs[:,-3:] == goals[task_idx:task_idx+env.num_envs]), f"Goal should not change during rollout, \
            #         but original goal was {goals[task_idx:task_idx+env.num_envs]} and final goal was {next_obs[:,-3:]} at step {step}"
            if step == horizon - 2:
                print("Final hand position:", next_obs[:, :3])
                print("Final goal position:", next_obs[:, -3:])
            obs = next_obs
            hand_pos = obs[:, :3]
            goal_pos = goals[task_idx : task_idx + env.num_envs]
            distances = np.linalg.norm(hand_pos - goal_pos, axis=1)  # calculate Euclidean distance
            if np.all(distances < threshold):  # replace 'threshold' with the desired distance
                batch_qpos_list = [
                    pos_and_vel[0] for pos_and_vel in env.env_method("get_env_state")
                ]  # second element is the velocity
                qpos_list.extend(batch_qpos_list)
                break
        else:
            print(f"Goal {goals[task_idx:task_idx+env.num_envs]} not reached in {horizon} steps")

        if args.render:
            save_dir = os.path.join("runs", args.env, "videos")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rollout_{task_idx}.mp4")
            imageio.mimsave(save_path, imgs, fps=24)

    # save qpos_list to a file
    # breakpoint()
    qpos_array = np.array(qpos_list)
    qpos_array = qpos_array.reshape(-1, qpos_array.shape[-1])
    print("Shape of qpos_array:", qpos_array.shape)
    save_dir = os.path.join("artifacts", "hand_starts", wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"hand_starts.npy")
    np.save(save_path, qpos_array)


if __name__ == "__main__":
    main()
