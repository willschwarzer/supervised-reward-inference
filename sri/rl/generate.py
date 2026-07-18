from sb3_contrib import TQC
from stable_baselines3 import SAC, PPO

# from stable_baselines3.common.evaluation import evaluate_policy

from sri.rl.train import make_env
from sri.utils import (
    load_config_with_defaults,
    update_wandb_with_namespaces_and_names,
    get_dataset_name,
)

# from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.env_dict_temp import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

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
import mujoco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--general-config",
        type=str,
        default="general/default_generation.yml",
        help="General configuration file",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="datasets/default.yml",
        help="Demonstration dataset configuration file",
    )
    parser.add_argument(
        "--load-goals-from-datasets-with-config",
        type=str,
        default=None,
        help="Load goal list from datasets matching specified config file",
    )
    parser.add_argument(
        "--dataset-idx",
        type=int,
        default=0,
        help="Which dataset to take goals from (must have sufficient datasets)",
    )
    parser.add_argument(
        "--noise-coeff",
        type=float,
        default=None,
        help="Noise coefficient for generating rollouts",
    )
    parser.add_argument(
        "--goal-pos-adjustment-factor",
        type=float,
        default=None,
        help="Factor to adjust goal position by (usually -1.0 to 1.0)",
    )
    # parser.add_argument(
    #     "--wandb-project",
    #     type=str,
    #     default=None,
    #     help="Wandb project name",
    # )
    args = parser.parse_args()
    # if args.render:
    #     print(
    #         "###########################################\n\
    #           WARNING: RENDERING IS ON. THIS WILL BE SLOW.\n\
    #           ############################################"
    #     )

    general_config = load_config_with_defaults(args.general_config)
    dataset_config = load_config_with_defaults(args.dataset_config)
    if args.load_goals_from_datasets_with_config is not None:
        goal_dataset_config = load_config_with_defaults(
            args.load_goals_from_datasets_with_config
        )
    else:
        goal_dataset_config = None
    # if args.wandb_project is not None:
    #     print("overwriting config wandb project")
    #     general_config.wandb_project = args.wandb_project
    assert (
        general_config.wandb_project == dataset_config.wandb_project
    ), "Wandb project must be the same in general and dataset configs"
    if args.noise_coeff is not None:
        print("overwriting config noise coeff")
        dataset_config.noise_coeff = args.noise_coeff
    if args.goal_pos_adjustment_factor is not None:
        print("overwriting config goals position adjustment factor")
        dataset_config.goal_pos_adjustment_factor = args.goal_pos_adjustment_factor
    if dataset_config.goal_pos_adjustment_factor != 1.0:
        assert (
            "reach" in dataset_config.env
        ), "Goal position adjustment factor only implemented for reach tasks"
    # model_config = load_config("model/dummy.yml")
    # rl_config = load_config("rl/dummy.yml")
    if dataset_config.render:
        print(
            "###########################################\n\
            WARNING: RENDERING IS ON. THIS WILL BE SLOW.\n\
            ############################################"
        )
    dataset_config.env = dataset_config.env + "-v2-goal-observable"
    if goal_dataset_config is not None:
        goal_dataset_config.env = goal_dataset_config.env + "-v2-goal-observable"
    # args.extra_success_reward = 0.0  # ditto
    # args.limit_reward_obs = False  # ditto
    # args.render_reward = False  # ditto
    # args.unscale_rl_rewards = False  # ditto
    # args.rl_use_gt_reward = False  # ditto
    # args.rl_scale_gt_reward = False  # ditto
    # args.rl_use_gt_goal = False  # ditto
    # args.rl_no_task_rep = False  # ditto
    # args.proximity_reward = 0.0  # ditto
    # # we reinit goals every rollouts_per_task rollouts though
    return general_config, dataset_config, goal_dataset_config, args.dataset_idx


def augment_obs(obs, infos):
    augmented = np.zeros((obs.shape[0], obs.shape[1] + 3 * 7))
    for i, info in enumerate(infos):
        left_pad = info["left_pad"]
        right_pad = info["right_pad"]
        init_left_pad = info["init_left_pad"]
        init_right_pad = info["init_right_pad"]
        tcp_center = info["tcp_center"]
        obj_init_pos = info["obj_init_pos"]
        hand_init_pos = info["hand_init_pos"]
        assert all(
            [
                item is not None
                for item in [
                    left_pad,
                    right_pad,
                    init_left_pad,
                    init_right_pad,
                    tcp_center,
                    obj_init_pos,
                    hand_init_pos,
                ]
            ]
        ), "All of left_pad, right_pad, init_left_pad, init_right_pad, tcp_center, obj_init_pos, and hand_init_pos must be present in info"
        augmented[i] = np.concatenate(
            (
                left_pad,
                right_pad,
                init_left_pad,
                init_right_pad,
                tcp_center,
                obj_init_pos,
                hand_init_pos,
                obs[i],
            )
        )
    return augmented


def set_random_hand_start(env, hand_starts):
    random_indices = np.random.choice(hand_starts.shape[0], env.num_envs)
    random_starts = hand_starts[random_indices]
    for env_idx in range(env.num_envs):
        qvel = np.zeros_like(env.get_attr("data", indices=[env_idx])[0].qvel)
        reset_state = (random_starts[env_idx], qvel)
        env.env_method(
            "set_reset_state", reset_state, reset_only_hand=True, indices=[env_idx]
        )


def main():
    general_config, dataset_config, goal_dataset_config, dataset_idx = parse_args(
    )

    goal_size = 3
    x_bounds = dataset_config.goal_x_bounds
    y_bounds = dataset_config.goal_y_bounds
    z_bounds = dataset_config.goal_z_bounds
    if goal_dataset_config is not None:
        goal_dataset_name = get_dataset_name(dataset_idx, general_config,
                                             goal_dataset_config)
        goal_dataset_path = os.path.join("runs", goal_dataset_config.env,
                                         "rollouts", goal_dataset_name,
                                         "goals.npy")
        goals = np.load(goal_dataset_path)
        # clip goals to be within bounds
        goals_clipped = np.clip(
            goals,
            a_min=[x_bounds[0], y_bounds[0], z_bounds[0]],
            a_max=[x_bounds[1], y_bounds[1], z_bounds[1]],
        )
        if not np.all(goals == goals_clipped):
            print(
                f"Warning: some goals were outside of bounds and were clipped to be within bounds: {goals[goals != goals_clipped]}"
            )
        goals = goals_clipped
        # breakpoint()
        if (goal_dataset_config.env == "push-v2-goal-observable"
                and dataset_config.env == "reach-v2-goal-observable"):
            # randomly select a z value for the reach task from dataset_config.goal_z_bounds
            z_vals = np.random.uniform(
                low=z_bounds[0],
                high=z_bounds[1],
                size=(dataset_config.num_tasks_per_env *
                      dataset_config.num_envs),
            )
            print(
                f"Randomly selected z values for reach task, first 10 values: {z_vals[:10]}"
            )
            goals[:, 2] = z_vals
        if dataset_config.goals_avoid_obj:
            # make sure no goals are within dataset_config.goals_avoid_obj_dist
            # of dataset_config.goals_avoid_obj_pos
            assert dataset_config.goals_avoid_obj_dist is not None
            assert dataset_config.goals_avoid_obj_pos is not None
            avoid_obj_dist = dataset_config.goals_avoid_obj_dist
            avoid_obj_pos = dataset_config.goals_avoid_obj_pos
            assert not np.any(
                np.linalg.norm(goals[:, :2] -
                               avoid_obj_pos[:2], axis=1) < avoid_obj_dist
            ), "Some goals are too close to the object, but can't resample due to using other dataset's goals"
    else:
        # randomly generate goals
        goals = np.random.uniform(
            low=[x_bounds[0], y_bounds[0], z_bounds[0]],
            high=[x_bounds[1], y_bounds[1], z_bounds[1]],
            size=(
                dataset_config.num_tasks_per_env * dataset_config.num_envs,
                goal_size,
            ),
        )
        goal_dataset_name = None
        if dataset_config.goals_avoid_obj:
            # make sure no goals are within dataset_config.goals_avoid_obj_dist
            # of dataset_config.goals_avoid_obj_pos
            assert dataset_config.goals_avoid_obj_dist is not None
            assert dataset_config.goals_avoid_obj_pos is not None
            avoid_obj_dist = dataset_config.goals_avoid_obj_dist
            avoid_obj_pos = dataset_config.goals_avoid_obj_pos
            with tqdm(desc="Resampling goals") as pbar:
                if dataset_config.goals_avoid_obj_xy:
                    condition = (np.linalg.norm(goals[:, :2] -
                                                avoid_obj_pos[:2],
                                                axis=1) < avoid_obj_dist)
                else:
                    condition = (np.linalg.norm(goals - avoid_obj_pos, axis=1)
                                 < avoid_obj_dist)
                    # while np.any(
                    #     np.linalg.norm(goals[:, :2] - avoid_obj_pos[:2], axis=1) < avoid_obj_dist
                    # ):
                    # resample_indices = (
                    #     np.linalg.norm(goals[:, :2] - avoid_obj_pos[:2], axis=1) < avoid_obj_dist
                    # )
                while np.any(condition):
                    resample_indices = condition
                    goals[resample_indices] = np.random.uniform(
                        low=[x_bounds[0], y_bounds[0], z_bounds[0]],
                        high=[x_bounds[1], y_bounds[1], z_bounds[1]],
                        size=(resample_indices.sum(), goal_size),
                    )
                    pbar.update(1)
                    if dataset_config.goals_avoid_obj_xy:
                        condition = (np.linalg.norm(goals[:, :2] -
                                                    avoid_obj_pos[:2],
                                                    axis=1) < avoid_obj_dist)
                    else:
                        condition = (np.linalg.norm(goals - avoid_obj_pos,
                                                    axis=1) < avoid_obj_dist)
            # sort goals by distance to avoid_obj_pos
            # and print first 10 to show that they're far enough
            # distances = np.linalg.norm(goals[:, :2] - avoid_obj_pos[:2],
            #                            axis=1)
            if dataset_config.goals_avoid_obj_xy:
                distances = np.linalg.norm(goals[:, :2] - avoid_obj_pos[:2],
                                           axis=1)
            else:
                distances = np.linalg.norm(goals - avoid_obj_pos, axis=1)
            sorted_indices = np.argsort(distances)
            goals_by_dist = goals[sorted_indices]
            print(
                f"First 10 goals by distance to avoid_obj_pos: {goals_by_dist[:10]}"
            )
            sorted_dists = distances[sorted_indices]
            print(
                f"Distance from {avoid_obj_pos} to first 10 goals: {sorted_dists[:10]}"
            )

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[dataset_config.env]
    # breakpoint()
    if dataset_config.num_envs == 1:
        env = DummyVecEnv([make_env(env_cls, dataset_config)])
        env = VecMonitor(env)
    else:
        # env = SubprocVecEnv([make_env(env_cls, args) for _ in range(args.num_envs)])
        test_env = make_env(env_cls, dataset_config)
        # test_env()
        env = SubprocVecEnv([
            make_env(env_cls, dataset_config)
            for _ in range(dataset_config.num_envs)
        ])
        env = VecMonitor(env)
        print(f"Using SubprocVecEnv with {env.num_envs} environments")

    # set environment horizon
    env.set_attr("max_path_length", dataset_config.horizon)
    if dataset_config.model_path is not None:
        base_path = os.path.join("runs", dataset_config.model_path, "models")
        # args_path = f"{base_path}/model.args"
        model_path = f"{base_path}/model.zip"
        # with open(args_path, "rb") as f:
        #     env_args = pickle.load(f)
        # if hasattr(env_args, "gripped_start"):
        #     if args.gripped_start != env_args.gripped_start:
        #         print(
        #             f"Warning: agent was trained with gripped_start={env_args.gripped_start}, but evaluation is being done with gripped_start={args.gripped_start}"
        #         )
        # if args.env != env_args.env:
        #     print(
        #         f"Warning: agent was trained with env={env_args.env}, but evaluation is being done with env={args.env}"
        #     )
        # Environment setup
        # Load model
        algorithm = dataset_config.algorithm.lower()
        model_class = {"sac": SAC, "ppo": PPO, "tqc": TQC}.get(algorithm)
        if model_class is None:
            raise ValueError(f"Unknown algorithm {algorithm}")
        model = model_class.load(model_path, env=env)

    else:
        # this means we're using explicit control
        model_class = {
            "reach-v2-goal-observable": SawyerReachV2Policy,
            "push-v2-goal-observable": SawyerPushV2Policy,
            "pick-place-v2-goal-observable": SawyerPickPlaceV2Policy,
        }.get(dataset_config.env)
        model = model_class(
            circle_around=dataset_config.circle_around,
            circle_radius=dataset_config.circle_radius,
            horizon=dataset_config.horizon,
            mirror_goal=dataset_config.mirror_goal,
            x_bounds=dataset_config.goal_x_bounds,
            y_bounds=dataset_config.goal_y_bounds,
            z_bounds=dataset_config.goal_z_bounds,
            hand_speed=dataset_config.hand_speed,
            noise_coeff=dataset_config.noise_coeff,
            random_gripping=dataset_config.random_gripping,
            num_envs=dataset_config.num_envs,
            goal_pos_adjustment_factor=dataset_config.
            goal_pos_adjustment_factor,
        )
    # print(env_args)

    # initialize wandb
    # breakpoint()
    wandb.init(entity=general_config.wandb_entity,
               project=general_config.wandb_project)
    # wandb.config.update(args)
    namespaces_and_names = [
        (general_config, "general_config"),
        (dataset_config, "dataset_config"),
    ]
    if goal_dataset_config is not None:
        namespaces_and_names.append(
            (goal_dataset_config, "goal_dataset_config"))
    update_wandb_with_namespaces_and_names(namespaces_and_names)
    wandb.config.update({"goal_dataset": goal_dataset_name})

    # Evaluation loop for VecEnv
    num_tasks = dataset_config.num_tasks_per_env * dataset_config.num_envs
    obs_size = (env.observation_space.shape[0] - 3
                )  # -3 because the last 3 dimensions are the goal
    if dataset_config.extra_reward_info:
        # 3 for each: left_pad, right_pad, init_left_pad, init_right_pad, tcp_center, obj_init_pos, hand_init_pos
        # obs_size += 3*7
        num_augmented_dims = 3 * 7
        obs_size += num_augmented_dims
    actions_size = env.action_space.shape[0]

    observations = np.zeros(
        (
            num_tasks,
            dataset_config.num_rollouts_per_task,
            dataset_config.horizon + 1,
            obs_size,
        ),
        dtype=np.float32,
    )
    actions = np.zeros(
        (
            num_tasks,
            dataset_config.num_rollouts_per_task,
            dataset_config.horizon,
            actions_size,
        ),
        dtype=np.float32,
    )
    # goals = np.zeros((num_tasks, goal_size), dtype=np.float32)

    env.set_attr("_freeze_rand_vec",
                 False)  # need new object positions for each rollout

    if dataset_config.random_hand_starts:
        # load hand starts
        hand_starts_path = os.path.join("artifacts", "hand_starts",
                                        "hand_starts.npy")
        hand_starts = np.load(hand_starts_path)

    for task_idx in tqdm(range(0, num_tasks, env.num_envs)):
        goal_batch = goals[task_idx:task_idx + env.num_envs]
        if dataset_config.random_hand_starts:
            # make first hand start selection
            set_random_hand_start(env, hand_starts)
            # print("Selected initial random hand starts", env.get_attr("reset_state"))
        # breakpoint()
        obs = env.reset(
        )  # this doesn't have info because of vecenv differences
        # env.env_method("_set_target_pos", goal_batch)
        for env_idx in range(env.num_envs):
            env.env_method("_set_target_pos",
                           goal_batch[env_idx],
                           indices=[env_idx])
        obs[:, -3:] = goal_batch
        hand_init_pos_array = np.array(env.get_attr("hand_init_pos"))
        # instead we get info from env.reset_infos
        info = (env.venv.reset_infos
                )  # but we have to do this because of the stupid wrapper
        if dataset_config.extra_reward_info:
            obs = augment_obs(obs, info)  # (infos, obs, goal)
        # goals[task_idx : task_idx + env.num_envs] = obs[:, -3:]
        # print("First hand init pos array:", hand_init_pos_array)
        for ep in tqdm(range(0, dataset_config.num_rollouts_per_task)):
            observations[task_idx:task_idx + env.num_envs, ep, 0] = obs[:, :-3]
            if dataset_config.random_hand_starts:
                # set random hand start for each rollout (this will be applied next time the env resets)
                set_random_hand_start(env, hand_starts)
                # print("Selected random hand starts", env.get_attr("reset_state"), "for episode", ep+1)
            if ep > 0:
                assert not np.all(obs[:, :-3] == observations[
                    task_idx:task_idx + env.num_envs, ep - 1,
                    0]), "Starting pos should change between rollouts"
            # set the goal for this rollout
            for env_idx in range(env.num_envs):
                env.env_method("_set_target_pos",
                               goals[task_idx + env_idx],
                               indices=[env_idx])
            # this might be what I change if I also want suboptimal demonstrations: feed in different goals to policies
            # obs[:, -3:] = goals[task_idx : task_idx + env.num_envs]
            obs[:, -3:] = goal_batch
            if dataset_config.render:
                imgs = []
            for step in range(dataset_config.horizon):
                if dataset_config.extra_reward_info:
                    action_obs = obs[:, num_augmented_dims:]
                else:
                    action_obs = obs
                if dataset_config.model_path is None:
                    oracle_actions = []
                    for env_idx in range(env.num_envs):
                        oracle_actions.append(
                            model.get_action(action_obs[env_idx], step,
                                             env_idx))
                    action = np.array(oracle_actions)
                else:
                    action, _ = model.predict(action_obs, deterministic=True)
                # print("Hand init pos array before step", step, ":", env.get_attr("hand_init_pos"))
                next_obs, _, dones, info = env.step(action)
                # big_diff = np.any(np.abs(np.array(next_obs)[:, :3] - np.array([info_dict["tcp_center"] for info_dict in info])) > 0.06)
                # if big_diff:
                #     breakpoint()
                # print("Hand init pos array after step", step, ":", env.get_attr("hand_init_pos"))
                if dataset_config.render:
                    img_array = env.render()
                    imgs.append(img_array)
                # we need to process observations differently depending on whether we've finished the rollout or not
                if step == dataset_config.horizon - 1:
                    # in this case, the observation we want is actually info["terminal_observation"]
                    # and the info we want is actually env.venv.reset_infos (don't ask me why we have to get env.venv)
                    # (it has something to do with the wrapper)
                    # stable-baselines3 vecenvs deal with resetting extremely strangely.
                    # on the step when they reset - which happens when done is True - they return the
                    # *new* observation, but the *old* info.
                    # So for the last step of this rollout, we need to use this info, but
                    # info["terminal_observation"] as the observation.
                    # Then, to initialize the next rollout, we need to use next_obs as the observation,
                    # but env.venv.reset_infos as the info. Ugh. Why.
                    # breakpoint()
                    this_ep_info = info
                    this_ep_obs = np.array([
                        info_dict["terminal_observation"]
                        for info_dict in this_ep_info
                    ])
                    if dataset_config.extra_reward_info:
                        this_ep_obs = augment_obs(this_ep_obs, this_ep_info)
                    observations[task_idx:task_idx + env.num_envs, ep,
                                 step + 1] = (this_ep_obs[:, :-3])
                    # now we need to set up the next rollout
                    next_ep_info = env.venv.reset_infos
                    if dataset_config.extra_reward_info:
                        next_obs = augment_obs(next_obs, next_ep_info)

                    # just some sanity checks
                    assert dones.all(
                    ), "Environment should be done at max_steps"
                    hand_init_pos_array = np.array(
                        env.get_attr("hand_init_pos"))
                else:
                    if dataset_config.extra_reward_info:
                        next_obs = augment_obs(next_obs, info)
                    # just some sanity checks
                    assert (
                        not dones.any()
                    ), "Environment should not be done before max_steps"
                    # using allclose because of floating point errors (not sure why there are any)
                    assert np.allclose(
                        next_obs[:,
                                 -3:], goals[task_idx:task_idx + env.num_envs]
                    ), f"Goal should not change during rollout, \
                        but original goal was {goals[task_idx:task_idx+env.num_envs]} and final goal was {next_obs[:,-3:]} at step {step}"

                    if dataset_config.extra_reward_info:
                        # breakpoint()
                        obs_hand_init_pos = next_obs[:, 18:21]
                        assert np.all(
                            obs_hand_init_pos == hand_init_pos_array
                        ), f"Hand init pos should not change during rollout, \
                            but original hand init pos was {hand_init_pos_array} and final hand init pos was {obs_hand_init_pos} at step {step}"

                    observations[task_idx:task_idx + env.num_envs, ep,
                                 step + 1] = (next_obs[:, :-3])

                actions[task_idx:task_idx + env.num_envs, ep, step] = action
                obs = next_obs
            if dataset_config.render:
                # save_dir = os.path.join("scratch", "runs", dataset_config.env, "videos")
                save_dir = os.path.join("scratch_2", "runs",
                                        dataset_config.env, "videos")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir,
                                         f"rollout_{task_idx}_{ep}.mp4")
                imageio.mimsave(save_path, imgs, fps=24)

    print(f"Rollouts shape: {observations.shape}")
    print(f"Goals shape: {goals.shape}")

    # if dataset_config.wandb_run_name is not None:
    #     base_rollout_dir = os.path.join("runs", dataset_config.wandb_run_name, "rollouts")
    # else:
    # base_rollout_dir = os.path.join("runs", dataset_config.env, "rollouts")
    # base_rollout_dir = os.path.join("scratch", "runs", dataset_config.env, "rollouts")
    # base_rollout_dir = os.path.join("scratch_2", "runs", dataset_config.env,
    #                                 "rollouts")
    base_rollout_dir = os.path.join("scratch", "runs",
                                    dataset_config.env, "rollouts")

    if dataset_config.chai_rollouts:
        trajectories = []
        print("Converting to CHAI rollouts")
        # for ep in tqdm(range(n_eval_episodes)):
        for task in tqdm(range(num_tasks)):
            for ep in range(dataset_config.num_rollouts_per_task):
                trajectory = Trajectory(
                    obs=observations[task, ep],
                    acts=actions[task, ep],
                    infos=None,
                    terminal=True,
                )
                trajectories.append(trajectory)
            trajectories.append(trajectory)
        run_name = wandb.run.name
        save_path = os.path.join(base_rollout_dir, "chai", run_name)
        os.makedirs(save_path, exist_ok=True)
        serialize.save(save_path, trajectories)
        print(f"Saved rollouts to {save_path}")
    else:
        print("Not converting to CHAI rollouts")
        save_path = base_rollout_dir
        run_name = wandb.run.name
        save_path = os.path.join(base_rollout_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "observations.npy"), observations)
        np.save(os.path.join(save_path, "actions.npy"), actions)
        np.save(os.path.join(save_path, "goals.npy"), goals)
        print(
            f"Saved rollouts to {save_path}/{{observations,actions,goals}}.npy"
        )


if __name__ == "__main__":
    main()
