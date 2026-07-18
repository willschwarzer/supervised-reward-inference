import numpy as np

# from pynvml import *
from tqdm import tqdm
import torch
from imitation.data.types import Trajectory
from imitation.data import serialize
import wandb
import os
import yaml
import argparse
import ipdb

# def get_freest_gpu():
#     nvmlInit()
#     NUM_DEVICES = 4
#     free_mems = []
#     for i in range(NUM_DEVICES):
#         h = nvmlDeviceGetHandleByIndex(i)
#         info = nvmlDeviceGetMemoryInfo(h)
#         free_mems.append(info.free)
#     return np.argmax(free_mems)
#     # mems = [torch.cuda.memory_allocated(i) for i in range(num_devices)]
#     # mems2 = [torch.cuda.memory_reserved(i) for i in range(num_devices)]
#     # return mems, mems2


def scale(x, low, high, end_low=-3, end_high=3):
    # Scales torch float tensor x from [low, high] to [end_low, end_high]
    return (x - low) / (high - low) * (end_high - end_low) + end_low


def convert_chai_rollouts(rollouts, horizon, obs_size, dtype):
    states = np.zeros((len(rollouts), horizon, obs_size), dtype=dtype)
    rewards = np.zeros((len(rollouts), horizon), dtype=float)
    for idx, rollout in tqdm(enumerate(rollouts)):
        rollout_traj = rollout.obs[:-2]
        if len(rollout_traj.shape) == 3:
            rollout_traj = np.reshape(rollout_traj, (rollout_traj.shape[0], -1))
        states[idx] = rollout_traj
        rewards[idx] = rollout.rews[:-1]
    return states, rewards


def convert_rollouts_to_chai(observations, actions):
    num_tasks = observations.shape[0]
    num_rollouts_per_task = observations.shape[1]
    trajectories_by_task = []
    for task in tqdm(range(num_tasks)):
        trajectories = []
        # breakpoint()
        for ep in range(num_rollouts_per_task):
            trajectory = Trajectory(
                obs=observations[task, ep].clone().numpy(),
                acts=actions[task, ep].clone().numpy(),
                infos=None,
                terminal=False,
            )
            trajectories.append(trajectory)
        trajectories_by_task.append(trajectories)
    return trajectories_by_task


def save_chai_rollouts(observations, actions, directory):
    num_tasks = observations.shape[0]
    num_rollouts_per_task = observations.shape[1]
    trajectories = convert_rollouts_to_chai(observations, actions)
    filename = f"{directory}/chai_rollouts"
    serialize.save(filename, trajectories)


def generate_gadget_demonstration(reward_weights):
    """
    Generate a single gadget demonstration based on reward weights.

    :param reward_weights: A single set of reward weights of length 10.
    :return: A single gadget demonstration.
    """
    num_rings = 5  # Assuming five rings
    trajectory_length = 50  # Total trajectory length for all pairs

    # Initialize the demonstrations array
    demonstration = np.zeros((trajectory_length, num_rings * 2))

    # Convert flattened weights to a matrix
    weight_matrix = np.zeros((num_rings, num_rings))
    weight_matrix[np.tril_indices(num_rings, k=-1)] = reward_weights

    # breakpoint()

    # Iterate through each pair of rings
    step = 0
    for i in range(num_rings):
        for j in range(i + 1, num_rings):
            # breakpoint()
            weight = weight_matrix[j, i]  # this looks wrong, but it's not
            # (remember that the matrix is lower triangular, so as long as
            # we get a non-zero weight, it doesn't matter whether we got it
            # with i or j as the row index)

            # Initialize positions
            if weight <= 0:
                pos1, pos2 = [-0.5, 0], [-0.5, 0]
            else:
                pos1, pos2 = [-0.5, -0.5], [-0.5, 0.5]

            delta = (1 / 8) * weight
            # Update positions for next four steps
            for t in range(5):
                # Update the positions in the demonstration
                demonstration[step, i * 2] = pos1[0]
                demonstration[step, i * 2 + 1] = pos1[1]
                demonstration[step, j * 2] = pos2[0]
                demonstration[step, j * 2 + 1] = pos2[1]
                if weight <= 0:
                    pos1[1] -= delta
                    pos2[1] += delta
                else:
                    pos1[1] += delta
                    pos2[1] -= delta
                step += 1
                if np.any(demonstration > 0.5) or np.any(demonstration < -0.5):
                    assert False, "Demonstration out of bounds"

    return torch.tensor(demonstration, dtype=torch.float32)


def namespace_to_dict(namespace, prefix):
    return {f"{prefix}_{key}": value for key, value in vars(namespace).items()}


def load_config(file_path):
    new_path = os.path.join("config", file_path)
    # breakpoint()
    with open(new_path, "r") as f:
        config = yaml.safe_load(f)
        config = {key.replace("-", "_"): value for key, value in config.items()}
        return argparse.Namespace(**config)


def add_prefix_to_namespace(namespace, prefix):
    return argparse.Namespace(
        **{f"{prefix}_{key}": value for key, value in vars(namespace).items()}
    )


def get_dataset_name(
    dataset_idx,
    general_config,
    dataset_config,
    include_goal_params=True,
    most_recent_first=True,
):
    # Initialize Weights & Biases
    api = wandb.Api()

    # Filter runs using the API for certain parameters
    filters = {
        "state": "finished",
        "config.dataset_config_circle_around": dataset_config.circle_around,
        "config.dataset_config_circle_radius": dataset_config.circle_radius,
        "config.dataset_config_mirror_goal": dataset_config.mirror_goal,
        "config.dataset_config_env": dataset_config.env,
        "config.dataset_config_noise_coeff": dataset_config.noise_coeff,
        "config.dataset_config_num_tasks_per_env": dataset_config.num_tasks_per_env,
        "config.dataset_config_num_envs": dataset_config.num_envs,
        "config.dataset_config_num_rollouts_per_task": dataset_config.num_rollouts_per_task,
        "config.dataset_config_gripped_start": dataset_config.gripped_start,
        "config.dataset_config_hand_speed": dataset_config.hand_speed,
        "config.dataset_config_horizon": dataset_config.horizon,
        "config.dataset_config_random_hand_starts": dataset_config.random_hand_starts,
    }
    # breakpoint()
    if include_goal_params:
        filters.update(
            {
                "config.dataset_config_goal_x_bounds": dataset_config.goal_x_bounds,
                "config.dataset_config_goal_y_bounds": dataset_config.goal_y_bounds,
                "config.dataset_config_goal_z_bounds": dataset_config.goal_z_bounds,
            }
        )
        # if hasattr(dataset_config, "goals_avoid_obj"):
        #     filters.update({
        #         "config.dataset_config_goals_avoid_obj":
        #         dataset_config.goals_avoid_obj,
        #         "config.dataset_config_goals_avoid_obj_xy":
        #         dataset_config.goals_avoid_obj_xy,
        #         "config.dataset_config_goals_avoid_obj_dist":
        #         dataset_config.goals_avoid_obj_dist,
        #         "config.dataset_config_goals_avoid_obj_pos":
        #         dataset_config.goals_avoid_obj_pos,
        #     })
        # "config.dataset_config_goals_avoid_obj": dataset_config.goals_avoid_obj,
        # "config.dataset_config_goals_avoid_obj_xy": dataset_config.goals_avoid_obj_xy,
        # "config.dataset_config_goals_avoid_obj_dist": dataset_config.goals_avoid_obj_dist,
        # "config.dataset_config_goals_avoid_obj_pos": dataset_config.goals_avoid_obj_pos,
    # goals-avoid-obj: true
    # goals-avoid-obj-xy: true
    # goals-avoid-obj-dist: 0.06
    # goals-avoid-obj-pos: [0.0, 0.6, 0.02]
    # if dataset_config.experiment_id is not None:
    #     filters["config.dataset_config_experiment_id"] = dataset_config.experiment_id
    runs = list(
        api.runs(
            path=f"{general_config.wandb_entity}/{dataset_config.wandb_project}",
            filters=filters,
        )
    )
    # Make ordering explicit and stable across wandb client/API changes.
    runs.sort(key=lambda run: run.created_at or "")
    if most_recent_first:
        runs = list(reversed(runs))

    assert len(runs) > dataset_idx, f"Only {len(runs)} runs found with these parameters"
    # run = runs[dataset_idx]
    cur_idx = 0
    # breakpoint()
    for run in runs:
        if (
            "dataset_config_random_gripping" not in run.config
            and dataset_config.random_gripping
        ):
            continue
        if hasattr(dataset_config, "goals_avoid_obj"):
            if (
                "dataset_config_goals_avoid_obj" not in run.config
                and dataset_config.goals_avoid_obj
            ):
                continue
            elif (
                "dataset_config_goals_avoid_obj" in run.config
                and run.config["dataset_config_goals_avoid_obj"]
                != dataset_config.goals_avoid_obj
            ):
                continue
            if (
                "dataset_config_goals_avoid_obj" in run.config
                and run.config["dataset_config_goals_avoid_obj"]
            ):
                if (
                    "dataset_config_goals_avoid_obj_xy" not in run.config
                    or run.config["dataset_config_goals_avoid_obj_xy"]
                    != dataset_config.goals_avoid_obj_xy
                ):
                    continue
                if (
                    "dataset_config_goals_avoid_obj_dist" not in run.config
                    or run.config["dataset_config_goals_avoid_obj_dist"]
                    != dataset_config.goals_avoid_obj_dist
                ):
                    continue
                if (
                    "dataset_config_goals_avoid_obj_pos" not in run.config
                    or run.config["dataset_config_goals_avoid_obj_pos"]
                    != dataset_config.goals_avoid_obj_pos
                ):
                    continue
        if hasattr(dataset_config, "goal_pos_adjustment_factor"):
            if (
                "dataset_config_goal_pos_adjustment_factor" not in run.config
                and dataset_config.goal_pos_adjustment_factor != 1.0
            ):
                continue
            elif (
                "dataset_config_goal_pos_adjustment_factor" in run.config
                and run.config["dataset_config_goal_pos_adjustment_factor"]
                != dataset_config.goal_pos_adjustment_factor
            ):
                continue
        if cur_idx == dataset_idx:
            run_name = run.name
            return run_name
        cur_idx += 1
    raise ValueError(
        "Not enough runs found for dataset_idx", dataset_idx, "only found", cur_idx
    )


def get_model_names(
    general_config,
    model_config,
    train_config,
    dataset_config,
    obs_dataset_config,
    most_recent_first=True,
    model_idxs=[0],
    rl=False,
    rl_config=None,
    baseline=False,
    baselines_config=None,
    inference_config=None,
):
    if rl:
        assert rl_config is not None, "rl_config must be provided if rl is True"
    # Ensure model_idxs is monotonically increasing
    assert all(
        x < y for x, y in zip(model_idxs, model_idxs[1:])
    ), "model_idxs must be monotonically increasing"
    if baseline:
        assert not rl, "Cannot have baseline and rl be True at the same time"
        assert (
            baselines_config is not None
        ), "baselines_config must be provided if baseline is True"

    # Initialize Weights & Biases
    api = wandb.Api()
    filters = {
        # "state": "finished",
        "config.dataset_config_horizon": dataset_config.horizon,
        "config.dataset_config_noise_coeff": dataset_config.noise_coeff,
        "config.dataset_config_circle_around": dataset_config.circle_around,
        "config.dataset_config_circle_radius": dataset_config.circle_radius,
        "config.dataset_config_mirror_goal": dataset_config.mirror_goal,
        "config.dataset_config_num_tasks_per_env": dataset_config.num_tasks_per_env,
        "config.dataset_config_num_envs": dataset_config.num_envs,
        "config.dataset_config_num_rollouts_per_task": dataset_config.num_rollouts_per_task,
        "config.dataset_config_env": dataset_config.env,
        "config.dataset_config_goal_x_bounds": dataset_config.goal_x_bounds,
        "config.dataset_config_goal_y_bounds": dataset_config.goal_y_bounds,
        "config.dataset_config_goal_z_bounds": dataset_config.goal_z_bounds,
        "config.train_config_train_split": train_config.train_split,
    }
    # breakpoint()
    if not baseline:
        filters.update(
            {
                "config.train_config_env": train_config.env,
                "config.train_config_num_epochs": train_config.num_epochs,
                "config.train_config_num_obs": train_config.num_obs,
                "config.train_config_synthesize_obs": train_config.synthesize_obs,
                "config.train_config_synth_on_grid": train_config.synth_on_grid,
                "config.train_config_synth_grid_size": train_config.synth_grid_size,
                "config.train_config_synth_frame_stacking": train_config.synth_frame_stacking,
                "config.train_config_n": train_config.n,
                "config.model_config_demonstration_rep_dim": model_config.demonstration_rep_dim,
                "config.model_config_state_rep_dim": model_config.state_rep_dim,
                "config.model_config_internal_tst_dim": model_config.internal_tst_dim,
                "config.model_config_state_hidden_size": model_config.state_hidden_size,
                "config.model_config_reward_hidden_size": model_config.reward_hidden_size,
                "config.model_config_demonstration_hidden_size": model_config.demonstration_hidden_size,
                "config.model_config_num_demonstration_layers": model_config.num_demonstration_layers,
                "config.model_config_num_state_layers": model_config.num_state_layers,
                "config.model_config_mlp": model_config.mlp,
                "config.model_config_dem_encoder_type": model_config.dem_encoder_type,
                # "config.model_config_direct_goal_inference": model_config.direct_goal_inference,
                "config.obs_dataset_config_noise_coeff": obs_dataset_config.noise_coeff,
                "config.obs_dataset_config_circle_around": obs_dataset_config.circle_around,
                "config.obs_dataset_config_circle_radius": obs_dataset_config.circle_radius,
                "config.obs_dataset_config_mirror_goal": obs_dataset_config.mirror_goal,
                "config.obs_dataset_config_num_tasks_per_env": obs_dataset_config.num_tasks_per_env,
                "config.obs_dataset_config_num_envs": obs_dataset_config.num_envs,
                "config.obs_dataset_config_num_rollouts_per_task": obs_dataset_config.num_rollouts_per_task,
                "config.obs_dataset_config_env": obs_dataset_config.env,
                "config.obs_dataset_config_goal_x_bounds": obs_dataset_config.goal_x_bounds,
                "config.obs_dataset_config_goal_y_bounds": obs_dataset_config.goal_y_bounds,
                "config.obs_dataset_config_goal_z_bounds": obs_dataset_config.goal_z_bounds,
            }
        )
        if hasattr(model_config, "direct_goal_inference"):
            filters.update(
                {
                    "config.model_config_direct_goal_inference": model_config.direct_goal_inference
                }
            )
    if rl:
        filters.update(
            {
                "config.rl_config_learning_steps": rl_config.learning_steps,
                "config.rl_config_algorithm": rl_config.algorithm,
                "config.rl_config_use_gt_reward": rl_config.use_gt_reward,
                "config.rl_config_scale_gt_reward": rl_config.scale_gt_reward,
                "config.rl_config_policy_latent_dim": rl_config.policy_latent_dim,
                "config.rl_config_policy_num_layers": rl_config.policy_num_layers,
                "config.rl_config_use_sde": rl_config.use_sde,
                "config.rl_config_offline_rl": rl_config.offline_rl,
                "config.rl_config_no_task_rep": rl_config.no_task_rep,
                "config.rl_config_ppo_n_steps": rl_config.ppo_n_steps,
                "config.rl_config_third_gt": rl_config.third_gt,
                "config.rl_config_half_gt": rl_config.half_gt,
                "config.rl_config_full_legacy": rl_config.full_legacy,
                "config.rl_config_extra_success_reward": rl_config.extra_success_reward,
                "config.rl_config_no_init_success": rl_config.no_init_success,
                "config.rl_config_success_requires_touch": rl_config.success_requires_touch,
            }
        )
    if baseline:
        # inference config:
        # env: 'reach'
        # horizon: 500
        # num-goals: 1
        # n: 100
        # include-actions: true
        # baselines-only: true
        # include-partial-reward-info: true
        # mask-obj: true

        # baselines config:
        # baselines: 'none'
        # bc-epochs: 100
        # adv-its: 100000

        # rl_config:
        # policy-latent-dim: 512
        # policy-num-layers: 2
        filters.update(
            {
                "config.inference_config_env": inference_config.env,
                "config.inference_config_horizon": inference_config.horizon,
                "config.inference_config_num_goals": inference_config.num_goals,
                "config.inference_config_n": inference_config.n,
                "config.inference_config_include_actions": inference_config.include_actions,
                "config.inference_config_baselines_only": inference_config.baselines_only,
                "config.inference_config_include_partial_reward_info": inference_config.include_partial_reward_info,
                "config.inference_config_mask_obj": inference_config.mask_obj,
                "config.baselines_config_baselines": baselines_config.baselines,
                "config.baselines_config_bc_epochs": baselines_config.bc_epochs,
                "config.baselines_config_adv_its": baselines_config.adv_its,
                "config.rl_config_policy_latent_dim": rl_config.policy_latent_dim,
                "config.rl_config_policy_num_layers": rl_config.policy_num_layers,
            }
        )

    wandb_project = (
        train_config.wandb_project if not baseline else baselines_config.wandb_project
    )
    runs = list(
        api.runs(
            path=f"{general_config.wandb_entity}/{wandb_project}",
            filters=filters,
        )
    )
    # Make ordering explicit and stable across wandb client/API changes.
    runs.sort(key=lambda run: run.created_at or "")
    if most_recent_first:
        runs = list(reversed(runs))
    assert len(runs) > 0, "No runs found with these parameters"
    big_int = int(
        1e18
    )  # anything bigger causes wandb to error, saying it's a float? bizarre
    cur_idx = 0
    model_names = []
    for run in runs:
        if rl or baseline or "average train loss" in run.summary:
            # if train_config.minimal_synth and ("train_config_minimal_synth" not in run.config or not run.config["train_config_minimal_synth"]):
            #     continue
            # if train_config.synth_random_grip and ("train_config_synth_random_grip" not in run.config or not run.config["train_config_synth_random_grip"]):
            #     continue
            # for parameters we added later, we'll assume they were "false"
            # if they're not in the run config
            if not baseline:
                if (
                    "train_config_minimal_synth" not in run.config
                    and train_config.minimal_synth
                ) or (
                    "train_config_minimal_synth" in run.config
                    and run.config["train_config_minimal_synth"]
                    != train_config.minimal_synth
                ):
                    continue
                if (
                    "train_config_synth_random_grip" not in run.config
                    and train_config.synth_random_grip
                ) or (
                    "train_config_synth_random_grip" in run.config
                    and run.config["train_config_synth_random_grip"]
                    != train_config.synth_random_grip
                ):
                    continue
            if hasattr(dataset_config, "goal_pos_adjustment_factor"):
                if (
                    "dataset_config_goal_pos_adjustment_factor" not in run.config
                    and dataset_config.goal_pos_adjustment_factor != 1.0
                ):
                    continue
                elif (
                    "dataset_config_goal_pos_adjustment_factor" in run.config
                    and run.config["dataset_config_goal_pos_adjustment_factor"]
                    != dataset_config.goal_pos_adjustment_factor
                ):
                    continue
            if hasattr(dataset_config, "goals_avoid_obj"):
                if (
                    "dataset_config_goals_avoid_obj" not in run.config
                    and dataset_config.goals_avoid_obj
                ):
                    continue
                elif (
                    "dataset_config_goals_avoid_obj" in run.config
                    and run.config["dataset_config_goals_avoid_obj"]
                    != dataset_config.goals_avoid_obj
                ):
                    continue
                if (
                    "dataset_config_goals_avoid_obj" in run.config
                    and run.config["dataset_config_goals_avoid_obj"]
                ):
                    if (
                        "dataset_config_goals_avoid_obj_xy" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_xy"]
                        != dataset_config.goals_avoid_obj_xy
                    ):
                        continue
                    if (
                        "dataset_config_goals_avoid_obj_dist" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_dist"]
                        != dataset_config.goals_avoid_obj_dist
                    ):
                        continue
                    if (
                        "dataset_config_goals_avoid_obj_pos" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_pos"]
                        != dataset_config.goals_avoid_obj_pos
                    ):
                        continue
            if hasattr(train_config, "grasp_rew_only"):
                if (
                    "train_config_grasp_rew_only" not in run.config
                    and train_config.grasp_rew_only
                ):
                    continue
                elif (
                    "train_config_grasp_rew_only" in run.config
                    and run.config["train_config_grasp_rew_only"]
                    != train_config.grasp_rew_only
                ):
                    continue
            if rl:
                if (
                    "rl_config_ensembling" not in run.config
                    or run.config["rl_config_ensembling"] != rl_config.ensembling
                ):
                    continue
            if hasattr(train_config, "start_prop_obs_1"):
                if (
                    "train_config_start_prop_obs_1" not in run.config
                    and train_config.start_prop_obs_1 != "auto"
                ):
                    continue
                elif (
                    "train_config_start_prop_obs_1" in run.config
                    and run.config["train_config_start_prop_obs_1"]
                    != train_config.start_prop_obs_1
                ):
                    continue
            if hasattr(train_config, "end_prop_obs_1"):
                if (
                    "train_config_end_prop_obs_1" not in run.config
                    and train_config.end_prop_obs_1 != "auto"
                ):
                    continue
                elif (
                    "train_config_end_prop_obs_1" in run.config
                    and run.config["train_config_end_prop_obs_1"]
                    != train_config.end_prop_obs_1
                ):
                    continue
            if not rl and not baseline:
                train_loss_history = run.history(
                    keys=["average train loss"], samples=big_int
                )
                if len(train_loss_history) == train_config.num_epochs:
                    if cur_idx in model_idxs:
                        model_names.append(run.name)
                    cur_idx += 1
                    # print(f"Run {run.name} was valid, incrementing cur_idx to {cur_idx}")
            elif rl:
                # check if "final.zip" is in the run's files, specifically
                # runs/run_name/models/final.zip
                # if os.path.exists(f"runs/{run.name}/models/final.zip"):
                if find_path(general_config.run_dirs, [run.name, "models"], ["final.zip"], raise_on_missing=False) is not None:
                    if cur_idx in model_idxs:
                        model_names.append(run.name)
                    cur_idx += 1
            elif baseline:
                assert (
                    len(baselines_config.baselines) == 1
                ), "Only one baseline can be loaded for eval at a time"
                method = baselines_config.baselines[0]
                if method == "pemirl":
                    baseline_exists = (
                        find_path(
                            general_config.run_dirs,
                            [run.name, "pemirl"],
                            ["meta_checkpoint.pt"],
                            raise_on_missing=False,
                        )
                        is not None
                    )
                else:
                    # if os.path.exists(f"runs/{run.name}/{method}/policy.zip"):
                    baseline_exists = (
                        find_path(
                            general_config.run_dirs,
                            [run.name, method],
                            ["policy.zip"],
                            raise_on_missing=False,
                        )
                        is not None
                    )
                if baseline_exists:
                    if cur_idx in model_idxs:
                        model_names.append(run.name)
                    cur_idx += 1
            else:
                print("what? <(O.O)> <(O.O<) <(O.O)> (>O.O)>")
            if len(model_names) == len(model_idxs):
                break
    if not model_names:
        # raise ValueError("No run found with 2000 train loss history for the given indices")
        raise ValueError(
            f"Not enough runs found for model_idxs {model_idxs} with {train_config.num_epochs} train history, only found {cur_idx}"
        )
    return model_names


def get_eval_results(
    general_config,
    model_config,
    train_config,
    dataset_config,
    obs_dataset_config,
    most_recent_first=True,
    model_idxs=[0],
    rl=False,
    rl_config=None,
    baseline=False,
    baselines_config=None,
    inference_config=None,
    unique=False,
):
    if rl:
        assert rl_config is not None, "rl_config must be provided if rl is True"
    # Ensure model_idxs is monotonically increasing
    assert all(
        x < y for x, y in zip(model_idxs, model_idxs[1:])
    ), "model_idxs must be monotonically increasing"
    if baseline:
        assert not rl, "Cannot have baseline and rl be True at the same time"
        assert (
            baselines_config is not None
        ), "baselines_config must be provided if baseline is True"

    if unique:
        assert rl or baseline, "unique=True only works for rl or baseline (right now)"
        rl_run_names_seen = set()
        if not baseline:
            model_run_names_seen = set()
    # Initialize Weights & Biases
    api = wandb.Api()
    filters = {
        # "state": "finished",
        "config.dataset_config_horizon": dataset_config.horizon,
        "config.dataset_config_noise_coeff": dataset_config.noise_coeff,
        "config.dataset_config_circle_around": dataset_config.circle_around,
        "config.dataset_config_circle_radius": dataset_config.circle_radius,
        "config.dataset_config_mirror_goal": dataset_config.mirror_goal,
        "config.dataset_config_num_tasks_per_env": dataset_config.num_tasks_per_env,
        "config.dataset_config_num_envs": dataset_config.num_envs,
        "config.dataset_config_num_rollouts_per_task": dataset_config.num_rollouts_per_task,
        "config.dataset_config_env": dataset_config.env,
        "config.train_config_train_split": train_config.train_split,
    }
    # breakpoint()
    # ipdb.set_trace()
    if not baseline:
        filters.update(
            {
                "config.train_config_env": train_config.env,
                "config.train_config_num_epochs": train_config.num_epochs,
                "config.train_config_num_obs": train_config.num_obs,
                "config.train_config_synthesize_obs": train_config.synthesize_obs,
                "config.train_config_synth_on_grid": train_config.synth_on_grid,
                "config.train_config_synth_grid_size": train_config.synth_grid_size,
                "config.train_config_synth_frame_stacking": train_config.synth_frame_stacking,
                "config.train_config_n": train_config.n,
                "config.model_config_demonstration_rep_dim": model_config.demonstration_rep_dim,
                "config.model_config_state_rep_dim": model_config.state_rep_dim,
                "config.model_config_internal_tst_dim": model_config.internal_tst_dim,
                "config.model_config_state_hidden_size": model_config.state_hidden_size,
                "config.model_config_reward_hidden_size": model_config.reward_hidden_size,
                "config.model_config_demonstration_hidden_size": model_config.demonstration_hidden_size,
                "config.model_config_num_demonstration_layers": model_config.num_demonstration_layers,
                "config.model_config_num_state_layers": model_config.num_state_layers,
                "config.model_config_mlp": model_config.mlp,
                "config.model_config_dem_encoder_type": model_config.dem_encoder_type,
                # "config.model_config_direct_goal_inference": model_config.direct_goal_inference,
                "config.obs_dataset_config_noise_coeff": obs_dataset_config.noise_coeff,
                "config.obs_dataset_config_circle_around": obs_dataset_config.circle_around,
                "config.obs_dataset_config_circle_radius": obs_dataset_config.circle_radius,
                "config.obs_dataset_config_mirror_goal": obs_dataset_config.mirror_goal,
                "config.obs_dataset_config_num_tasks_per_env": obs_dataset_config.num_tasks_per_env,
                "config.obs_dataset_config_num_envs": obs_dataset_config.num_envs,
                "config.obs_dataset_config_num_rollouts_per_task": obs_dataset_config.num_rollouts_per_task,
                "config.obs_dataset_config_env": obs_dataset_config.env,
            }
        )
    filters.update(
        {
            "config.inference_config_env": inference_config.env,
            "config.inference_config_horizon": inference_config.horizon,
            "config.inference_config_num_goals": inference_config.num_goals,
        }
    )

    if rl:
        filters.update(
            {
                "config.rl_config_learning_steps": rl_config.learning_steps,
                "config.rl_config_algorithm": rl_config.algorithm,
                "config.rl_config_use_gt_reward": rl_config.use_gt_reward,
                "config.rl_config_scale_gt_reward": rl_config.scale_gt_reward,
                "config.rl_config_policy_latent_dim": rl_config.policy_latent_dim,
                "config.rl_config_policy_num_layers": rl_config.policy_num_layers,
                "config.rl_config_use_sde": rl_config.use_sde,
                "config.rl_config_offline_rl": rl_config.offline_rl,
                "config.rl_config_no_task_rep": rl_config.no_task_rep,
                "config.rl_config_ppo_n_steps": rl_config.ppo_n_steps,
                "config.rl_config_third_gt": rl_config.third_gt,
                "config.rl_config_half_gt": rl_config.half_gt,
                "config.rl_config_full_legacy": rl_config.full_legacy,
                "config.rl_config_extra_success_reward": rl_config.extra_success_reward,
                "config.rl_config_no_init_success": rl_config.no_init_success,
                "config.rl_config_success_requires_touch": rl_config.success_requires_touch,
            }
        )
    if baseline:
        filters.update(
            {
                "config.inference_config_n": inference_config.n,
                "config.inference_config_include_actions": inference_config.include_actions,
                "config.inference_config_include_partial_reward_info": inference_config.include_partial_reward_info,
                "config.inference_config_mask_obj": inference_config.mask_obj,
                "config.baselines_config_baselines": baselines_config.baselines,
                "config.baselines_config_bc_epochs": baselines_config.bc_epochs,
                "config.baselines_config_adv_its": baselines_config.adv_its,
                "config.rl_config_policy_latent_dim": rl_config.policy_latent_dim,
                "config.rl_config_policy_num_layers": rl_config.policy_num_layers,
            }
        )

    # filters["config.inference_config_baselines_only"] = inference_config.baselines_only
    # if hasattr(model_config, "output_type"):
    #     filters["config.model_config_output_type"] = model_config.output_type
    wandb_project = "sri-evaluation"  # lol I love hardcoding so much, feels so good
    runs = api.runs(
        path=f"{general_config.wandb_entity}/{wandb_project}",
        filters=filters,
    )
    if most_recent_first:
        runs = list(reversed(runs))
    assert len(runs) > 0, "No runs found with these parameters"
    # if len(runs) == 0:
    big_int = int(
        1e18
    )  # anything bigger causes wandb to error, saying it's a float? bizarre
    cur_idx = 0
    eval_results = []
    for run in runs:
        if rl or baseline or "average train loss" in run.summary:
            # if train_config.minimal_synth and ("train_config_minimal_synth" not in run.config or not run.config["train_config_minimal_synth"]):
            #     continue
            # if train_config.synth_random_grip and ("train_config_synth_random_grip" not in run.config or not run.config["train_config_synth_random_grip"]):
            #     continue
            # for parameters we added later, we'll assume they were "false"
            # if they're not in the run config
            if not baseline:
                if (
                    "model_config_output_type" not in run.config
                    and model_config.output_type != "reward"
                ) or (
                    "model_config_output_type" in run.config
                    and run.config["model_config_output_type"] != model_config.output_type
                ):
                    continue
                if (
                    "train_config_minimal_synth" not in run.config
                    and train_config.minimal_synth
                ) or (
                    "train_config_minimal_synth" in run.config
                    and run.config["train_config_minimal_synth"]
                    != train_config.minimal_synth
                ):
                    continue
                if (
                    "train_config_synth_random_grip" not in run.config
                    and train_config.synth_random_grip
                ) or (
                    "train_config_synth_random_grip" in run.config
                    and run.config["train_config_synth_random_grip"]
                    != train_config.synth_random_grip
                ):
                    continue
            if hasattr(dataset_config, "goal_pos_adjustment_factor"):
                if (
                    "dataset_config_goal_pos_adjustment_factor" not in run.config
                    and dataset_config.goal_pos_adjustment_factor != 1.0
                ):
                    continue
                elif (
                    "dataset_config_goal_pos_adjustment_factor" in run.config
                    and run.config["dataset_config_goal_pos_adjustment_factor"]
                    != dataset_config.goal_pos_adjustment_factor
                ):
                    continue
            if hasattr(dataset_config, "goals_avoid_obj"):
                if (
                    "dataset_config_goals_avoid_obj" not in run.config
                    and dataset_config.goals_avoid_obj
                ):
                    continue
                elif (
                    "dataset_config_goals_avoid_obj" in run.config
                    and run.config["dataset_config_goals_avoid_obj"]
                    != dataset_config.goals_avoid_obj
                ):
                    continue
                if (
                    "dataset_config_goals_avoid_obj" in run.config
                    and run.config["dataset_config_goals_avoid_obj"]
                ):
                    if (
                        "dataset_config_goals_avoid_obj_xy" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_xy"]
                        != dataset_config.goals_avoid_obj_xy
                    ):
                        continue
                    if (
                        "dataset_config_goals_avoid_obj_dist" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_dist"]
                        != dataset_config.goals_avoid_obj_dist
                    ):
                        continue
                    if (
                        "dataset_config_goals_avoid_obj_pos" not in run.config
                        or run.config["dataset_config_goals_avoid_obj_pos"]
                        != dataset_config.goals_avoid_obj_pos
                    ):
                        continue
            if rl:
                if (
                    "rl_config_ensembling" not in run.config
                    or run.config["rl_config_ensembling"] != rl_config.ensembling
                ):
                    continue
        # if baseline and baselines_config.baselines[0] == "bc":
        #     breakpoint()
        if cur_idx in model_idxs:
            # eval_results.append(run.summary)
            history = run.history(
                keys=[
                    "ave_norm_tcp_closeness",
                    "ave_norm_tcp_dist",
                    "ave_norm_obj_closeness",
                    "ave_norm_obj_dist",
                ]
            )
            # Access the logged value from the history
            # The history is typically a pandas DataFrame, so you can select a specific row
            # For example, you can get the last logged value using .iloc[-1]:
            # last_value = history["ave_tcp_closeness"].iloc[-1]
            if "reach" in inference_config.env:
                if "ave_norm_tcp_closeness" not in history:
                    continue
                value_to_return = history["ave_norm_tcp_closeness"].iloc[-1]
            elif "pick-place" in inference_config.env:
                if "ave_norm_obj_closeness" not in history:
                    continue
                value_to_return = history["ave_norm_obj_closeness"].iloc[-1]
            else:
                raise NotImplementedError("we ain't got that environment")
            if unique:
                rl_run_name = run.config["loaded_rl_name"]
                if rl_run_name in rl_run_names_seen:
                    print(
                        f"Skipping {run.name} because policy run {rl_run_name} already seen"
                    )
                    continue
            # if not baseline:
            #     model_run_name = run.config["loaded_model_name"]
            #     if model_run_name in model_run_names_seen:
            #         print(
            #             f"Skipping {run.name} because SRI model run {model_run_name} already seen"
            #         )
            #         continue
            eval_results.append(value_to_return)
        cur_idx += 1
        if len(eval_results) == len(model_idxs):
            break
    if not eval_results:
        # raise ValueError("No run found with 2000 train loss history for the given indices")
        raise ValueError(
            f"Not enough runs found for model_idxs {model_idxs} with 2000 train history, only found {cur_idx}"
        )
    return eval_results


# model_config.demonstration_rep_dim,
# model_config.state_rep_dim,
# model_config.internal_tst_dim,
# model_config.state_hidden_size,
# model_config.reward_hidden_size,  # note that this isn't always used if mlp is false
# model_config.demonstration_hidden_size,
# obs_size,
# dem_obs_size,
# horizon,
# model_config.num_demonstration_layers,
# model_config.num_state_layers,
# mlp=model_config.mlp,
# dem_encoder_type=model_config.dem_encoder_type,
# direct_goal_inference=model_config.direct_goal_inference,


def update_wandb_with_namespaces_and_names(namespaces_and_names):
    config_dict = {}
    for namespace, name in namespaces_and_names:
        if namespace is not None:
            config_dict.update(namespace_to_dict(namespace, name))
        else:
            config_dict[name] = None
    wandb.config.update(config_dict)


def load_config_with_defaults(config_path):
    config_path = os.path.join("config", config_path)

    def recursive_load(config_path, loaded_configs):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            defaults_path = config.get("defaults-config")
            if defaults_path:
                defaults_path = os.path.join(
                    os.path.dirname(config_path), defaults_path
                )
                if defaults_path in loaded_configs:
                    raise ValueError(
                        f"Circular reference detected in config files: {defaults_path}"
                    )
                loaded_configs.add(defaults_path)
                defaults_config = recursive_load(defaults_path, loaded_configs)
                defaults_config.update(config)
                config = defaults_config
        config = {key.replace("-", "_"): value for key, value in config.items()}
        return config

    loaded_configs = set()
    config = recursive_load(config_path, loaded_configs)
    return argparse.Namespace(**config)


def get_reward_bins(
    env, num_bins, extra_success_reward, scale_rewards, proximity_reward=0.0
):
    if "reach" in env:
        reward_bounds = (
            0,
            10,
        )  # don't include extra_success_reward because there should only
        # be one bin for 10 to 10+extra_success_reward
    elif "pick-place" in env or "push" in env:
        reward_bounds = (
            0,
            6 + proximity_reward + max(1, proximity_reward),
        )
    else:
        raise NotImplementedError(
            "Rejection sampling not implemented for this environment"
        )
        # if (not rl_config.use_gt_reward and not rl_config.unscale_rewards) or (
        #     rl_config.scale_gt_reward and rl_config.use_gt_reward
        # ):
        # reward_bounds = (-3, 4)
        new_low = scale(reward_bounds[0], 0, 10 + extra_success_reward, -3, 3)
        new_high = scale(reward_bounds[1], 0, 10 + extra_success_reward, -3, 3)
        reward_bounds = (new_low, new_high)
    reward_bins = np.linspace(
        reward_bounds[0],
        reward_bounds[1],
        num_bins,
    )
    return reward_bins

def find_path(
    run_dirs,
    path_components,
    target_files,
    raise_on_missing=True,
):
    """
    Searches through all run_dirs for a path that contains all target_files.
    
    Parameters
    ----------
    run_dirs : list of str
        A list of directories (strings) in which to look for the dataset.
    path_components : list of str
        Additional path components that will be joined to each run_dir.
        For example, if you need to check something like:
            os.path.join(run_dir, "some_env", "rollouts", "dataset_name")
        you could pass path_components = ["some_env", "rollouts", "dataset_name"].
    target_files : list of str
        The filenames that must exist in the final path for it to be considered valid.
    raise_on_missing : bool
        Whether to raise a FileNotFoundError if the files are not found 
        in any of the run_dirs. Defaults to True.
        
    Returns
    -------
    str or None
        The path to the directory containing the target_files if found, 
        otherwise None (or raises a FileNotFoundError).
    """
    for run_dir in run_dirs:
        candidate_path = os.path.join(run_dir, *path_components)
        # Check if all target files exist in candidate_path
        if all(os.path.exists(os.path.join(candidate_path, f)) for f in target_files):
            return candidate_path
    
    if raise_on_missing:
        # Build a more informative error message
        not_found_str = os.path.join(*path_components)
        raise FileNotFoundError(
            f"Could not find the required files {target_files} "
            f"in any of the run_dirs: {run_dirs} under path '{not_found_str}'."
        )
    
    # Return None if we prefer not to raise an exception but simply signal "not found"
    return None
