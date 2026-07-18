import argparse
import os
import numpy as np
import ast
from dataclasses import asdict
from tqdm import tqdm
import wandb
import torch
from sri.reward_inference.train import (
    train as train_bid_model,
    load_model,
    load_data,
)
from sri.reward_inference.IL_evaluation import train_bc, train_adversarial
from sri.rl.train import train as train_policy
from sri.rl.train import make_env
from sri.pemirl import PEMIRLConfig, PEMIRLModel, save_adapted_policy
# no need to import hidden, you can just set env._partially_observable = True
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import SawyerPickPlaceV2Policy
from sri.utils import (
    namespace_to_dict,
    convert_rollouts_to_chai,
    get_dataset_name,
    update_wandb_with_namespaces_and_names,
    get_model_names,
    load_config_with_defaults,
    find_path,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import TQC
import pandas as pd
import imageio
import yaml

rng = np.random.default_rng()


def process_configs(
    general_config,
    dataset_config,
    obs_dataset_config,
    model_config,
    train_config,
    rl_config,
    inference_config,
    baselines_config,
    orl_dataset_config=None,
    obs_dataset_config_2=None,
):
    if (
        not train_config.skip_reward_training
        and train_config.env == "push"
        and train_config.synthesize_obs
    ):
        assert (
            train_config.z_bounds[1] < 0.03
        ), "Push environment goals must be on the table"
    if inference_config.env == "push":
        assert (
            inference_config.goal_z_bounds[1] < 0.03
        ), "Push environment goals must be on the table"
    if rl_config.render_duration < 0:
        rl_config.render_duration = inference_config.horizon
    assert (
        rl_config.render_interval > rl_config.render_duration
    ), "render_interval must be greater than render_duration"
    if train_config.num_obs is None:
        assert train_config.dem_obs_share_goals

    # if not train_config.skip_reward_training:
    # train_config.env = train_config.env + "-v2-goal-observable"
    # inference_config.env = inference_config.env + "-v2-goal-observable"
    # dataset_config.env = dataset_config.env + "-v2-goal-observable"
    # obs_dataset_config.env = obs_dataset_config.env + "-v2-goal-observable"
    # if orl_dataset_config is not None:
    #     orl_dataset_config.env = orl_dataset_config.env + "-v2-goal-observable"
    # if obs_dataset_config_2 is not None:
    #     obs_dataset_config_2.env = obs_dataset_config_2.env + "-v2-goal-observable"
    if "-v2-goal-observable" not in train_config.env:
        train_config.env = train_config.env + "-v2-goal-observable"
    if "-v2-goal-observable" not in inference_config.env:
        inference_config.env = inference_config.env + "-v2-goal-observable"
    if "-v2-goal-observable" not in dataset_config.env:
        dataset_config.env = dataset_config.env + "-v2-goal-observable"
    if "-v2-goal-observable" not in obs_dataset_config.env:
        obs_dataset_config.env = obs_dataset_config.env + "-v2-goal-observable"
    if orl_dataset_config is not None:
        if "-v2-goal-observable" not in orl_dataset_config.env:
            orl_dataset_config.env = orl_dataset_config.env + "-v2-goal-observable"
    if obs_dataset_config_2 is not None:
        if "-v2-goal-observable" not in obs_dataset_config_2.env:
            obs_dataset_config_2.env = obs_dataset_config_2.env + "-v2-goal-observable"

    if inference_config.mask_obj:
        assert (
            "pick-place" not in inference_config.env
        ), "Can't mask object for pick-place"

    if (
        "bc" in baselines_config.baselines
        or "gail" in baselines_config.baselines
        or "airl" in baselines_config.baselines
        or "pemirl" in baselines_config.baselines
    ):
        assert (
            inference_config.include_actions
        ), "Must include actions for BC, GAIL, AIRL, and PEMIRL baselines"

    inference_config.num_envs = inference_config.batch_size

    if inference_config.num_goals != -1:
        inference_config.batch_size = min(
            inference_config.batch_size, inference_config.num_goals
        )

    assert (
        not hasattr(dataset_config, "dataset_path")
        or dataset_config.dataset_path is None
    ), "Not doing this anymore"
    assert (
        not hasattr(obs_dataset_config, "dataset_path")
        or obs_dataset_config.dataset_path is None
    ), "Not doing this anymore"
    if orl_dataset_config is not None:
        assert (
            not hasattr(orl_dataset_config, "dataset_path")
            or orl_dataset_config.dataset_path is None
        ), "Not doing this anymore"
    if obs_dataset_config_2 is not None:
        assert (
            not hasattr(obs_dataset_config_2, "dataset_path")
            or obs_dataset_config_2.dataset_path is None
        ), "Not doing this anymore"
    assert (
        not hasattr(model_config, "model_path") or model_config.model_path is None
    ), "Not doing this anymore"

    if inference_config.use_goal is not None:
        assert (
            inference_config.num_goals == 1
        ), "Can only use a specific goal for inference if num_goals is 1"
        inference_config.use_goal = torch.Tensor(inference_config.use_goal)

    if not train_config.skip_reward_training and train_config.synthesize_obs:
        max_grid_points = train_config.synth_grid_size**3
        if train_config.synth_edges_only:
            max_grid_points = (
                6 * train_config.synth_grid_size**2
                - 12 * train_config.synth_grid_size
                + 8
            )
        num_obs = (
            train_config.num_obs
            if train_config.num_obs is not None
            else train_config.n * (dataset_config.horizon + 1)
        )
        assert (
            train_config.num_obs is None or not train_config.dem_obs_share_goals
        ), "limited obs only supported for randomly selected obs"
        max_synth_prop = max_grid_points / num_obs
        adjusted_synth_prop = min(train_config.synth_prop, max_synth_prop)
        if adjusted_synth_prop < train_config.synth_prop:
            print(
                f"Reducing synth_prop from {train_config.synth_prop} to {adjusted_synth_prop} to fit within grid size"
            )
        train_config.synth_prop = adjusted_synth_prop

    if train_config.grasp_rew_only:
        assert (
            inference_config.skip_all_inference or inference_config.only_rl
        ) and not inference_config.baselines_only, (
            "Can't do inference with grasp_rew_only"
        )
    if not inference_config.skip_all_inference:
        assert (
            train_config.grasp_rew_only == rl_config.grasp_rew_only
        ), "grasp_rew_only must be the same for training and inference"
        assert not (
            inference_config.reinit_obj_pos and inference_config.multi_obj_pos
        ), "Can't do both reinit_obj_pos and multi_obj_pos"
    if not inference_config.baselines_only:
        inference_config.n = train_config.n
        print(
            f"Setting inference_config.n to {inference_config.n} = train_config.n for SRI"
        )


def update_config_from_args(config, args_str, config_name):
    if args_str is not None:
        for kv_pair in args_str.split(","):
            if "=" not in kv_pair:
                raise ValueError(f"Invalid {config_name} argument: {kv_pair}")
            key, value = kv_pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            if hasattr(config, key):
                attr = getattr(config, key)
                attr_type = type(attr)

                # Handle booleans explicitly
                if attr_type is bool:
                    if value.lower() in ("true", "1"):
                        value = True
                    elif value.lower() in ("false", "0"):
                        value = False
                    else:
                        raise ValueError(f"Invalid boolean value for {key}: {value}")
                else:
                    try:
                        # Try to evaluate the value (handles numbers, lists, etc.)
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        # If evaluation fails, keep it as a string
                        pass

                    # Attempt to cast the value to the type of the existing attribute
                    try:
                        if attr_type is not str:
                            value = attr_type(value)
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Cannot cast value of {key} to {attr_type.__name__}"
                        )

                setattr(config, key, value)
            else:
                raise AttributeError(f"{config_name} has no attribute '{key}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised reward inference models"
    )
    parser.add_argument(
        "--general-config",
        type=str,
        default="general/default_experiment.yml",
        help="General configuration file",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="datasets/default.yml",
        help="Demonstration dataset configuration file",
    )
    parser.add_argument(
        "--obs-dataset-config",
        type=str,
        default="datasets/default.yml",
        help="Observation dataset configuration file",
    )
    parser.add_argument(
        "--obs-dataset-config-2",
        type=str,
        default=None,
        help="Second observation dataset configuration file",
    )
    parser.add_argument(
        "--orl-dataset-config",
        type=str,
        default=None,
        help="Offline RL dataset configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="model/default.yml",
        help="Model configuration file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="train/default.yml",
        help="Training configuration file",
    )
    parser.add_argument(
        "--rl-config", type=str, default="rl/no_orl.yml", help="RL configuration file"
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default="inference/default.yml",
        help="Inference configuration file",
    )
    parser.add_argument(
        "--baselines-config",
        type=str,
        default="baselines/default.yml",
        help="Baselines configuration file",
    )
    parser.add_argument(
        "--dataset-idx", type=int, default=0, help="Index of dataset to use"
    )
    parser.add_argument(
        "--obs-dataset-idx",
        type=int,
        default=0,
        help="Index of observation dataset to use",
    )
    parser.add_argument(
        "--obs-dataset-2-idx",
        type=int,
        default=None,
        help="Index of second observation dataset to use",
    )
    parser.add_argument(
        "--orl-dataset-idx",
        type=int,
        default=0,
        help="Index of offline RL dataset to use",
    )
    parser.add_argument(
        "--model-idxs",  # note that --model-idx will also work because of argparse abbreviation
        type=int,
        nargs="+",
        default=[0],
        help="Index of model(s) to use",
    )
    parser.add_argument(
        "--general-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override general config parameters",
    )
    parser.add_argument(
        "--dataset-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override dataset config parameters",
    )
    parser.add_argument(
        "--obs-dataset-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override observation dataset config parameters",
    )
    parser.add_argument(
        "--orl-dataset-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override offline RL dataset config parameters",
    )
    parser.add_argument(
        "--obs-dataset-2-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override second observation dataset config parameters",
    )
    parser.add_argument(
        "--model-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override model config parameters",
    )
    parser.add_argument(
        "--train-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override train config parameters",
    )
    parser.add_argument(
        "--rl-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override RL config parameters",
    )
    parser.add_argument(
        "--inference-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override inference config parameters",
    )
    parser.add_argument(
        "--baselines-args",
        type=str,
        default=None,
        help="Comma-separated key=value pairs to override baselines config parameters",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Explicit policy/checkpoint path to use for evaluation-time inference",
    )

    args = parser.parse_args()

    general_config = load_config_with_defaults(args.general_config)
    dataset_config = load_config_with_defaults(args.dataset_config)
    obs_dataset_config = load_config_with_defaults(args.obs_dataset_config)
    if args.orl_dataset_config is not None:
        orl_dataset_config = load_config_with_defaults(args.orl_dataset_config)
    else:
        orl_dataset_config = None
    if args.obs_dataset_config_2 is not None:
        obs_dataset_config_2 = load_config_with_defaults(args.obs_dataset_config_2)
    else:
        obs_dataset_config_2 = None
    model_config = load_config_with_defaults(args.model_config)
    train_config = load_config_with_defaults(args.train_config)
    rl_config = load_config_with_defaults(args.rl_config)
    inference_config = load_config_with_defaults(args.inference_config)
    baselines_config = load_config_with_defaults(args.baselines_config)

    update_config_from_args(general_config, args.general_args, "general")
    update_config_from_args(dataset_config, args.dataset_args, "dataset")
    update_config_from_args(obs_dataset_config, args.obs_dataset_args, "obs_dataset")
    if args.orl_dataset_args is not None:
        update_config_from_args(
            orl_dataset_config, args.orl_dataset_args, "orl_dataset"
        )
    if args.obs_dataset_2_args is not None:
        update_config_from_args(
            obs_dataset_config_2, args.obs_dataset_2_args, "obs_dataset_2"
        )
    update_config_from_args(model_config, args.model_args, "model")
    update_config_from_args(train_config, args.train_args, "train")
    update_config_from_args(rl_config, args.rl_args, "rl")
    update_config_from_args(inference_config, args.inference_args, "inference")
    update_config_from_args(baselines_config, args.baselines_args, "baselines")

    # if args.noise_coeff is not None:
    #     dataset_config.noise_coeff = args.noise_coeff
    # if args.obs_noise_coeff is not None:
    #     obs_dataset_config.noise_coeff = args.obs_noise_coeff
    # if obs_dataset_config_2 is not None and args.obs_noise_coeff is not None:
    #     print("Warning: not setting noise coefficient for second observation dataset")
    # if args.orl_noise_coeff is not None and orl_dataset_config is not None:
    #     orl_dataset_config.noise_coeff = args.orl_noise_coeff
    # if args.goal_pos_adjustment_factor is not None:
    #     dataset_config.goal_pos_adjustment_factor = args.goal_pos_adjustment_factor

    process_configs(
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
        orl_dataset_config=orl_dataset_config,
        obs_dataset_config_2=obs_dataset_config_2,
    )

    if rl_config.ensembling is not None:
        assert len(args.model_idxs) > 1, "Ensembling requires multiple models"
        assert len(args.model_idxs) == len(
            set(args.model_idxs)
        ), "Model indices must be distinct"
    else:
        assert (
            len(args.model_idxs) == 1
        ), "Only one model can be can be used for RL without ensembling"

    if inference_config.evaluation:
        assert len(args.model_idxs) == 1, "Can only evaluate one model at a time"

    return (
        args.dataset_idx,
        args.obs_dataset_idx,
        args.obs_dataset_2_idx,
        args.orl_dataset_idx,
        args.model_idxs,
        args.policy_path,
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
        orl_dataset_config,
        obs_dataset_config_2,
    )

class BIDPolicy:
    # Mainly just used for BID-A and BID-G
    # (BID-R policies are SB3 policies trained with RL)
    def __init__(self, oracle_policy=None, state_encoder=None, final_layer=None, dem_rep_dim=None):
        assert (
            (oracle_policy is not None or
            (state_encoder is not None and final_layer is not None)) and
            (oracle_policy is None) != (state_encoder is None)
        ), "Exactly one of oracle_policy or state_encoder and final_layer must be provided"
        self.oracle_policy = oracle_policy
        self.state_encoder = state_encoder
        self.final_layer = final_layer
        self.dem_rep_dim = dem_rep_dim

    def predict(self, obs, deterministic=True):
        # obs: (num_envs, obs_dim)
        # Last n elements of obs are the goal
        if self.oracle_policy is not None:
            obs_torch = torch.tensor(obs, device='cpu')
            actions = self.oracle_policy.get_action_batch(obs_torch)
            actions = actions.cpu().numpy()
        else:
            device = next(self.state_encoder.parameters()).device
            dtype = next(self.state_encoder.parameters()).dtype
            enc_input = obs[:, :-self.dem_rep_dim]
            enc_input = torch.tensor(enc_input, device=device, dtype=dtype)
            enc = self.state_encoder(enc_input)
            dem_rep = obs[:, -self.dem_rep_dim:]
            dem_rep = torch.tensor(dem_rep, device=device, dtype=dtype)
            actions = self.final_layer(dem_rep, enc)
            actions = actions.cpu().numpy()
        return actions, None

def inference(
    env,
    policy_path,
    general_config,
    inference_config,
    rl_config,
    model_config=None,
    policy=None,
):
    print(f"Using SubprocVecEnv with {env.num_envs} environments")

    # set environment horizon
    # env.set_attr("max_path_length", inference_config.horizon)

    if policy is None:
        assert policy_path is not None, "Need a policy to do evaluation"
        if inference_config.baselines_only:
            policy_class = PPO
        else:
            policy_classes = {
                "ppo": PPO,
                "sac": SAC,
                "tqc": TQC,
            }
            policy_class = policy_classes[rl_config.algorithm.lower()]
        # note that we're using env = None here; they say you can do it for inference, but we should test and make sure
        # policy = policy_class.load(policy_path)
        try:
            policy = policy_class.load(policy_path, env=env)
        except AssertionError:
            policy = ActorCriticPolicy.load(policy_path)

    succeeded_at_end = np.zeros(
        (
            # len(dataloader),
            inference_config.num_envs,
            inference_config.episodes,
        ),
        dtype=bool,
    )
    succeeded_at_any_point = np.zeros(
        (
            inference_config.num_envs,
            inference_config.episodes,
        ),
        dtype=bool,
    )
    tcp_centers = np.zeros(
        (
            inference_config.num_envs,
            inference_config.episodes,
            inference_config.horizon + 1,
            3,
        ),
        dtype=float,
    )
    obj_positions = np.zeros(
        (
            inference_config.num_envs,
            inference_config.episodes,
            inference_config.horizon + 1,
            3,
        ),
        dtype=float,
    )
    goals = np.zeros(
        (
            inference_config.num_envs,
            inference_config.episodes,
            3,
        ),
        dtype=float,
    )
    success = np.zeros(
        (
            inference_config.num_envs,
            inference_config.episodes,
            inference_config.horizon + 1,
        ),
    )
    # env.set_attr("_freeze_rand_vec",
    #              True)  # Don't want new object positions every time
    episodes = range(inference_config.episodes)
    with tqdm(episodes, unit="episode") as tepoch:
        for ep in tepoch:
            obs = env.reset()
            for env_idx in range(env.num_envs):
                goals[env_idx, ep] = env.get_attr("_target_pos", indices=[env_idx])[0]
                tcp_centers[env_idx, ep, 0] = env.get_attr(
                    "tcp_center", indices=[env_idx]
                )[0]
                if inference_config.include_extra_reward_info:
                    obj_positions[env_idx, ep, 0] = obs[env_idx, 25:28]
                elif inference_config.include_partial_reward_info:
                    obj_positions[env_idx, ep, 0] = obs[env_idx, 13:16]
                else:
                    obj_positions[env_idx, ep, 0] = obs[env_idx, 4:7]
            if inference_config.render:
                ep_imgs = []
                for env_idx in range(env.num_envs):
                    ep_imgs.append([env.env_method("render", indices=[env_idx])[0]])
            for step in range(inference_config.horizon):
                action, _ = policy.predict(obs, deterministic=True)
                next_obs, _, dones, infos = env.step(action)
                if inference_config.render:
                    for env_idx in range(env.num_envs):
                        ep_imgs[env_idx].append(
                            env.env_method("render", indices=[env_idx])[0]
                        )
                for env_idx in range(env.num_envs):
                    succeeded_at_any_point[env_idx, ep] = (
                        succeeded_at_any_point[env_idx, ep] or infos[env_idx]["success"]
                    )
                    success[env_idx, ep, step] = infos[env_idx]["success"]
                    tcp_centers[env_idx, ep, step + 1] = infos[env_idx]["tcp_center"]
                if step < inference_config.horizon - 1:
                    assert (
                        not dones.any()
                    ), "Environment should not be done before max_steps"
                    # check that target pos's are still the same
                    for env_idx in range(env.num_envs):
                        assert np.allclose(
                            goals[env_idx, ep],
                            env.get_attr("_target_pos", indices=[env_idx])[0],
                        ), "Goal position should not change"
                        # obj_positions[env_idx, ep, step + 1] = next_obs[env_idx, 25:28]
                        if inference_config.include_extra_reward_info:
                            obj_positions[env_idx, ep, step + 1] = next_obs[
                                env_idx, 25:28
                            ]
                        elif inference_config.include_partial_reward_info:
                            obj_positions[env_idx, ep, step + 1] = next_obs[
                                env_idx, 13:16
                            ]
                        else:
                            obj_positions[env_idx, ep, step + 1] = next_obs[
                                env_idx, 4:7
                            ]
                elif step == inference_config.horizon - 1:
                    assert dones.all(), "Environment should be done at max_steps"
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
                    this_ep_obs = np.array(
                        [info_dict["terminal_observation"] for info_dict in infos]
                    )

                    for env_idx in range(env.num_envs):
                        succeeded_at_end[env_idx, ep] = infos[env_idx]["success"]
                        # obj_positions[env_idx, ep, step + 1] = this_ep_obs[
                        #     env_idx, 25:28
                        # ]
                        if inference_config.include_extra_reward_info:
                            obj_positions[env_idx, ep, step + 1] = this_ep_obs[
                                env_idx, 25:28
                            ]
                        elif inference_config.include_partial_reward_info:
                            obj_positions[env_idx, ep, step + 1] = this_ep_obs[
                                env_idx, 13:16
                            ]
                        else:
                            obj_positions[env_idx, ep, step + 1] = this_ep_obs[
                                env_idx, 4:7
                            ]
                        # print(f"Goal {goal_batch[env_idx]} succeeded: {infos[env_idx]['success']}")
                # if step == inference_config.horizon - 2:
                #     # imgs_as_array = [env.render(env_idx) for env_idx in range(env.num_envs)]
                #     # imgs_as_array = env.env_method("render")
                #     # imgs_as_array = []
                #     for env_idx in range(env.num_envs):
                #     #     imgs_as_array.append(env.env_method("render", indices=[env_idx]))
                #         print(f"Object position: {env.env_method('_get_pos_objects', indices=[env_idx])}")
                #         print(f"Set goal position: {goal_batch[env_idx]}")
                #         print(f"Observed goal position: {env.get_attr('_target_pos', indices=[env_idx])}")
                obs = next_obs
            if inference_config.render:
                os.makedirs(
                    f"scratch/runs/{wandb.run.name}",
                    exist_ok=True,
                )
                for env_idx in range(env.num_envs):
                    if not succeeded_at_end[env_idx, ep]:
                        # print(f"Goal {goal_batch[env_idx]} failed")
                        # imageio.imsave(f"runs/{wandb.run.name}/goal_{goal_batch[env_idx]}_failed.png", env.render(env_idx))
                        # img_as_array = env.env_method("render", indices=[e nv_idx])
                        # img_as_array = imgs_as_array[env_idx]
                        img_as_array = ep_imgs[env_idx]
                        imageio.mimsave(
                            f"scratch/runs/{wandb.run.name}/env_{env_idx}_ep_{ep}_failed.mp4",
                            img_as_array,
                        )
                    else:
                        # print(f"Goal {goal_batch[env_idx]} succeeded")
                        # imageio.imsave(f"runs/{wandb.run.name}/goal_{goal_batch[env_idx]}_succeeded.png", env.render(env_idx))
                        # img_as_array = env.env_method("render", indices=[env_idx])
                        # img_as_array = imgs_as_array[env_idx]
                        img_as_array = ep_imgs[env_idx]
                        imageio.mimsave(
                            f"scratch/runs/{wandb.run.name}/env_{env_idx}_ep_{ep}_succeeded.mp4",
                            img_as_array,
                        )
    (
        ave_norm_tcp_dist,
        ave_norm_tcp_closeness,
        ave_norm_obj_dist,
        ave_norm_obj_closeness,
    ) = calc_dists(tcp_centers, obj_positions, goals)
    ave_success = np.mean(success)
    if general_config.verbose:
        print(f"Average success rate at end: {np.mean(succeeded_at_end)}")
        print(f"Average success rate at any point: {np.mean(succeeded_at_any_point)}")
        print(f"Average success per step: {ave_success}")
        print(f"Average normalized TCP distance: {ave_norm_tcp_dist}")
        print(f"Average normalized TCP closeness: {ave_norm_tcp_closeness}")
        print(f"Average normalized object distance: {ave_norm_obj_dist}")
        print(f"Average normalized object closeness: {ave_norm_obj_closeness}")
    return (
        succeeded_at_end,
        succeeded_at_any_point,
        ave_success,
        ave_norm_tcp_dist,
        ave_norm_tcp_closeness,
        ave_norm_obj_dist,
        ave_norm_obj_closeness,
    )


def calc_dists(tcp_centers, obj_positions, goals, success_radius=0.05):
    # # Calculate initial distances
    # initial_tcp_dist = np.linalg.norm(tcp_centers[:, :, 0] - goals, axis=-1)
    # initial_obj_dist = np.linalg.norm(obj_positions[:, :, 0] - goals, axis=-1)

    # # Calculate distances across all steps at once using broadcasting
    # tcp_dists = np.linalg.norm(tcp_centers - goals[:, :, np.newaxis], axis=-1)
    # obj_dists = np.linalg.norm(obj_positions - goals[:, :, np.newaxis], axis=-1)

    # # Normalize distances
    # norm_tcp_dist = tcp_dists / initial_tcp_dist[:, :, np.newaxis]
    # norm_obj_dist = obj_dists / initial_obj_dist[:, :, np.newaxis]

    # # Closeness is 1 - normalized distance
    # norm_tcp_closeness = 1 - norm_tcp_dist
    # norm_obj_closeness = 1 - norm_obj_dist

    # # Calculate average normalized distances and closeness
    # ave_norm_tcp_dist = np.mean(norm_tcp_dist)
    # ave_norm_tcp_closeness = np.mean(norm_tcp_closeness)
    # ave_norm_obj_dist = np.mean(norm_obj_dist)
    # ave_norm_obj_closeness = np.mean(norm_obj_closeness)

    # return (
    #     ave_norm_tcp_dist,
    #     ave_norm_tcp_closeness,
    #     ave_norm_obj_dist,
    #     ave_norm_obj_closeness,
    # )
    # Calculate initial distances
    initial_tcp_dist = np.linalg.norm(tcp_centers[:, :, 0] - goals, axis=-1)
    initial_obj_dist = np.linalg.norm(obj_positions[:, :, 0] - goals, axis=-1)

    # Calculate distances across all steps using broadcasting
    tcp_dists = np.linalg.norm(tcp_centers - goals[:, :, np.newaxis], axis=-1)
    obj_dists = np.linalg.norm(obj_positions - goals[:, :, np.newaxis], axis=-1)

    # Apply the success radius threshold: distances within success_radius are set to 0
    tcp_dists = np.where(tcp_dists <= success_radius, 0, tcp_dists)
    obj_dists = np.where(obj_dists <= success_radius, 0, obj_dists)

    # Normalize distances
    norm_tcp_dist = tcp_dists / initial_tcp_dist[:, :, np.newaxis]
    norm_obj_dist = obj_dists / initial_obj_dist[:, :, np.newaxis]

    # Closeness is 1 - normalized distance (clipped to ensure no negative values)
    # norm_tcp_closeness = np.clip(1 - norm_tcp_dist, 0, 1)
    # norm_obj_closeness = np.clip(1 - norm_obj_dist, 0, 1)
    # no clipping, let it be negative
    norm_tcp_closeness = 1 - norm_tcp_dist
    norm_obj_closeness = 1 - norm_obj_dist

    # Calculate average normalized distances and closeness
    ave_norm_tcp_dist = np.mean(norm_tcp_dist)
    ave_norm_tcp_closeness = np.mean(norm_tcp_closeness)
    ave_norm_obj_dist = np.mean(norm_obj_dist)
    ave_norm_obj_closeness = np.mean(norm_obj_closeness)

    return (
        ave_norm_tcp_dist,
        ave_norm_tcp_closeness,
        ave_norm_obj_dist,
        ave_norm_obj_closeness,
    )


def single_env_inference(env, policy, inference_config):
    succeeded_at_end = np.zeros((inference_config.episodes), dtype=bool)
    succeeded_at_any_point = np.zeros((inference_config.episodes), dtype=bool)
    obs = env.reset()
    for ep in tqdm(range(0, inference_config.episodes)):
        # set observed (predicted) goals
        if inference_config.render:
            ep_imgs = []
            for env_idx in range(env.num_envs):
                ep_imgs.append(
                    [
                        env.env_method(
                            "render", render_reward=False, indices=[env_idx]
                        )[0]
                    ]
                )
        for step in range(inference_config.horizon):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                oracle_actions = []
                for env_idx in range(env.num_envs):
                    oracle_actions.append(model.get_action(obs[env_idx], step))
                action = np.array(oracle_actions)
            next_obs, _, dones, infos = env.step(action)
            for env_idx in range(env.num_envs):
                succeeded_at_any_point[ep] = (
                    succeeded_at_any_point[ep] or infos[env_idx]["success"]
                )
            if step < inference_config.horizon - 1:
                assert (
                    not dones.any()
                ), "Environment should not be done before max_steps"
            elif step == inference_config.horizon - 1:
                assert dones.all(), "Environment should be done at max_steps"
                for env_idx in range(env.num_envs):
                    succeeded_at_end[ep] = infos[env_idx]["success"]
            obs = next_obs
            if inference_config.render:
                for env_idx in range(env.num_envs):
                    ep_imgs[env_idx].append(
                        env.env_method(
                            "render", render_reward=False, indices=[env_idx]
                        )[0]
                    )
        if inference_config.render:
            for env_idx in range(env.num_envs):
                if not succeeded_at_end[ep]:
                    print(f"Failed")
                    img_as_array = ep_imgs[env_idx]
                    imageio.mimsave(
                        f"scratch/runs/{wandb.run.name}/failed.mp4",
                        img_as_array,
                    )
                else:
                    print(f"Succeeded")
                    img_as_array = ep_imgs[env_idx]
                    imageio.mimsave(
                        f"scratch/runs/{wandb.run.name}/succeeded.mp4",
                        img_as_array,
                    )
    if general_config.verbose:
        print(f"Average success rate at end: {np.mean(succeeded_at_end)}")
        print(f"Average success rate at any point: {np.mean(succeeded_at_any_point)}")
    return succeeded_at_end, succeeded_at_any_point


class LimitedInferenceDataLoader:
    def __init__(
        self,
        dataloader,
        num_inference_goals,
        goal_to_return=None,
        goal_bounds_x=None,
        goal_bounds_y=None,
        goal_bounds_z=None,
    ):
        self.dataloader = dataloader
        # If num_inference_goals is -1, use the total number of items in the dataset
        if num_inference_goals == -1:
            self.num_inference_goals = len(dataloader.dataset)
        else:
            self.num_inference_goals = num_inference_goals

        self.batch_size = dataloader.batch_size
        self.goal_to_return = goal_to_return
        self.goal_bounds_x = goal_bounds_x
        self.goal_bounds_y = goal_bounds_y
        self.goal_bounds_z = goal_bounds_z
        if self.goal_bounds_x is not None:
            num_goals_init = len(dataloader.dataset.goals)
            assert (
                self.goal_bounds_y is not None and self.goal_bounds_z is not None
            ), "Must specify all goal bounds"
            goal_within_bounds_idxs = [
                idx
                for idx, goal in enumerate(dataloader.dataset.goals)
                if self.goal_bounds_x[0] <= goal[0] <= self.goal_bounds_x[1]
                and self.goal_bounds_y[0] <= goal[1] <= self.goal_bounds_y[1]
                and self.goal_bounds_z[0] <= goal[2] <= self.goal_bounds_z[1]
            ]
            # now filter
            self.dataloader.dataset.goals = [
                dataloader.dataset.goals[idx] for idx in goal_within_bounds_idxs
            ]
            self.dataloader.dataset.dems = [
                dataloader.dataset.dems[idx] for idx in goal_within_bounds_idxs
            ]
            if self.dataloader.dataset.actions is not None:
                self.dataloader.dataset.actions = [
                    dataloader.dataset.actions[idx] for idx in goal_within_bounds_idxs
                ]
            num_goals_filtered = len(dataloader.dataset.goals)
            assert num_goals_filtered >= self.num_inference_goals, "Not enough goals"
            print(
                f"Started with {num_goals_init} goals, filtered based on bounds to {num_goals_filtered}"
            )
            # num_goals_within_bounds = len(
            #     [
            #         goal
            #         for goal in dataloader.dataset.goals
            #         if self.goal_bounds_x[0] <= goal[0] <= self.goal_bounds_x[1]
            #         and self.goal_bounds_y[0] <= goal[1] <= self.goal_bounds_y[1]
            #         and self.goal_bounds_z[0] <= goal[2] <= self.goal_bounds_z[1]
            #     ]
            # )
            # assert (
            #     num_goals_within_bounds >= self.num_inference_goals
            # ), "Not enough goals within bounds"
        if self.goal_to_return is not None:
            if type(self.goal_to_return) == list:
                self.goal_to_return = torch.Tensor(self.goal_to_return)
            assert (
                self.num_inference_goals == 1
            ), "Can only return a specific goal if num_inference_goals is 1"
            assert (
                self.batch_size == 1
            ), "Can only return a specific goal if batch size is 1"
            assert any(
                torch.allclose(goal, self.goal_to_return, atol=1e-6)
                for goal in self.dataloader.dataset.goals
            ), "goal_to_return is not in dataloader's goals"
            # move to same device as dataset
            self.goal_to_return = self.goal_to_return.to(
                next(iter(dataloader))[-1].device
            )
        # Calculate the number of batches, considering partial batches if num_inference_goals is less than a full batch size
        self.total_batches = (
            self.num_inference_goals + self.batch_size - 1
        ) // self.batch_size

    def __iter__(self):
        goals_yielded = 0
        # if self.goal_bounds_x is not None:
        #     num_goals_filtered = 0
        for data in self.dataloader:
            # print("Considering goals", [goal for goal in data[2]])
            # Determine how many items to yield from this batch
            items_to_yield = min(
                self.num_inference_goals - goals_yielded, self.batch_size
            )
            if items_to_yield <= 0:
                break

            # If we need to yield a partial batch, slice the data accordingly
            if items_to_yield < self.batch_size:
                ret = tuple(d[:items_to_yield] for d in data)
                # if isinstance(data, tuple) or isinstance(data, list):
                # else:
                #     # Assuming 'data' is a tensor or ndarray
                #     ret = data[:items_to_yield]
            else:
                ret = data
            if self.goal_to_return is not None:
                _, goal_batch = ret
                if not torch.allclose(goal_batch[0], self.goal_to_return, atol=1e-6):
                    continue
            # if self.goal_bounds_x is not None:
            #     _, _, goal_batch = ret
            #     filter_goal = False
            #     if not (
            #         self.goal_bounds_x[0] <= goal_batch[0, 0] <= self.goal_bounds_x[1]
            #     ):
            #         filter_goal = True
            #     if not (
            #         self.goal_bounds_y[0] <= goal_batch[0, 1] <= self.goal_bounds_y[1]
            #     ):
            #         filter_goal = True
            #     if not (
            #         self.goal_bounds_z[0] <= goal_batch[0, 2] <= self.goal_bounds_z[1]
            #     ):
            #         filter_goal = True
            yield ret

            goals_yielded += items_to_yield
            if goals_yielded >= self.num_inference_goals:
                break

    def __len__(self):
        return self.total_batches


def sri(
    models,
    inference_dataloader,
    orl_dataset_path,
    general_config,
    model_config,
    train_config,
    rl_config,
    inference_config,
    policy_path=None,
    goal_to_use=None,
):
    if rl_config.full_legacy:
        print("################## Legacy RL ##################")
        # skip all this silly SRI stuff, just run RL
        policy = train_policy(
            general_config,
            # model_config,
            inference_config,
            rl_config,
            # model.state_encoder,
            # reward_mlp,
            # dem_reps,
            # goals,
            # orl_dataset_path,
            # model_config.limit_reward_obs,
        )
    # first, set model to eval mode and turn off gradients
    print("Preparing model(s) for inference")
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        if not rl_config.gpu_state_encoder:
            model.state_encoder.cpu()
            if model_config.mlp:
                model.reward_layer.cpu()
    # Wrap the inference dataloader in case we want to use a subset of the data,
    # such as for not doing goal-conditioned RL
    # (if num_inference_goals is -1, use the whole dataset)
    print(f"Using {inference_config.num_goals} goals for inference")
    if inference_config.filter_inference_goals:
        goal_x_bounds = inference_config.goal_x_bounds
        goal_y_bounds = inference_config.goal_y_bounds
        goal_z_bounds = inference_config.goal_z_bounds
    else:
        goal_x_bounds = None
        goal_y_bounds = None
        goal_z_bounds = None
    inference_dataloader = LimitedInferenceDataLoader(
        inference_dataloader,
        inference_config.num_goals,
        goal_to_use,
        goal_x_bounds,
        goal_y_bounds,
        goal_z_bounds,
    )
    print("################## Inference ##################")
    # Precompute demonstration representations (roughly equivalent to goals)
    print("Processing demonstration representations")
    all_dem_reps = []
    for model in models:
        device = next(model.demonstration_encoder.parameters()).device
        dem_reps = torch.zeros(
            (
                len(inference_dataloader) * inference_config.batch_size,
                model_config.demonstration_rep_dim,
            )
        ).to(device)
        goals = np.zeros(
            (len(inference_dataloader) * inference_config.batch_size, 3)
        )
        batch_idx = 0
        with tqdm(inference_dataloader, unit="batch") as tepoch:
            for dem_batch, goal_batch in tepoch:
                dem_reps[
                    batch_idx
                    * inference_config.batch_size : (batch_idx + 1)
                    * inference_config.batch_size
                ] = model.demonstration_encoder(
                    dem_batch[:, : train_config.n].contiguous()
                ).squeeze()
                goals[
                    batch_idx
                    * inference_config.batch_size : (batch_idx + 1)
                    * inference_config.batch_size
                ] = (goal_batch.detach().cpu().numpy())
                batch_idx += 1
        all_dem_reps.append(dem_reps)
    if inference_config.num_goals == 1:
        print("Using goal", goals[0])
        wandb.run.summary.update({"goal": goals[0]})
        wandb.run.summary.update({"demonstration_representation": dem_reps[0]})
    if model_config.mlp:
        final_mlps = [model.final_layer for model in models]
    else:
        final_mlps = None
    if model_config.output_type != "goal":
        state_encoders = [model.state_encoder for model in models]
    else:
        state_encoders = None
    if policy_path is None and model_config.output_type == "reward":
        print("Training policy")
        policy = train_policy(
            general_config,
            inference_config,
            rl_config,
            state_encoders,
            final_mlps,
            all_dem_reps,
            goals,
            orl_dataset_path,
            model_config.limit_reward_obs,
        )
    else:
        policy = None
    if inference_config.only_rl:
        return
    reinit_goals = rl_config.full_legacy and rl_config.full_legacy_reinit_goals
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[inference_config.env]
    env = SubprocVecEnv(
        [
            make_env(
                env_cls,
                inference_config,
                state_encoders,
                final_mlps,
                all_dem_reps,
                goals,
                model_config.limit_reward_obs,
                reinit_goals=reinit_goals,
                extra_success_reward=rl_config.extra_success_reward,
                unscale_rewards=rl_config.unscale_rewards,
                use_gt_reward=rl_config.use_gt_reward,
                scale_gt_reward=rl_config.scale_gt_reward,
                use_gt_goal=rl_config.use_gt_goal,
                no_task_rep=rl_config.no_task_rep,
                proximity_reward=rl_config.proximity_reward,
                success_requires_touch=rl_config.success_requires_touch,
                third_gt=rl_config.third_gt,
                half_gt=rl_config.half_gt,
                ensembling=rl_config.ensembling,
                reinit_obj_pos=inference_config.reinit_obj_pos,
                random_hand_starts=inference_config.random_hand_starts,
                include_extra_reward_info=inference_config.include_extra_reward_info,
                include_partial_reward_info=inference_config.include_partial_reward_info,
            )
            for _ in range(inference_config.num_envs)
        ]
    )
    env.set_attr("max_path_length", inference_config.horizon)

    if model_config.output_type == "goal":
        # Make BIDPolicy with oracle_policy
        if "pick-place" in inference_config.env:
            oracle_policy = SawyerPickPlaceV2Policy()
        elif "reach" in inference_config.env:
            oracle_policy = SawyerReachV2Policy()
        else:
            raise NotImplementedError(
                f"Oracle policy not implemented for {inference_config.env}"
            )
        policy = BIDPolicy(oracle_policy=oracle_policy)
    elif model_config.output_type == "action":
        # In this case the model itself is the policy;
        # the demonstration representations already get included
        # in the environment, so we just need to pass the
        # state encoder and final layer
        policy = BIDPolicy(
            state_encoder=state_encoders[0],
            final_layer=final_mlps[0],
            dem_rep_dim=model_config.demonstration_rep_dim,
        )
    elif model_config.output_type == "reward":
        policy = None # policy will be loaded from file in inference
    else:
        raise NotImplementedError(
            f"How the heck did we get here without complaining about the model_config.output_type? {model_config.output_type}"
        )
    (
        succeeded_at_end,
        succeeded_at_any_point,
        ave_success,
        ave_norm_tcp_dist,
        ave_norm_tcp_closeness,
        ave_norm_obj_dist,
        ave_norm_obj_closeness,
    ) = inference(
        env,
        policy_path,
        general_config,
        inference_config,
        rl_config,
        model_config,
        policy=policy,
    )
    succeeded_at_end_table = wandb.Table(
        dataframe=pd.DataFrame(
            # succeeded_at_end.reshape(-1, succeeded_at_end.shape[-1])
            succeeded_at_end
        )
    )
    succeeded_at_any_point_table = wandb.Table(
        dataframe=pd.DataFrame(
            # succeeded_at_any_point.reshape(-1, succeeded_at_any_point.shape[-1])
            succeeded_at_any_point
        )
    )
    # wandb.run.summary.update({"succeeded_at_end": succeeded_at_end, "succeeded_at_any_point": succeeded_at_any_point})
    wandb.log(
        {
            "succeeded_at_end": succeeded_at_end_table,
            "succeeded_at_any_point": succeeded_at_any_point_table,
            "ave_success": ave_success,
            "ave_norm_tcp_dist": ave_norm_tcp_dist,
            "ave_norm_tcp_closeness": ave_norm_tcp_closeness,
            "ave_norm_obj_dist": ave_norm_obj_dist,
            "ave_norm_obj_closeness": ave_norm_obj_closeness,
        }
    )

    # also save to runs/wandb.run.name/succeeded_at_end.npy and succeeded_at_any_point.npy
    os.makedirs(f"scratch/runs/{wandb.run.name}", exist_ok=True)
    np.save(f"scratch/runs/{wandb.run.name}/succeeded_at_end.npy", succeeded_at_end)
    np.save(
        f"scratch/runs/{wandb.run.name}/succeeded_at_any_point.npy", succeeded_at_any_point
    )


def get_chai_rollouts_from_dataloader(
    dataloader, n, include_extra_reward_info, include_partial_reward_info, mask_obj
):
    dataset = dataloader.dataset
    observations = dataset.dems.clone()
    observations = observations[:, :n].contiguous()
    actions = dataset.actions.clone()
    actions = actions[:, :n].contiguous()
    assert not (
        include_extra_reward_info and include_partial_reward_info
    ), "Can't include both extra and partial reward info"
    if mask_obj:
        # # obs[4:7] = obs[22:25] = np.array([0, 0.6, 0.02], dtype=obs.dtype)
        # obs[7:11] = obs[25:29] = np.array([0, 0, 0, 1], dtype=obs.dtype)
        observations[..., 25:28] = observations[..., 43:46] = torch.tensor(
            [0, 0.6, 0.02], dtype=observations.dtype
        )
        observations[..., 28:32] = observations[..., 46:50] = torch.tensor(
            [0, 0, 0, 1], dtype=observations.dtype
        )
    if not include_extra_reward_info and not include_partial_reward_info:
        observations = observations[..., 21:]
    elif include_partial_reward_info:
        # exclude 6:12, 15:21
        observations = torch.cat(
            [
                observations[..., :6],
                observations[..., 12:15],
                observations[..., 21:],
            ],
            dim=-1,
        )

    return convert_rollouts_to_chai(observations, actions)


def run_baselines(
    inference_dataloader,
    general_config,
    inference_config,
    baselines_config,
    rl_config,
    policy_path,
    goal_to_use,
):
    if (
        "bc" in baselines_config.baselines
        or "gail" in baselines_config.baselines
        or "airl" in baselines_config.baselines
        or "pemirl" in baselines_config.baselines
    ):
        print("Getting Imitation-style rollouts for BC/GAIL/AIRL/PEMIRL")
        rollouts = get_chai_rollouts_from_dataloader(
            inference_dataloader,
            inference_config.n,
            inference_config.include_extra_reward_info,
            inference_config.include_partial_reward_info,
            inference_config.mask_obj,
        )
        print("Num tasks", len(rollouts))
        print("Num rollouts per task", len(rollouts[0]))
        print("Rollout shape", rollouts[0][0].obs.shape)
        print("Action shape", rollouts[0][0].acts.shape)
        gt_goals = inference_dataloader.dataset.goals.clone().numpy()

    if "random_rew" in baselines_config.baselines:
        print("Running random reward baseline")
        raise NotImplementedError
        # succeeded_at_end, succeeded_at_any_point = inference(
        #     None,
        #     inference_dataloader,
        #     None,
        #     general_config,
        #     inference_config,
        #     random_goal=True,
        # )
        succeeded_at_end_table = wandb.Table(
            dataframe=pd.DataFrame(
                succeeded_at_end.reshape(-1, succeeded_at_end.shape[-1])
            )
        )
        succeeded_at_any_point_table = wandb.Table(
            dataframe=pd.DataFrame(
                succeeded_at_any_point.reshape(-1, succeeded_at_any_point.shape[-1])
            )
        )
        wandb.log(
            {
                "random_rew_succeeded_at_end": succeeded_at_end_table,
                "random_rew_succeeded_at_any_point": succeeded_at_any_point_table,
            }
        )
        os.makedirs(f"scratch/runs/{wandb.run.name}/random_rew", exist_ok=True)
        np.save(
            f"scratch/runs/{wandb.run.name}/random_rew/succeeded_at_end.npy", succeeded_at_end
        )
        np.save(
            f"scratch/runs/{wandb.run.name}/random_rew/succeeded_at_any_point.npy",
            succeeded_at_any_point,
        )
    if "ground_truth" in baselines_config.baselines:
        print("Running ground truth baseline")
        raise NotImplementedError("This isn't how we run GT baseline")
        # succeeded_at_end, succeeded_at_any_point = inference(
        #     None,
        #     inference_dataloader,
        #     None,
        #     general_config,
        #     inference_config,
        #     ground_truth=True,
        # )
        succeeded_at_end_table = wandb.Table(
            dataframe=pd.DataFrame(
                succeeded_at_end.reshape(-1, succeeded_at_end.shape[-1])
            )
        )
        succeeded_at_any_point_table = wandb.Table(
            dataframe=pd.DataFrame(
                succeeded_at_any_point.reshape(-1, succeeded_at_any_point.shape[-1])
            )
        )
        wandb.log(
            {
                "ground_truth_succeeded_at_end": succeeded_at_end_table,
                "ground_truth_succeeded_at_any_point": succeeded_at_any_point_table,
            }
        )
        os.makedirs(f"scratch/runs/{wandb.run.name}/ground_truth", exist_ok=True)
        np.save(
            f"scratch/runs/{wandb.run.name}/ground_truth/succeeded_at_end.npy", succeeded_at_end
        )
        np.save(
            f"scratch/runs/{wandb.run.name}/ground_truth/succeeded_at_any_point.npy",
            succeeded_at_any_point,
        )

    def _build_pemirl_config() -> PEMIRLConfig:
        task_batch_size = getattr(baselines_config, "pemirl_task_batch_size", 16)
        return PEMIRLConfig(
            latent_dim=getattr(baselines_config, "pemirl_latent_dim", 3),
            meta_batch_size=getattr(baselines_config, "pemirl_meta_batch_size", 50),
            task_batch_size=task_batch_size,
            discrim_updates=getattr(baselines_config, "pemirl_discrim_updates", 20),
            pretrain_epochs=getattr(baselines_config, "pemirl_pretrain_epochs", 1000),
            info_coeff=getattr(baselines_config, "pemirl_info_coeff", 0.1),
            imitation_coeff=getattr(baselines_config, "pemirl_imitation_coeff", 0.01),
            entropy_weight=getattr(baselines_config, "pemirl_entropy_weight", 1.0),
            discount=getattr(baselines_config, "pemirl_discount", 0.99),
            trpo_step_size=getattr(baselines_config, "pemirl_trpo_step_size", 0.01),
            fusion_buffer_size=getattr(baselines_config, "pemirl_fusion_buffer_size", 100),
            fusion_subsample_ratio=getattr(
                baselines_config, "pemirl_fusion_subsample_ratio", 0.5
            ),
            policy_hidden_sizes=tuple(
                getattr(baselines_config, "pemirl_policy_hidden_sizes", [64, 64])
            ),
            context_hidden_sizes=tuple(
                getattr(baselines_config, "pemirl_context_hidden_sizes", [128, 128])
            ),
            reward_hidden_size=getattr(baselines_config, "pemirl_reward_hidden_size", 32),
            value_hidden_size=getattr(baselines_config, "pemirl_value_hidden_size", 32),
            meta_train_iters=getattr(baselines_config, "pemirl_meta_train_iters", 200),
            adapt_iters=getattr(baselines_config, "pemirl_adapt_iters", 200),
            seed=getattr(baselines_config, "pemirl_seed", 0),
            trpo_debug_nonfinite=getattr(
                baselines_config, "pemirl_trpo_debug_nonfinite", False
            ),
            trpo_reject_nonfinite_steps=getattr(
                baselines_config, "pemirl_trpo_reject_nonfinite_steps", True
            ),
            meta_train_rollout_budget=getattr(
                baselines_config, "pemirl_meta_train_rollout_budget", -1
            ),
            on_policy_training=getattr(
                baselines_config, "pemirl_on_policy_training", True
            ),
            on_policy_adaptation=getattr(
                baselines_config, "pemirl_on_policy_adaptation", True
            ),
            rollout_horizon=getattr(baselines_config, "pemirl_rollout_horizon", -1),
            adapt_rollouts_per_iter=getattr(
                baselines_config, "pemirl_adapt_rollouts_per_iter", task_batch_size
            ),
            adapt_deterministic_policy=getattr(
                baselines_config, "pemirl_adapt_deterministic_policy", False
            ),
        )

    def _apply_pemirl_rollout_budget(rollouts_by_task, goals_by_task, budget, seed):
        budget = int(budget)
        if budget <= 0:
            return (
                rollouts_by_task,
                np.asarray(goals_by_task),
                sum(len(task) for task in rollouts_by_task),
            )

        total_rollouts = sum(len(task) for task in rollouts_by_task)
        if total_rollouts <= budget:
            return rollouts_by_task, np.asarray(goals_by_task), total_rollouts

        flat_refs = []
        for task_idx, task_rollouts in enumerate(rollouts_by_task):
            for traj_idx in range(len(task_rollouts)):
                flat_refs.append((task_idx, traj_idx))

        rng = np.random.default_rng(seed)
        selected_refs = rng.choice(len(flat_refs), size=budget, replace=False)
        by_task = [[] for _ in range(len(rollouts_by_task))]
        for ref_idx in selected_refs.tolist():
            task_idx, traj_idx = flat_refs[int(ref_idx)]
            by_task[task_idx].append(rollouts_by_task[task_idx][traj_idx])

        # Drop empty tasks: PEMIRL batching requires at least one trajectory per task.
        filtered = []
        filtered_goals = []
        for task_idx, task_rollouts in enumerate(by_task):
            if len(task_rollouts) == 0:
                continue
            filtered.append(task_rollouts)
            filtered_goals.append(np.asarray(goals_by_task[task_idx]))
        actual = sum(len(task) for task in filtered)
        return filtered, np.asarray(filtered_goals), actual

    def _eval_adapted_policy(adapted_policy, gt_goal):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[inference_config.env]
        env = SubprocVecEnv(
            [
                make_env(
                    env_cls,
                    inference_config,
                    use_gt_goal=False,
                    goals=np.expand_dims(gt_goal, 0),
                    include_extra_reward_info=inference_config.include_extra_reward_info,
                    include_partial_reward_info=inference_config.include_partial_reward_info,
                    mask_obj=inference_config.mask_obj,
                    reinit_obj_pos=inference_config.reinit_obj_pos,
                    random_hand_starts=inference_config.random_hand_starts,
                )
                for _ in range(inference_config.num_envs)
            ]
        )
        env = VecMonitor(env)
        env.set_attr("max_path_length", inference_config.horizon)
        if not inference_config.reinit_obj_pos:
            env.set_attr("_freeze_rand_vec", True)
        else:
            env.set_attr("_freeze_rand_vec", False)
        env.env_method("_set_target_pos", gt_goal, set_last_rand_vec=True)
        return inference(
            env,
            None,
            general_config,
            inference_config,
            rl_config,
            policy=adapted_policy,
        )

    def run_pemirl():
        pemirl_cfg = _build_pemirl_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pemirl_dir = f"scratch/runs/{wandb.run.name}/pemirl"
        os.makedirs(pemirl_dir, exist_ok=True)
        os.makedirs(os.path.join(pemirl_dir, "adapted_policies"), exist_ok=True)

        def _collect_pemirl_rollouts(
            policy,
            context,
            goal,
            num_rollouts,
            horizon,
            deterministic=False,
        ):
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[inference_config.env]
            env = make_env(
                env_cls,
                inference_config,
                use_gt_goal=False,
                goals=np.expand_dims(np.asarray(goal, dtype=np.float32), 0),
                include_extra_reward_info=inference_config.include_extra_reward_info,
                include_partial_reward_info=inference_config.include_partial_reward_info,
                mask_obj=inference_config.mask_obj,
                reinit_obj_pos=inference_config.reinit_obj_pos,
                random_hand_starts=inference_config.random_hand_starts,
            )()
            if hasattr(env, "max_path_length"):
                env.max_path_length = int(horizon)
            if hasattr(env, "_freeze_rand_vec"):
                env._freeze_rand_vec = not inference_config.reinit_obj_pos

            out_paths = []
            context_t = torch.as_tensor(context, dtype=torch.float32, device=device).reshape(1, -1)
            try:
                for _ in range(int(num_rollouts)):
                    if hasattr(env, "_set_target_pos"):
                        try:
                            env._set_target_pos(
                                np.asarray(goal, dtype=np.float32), set_last_rand_vec=True
                            )
                        except TypeError:
                            env._set_target_pos(np.asarray(goal, dtype=np.float32))

                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    obs = np.asarray(obs, dtype=np.float32)

                    obs_buf, next_obs_buf, act_buf, logp_buf, rew_buf = [], [], [], [], []
                    for _step in range(int(horizon)):
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        policy_input = torch.cat([obs_t, context_t], dim=-1)
                        with torch.no_grad():
                            action_t, log_prob_t = policy.sample(
                                policy_input, deterministic=bool(deterministic)
                            )
                        action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

                        step_out = env.step(action)
                        if len(step_out) == 5:
                            next_obs, reward, terminated, truncated, info = step_out
                            done = bool(terminated or truncated)
                        else:
                            next_obs, reward, done, info = step_out
                        if isinstance(next_obs, tuple):
                            next_obs = next_obs[0]
                        next_obs = np.asarray(next_obs, dtype=np.float32)

                        obs_buf.append(obs.copy())
                        next_obs_buf.append(next_obs.copy())
                        act_buf.append(action.copy())
                        logp_buf.append(float(log_prob_t.squeeze().detach().cpu()))
                        rew_buf.append(float(reward))
                        obs = next_obs
                        if done:
                            break

                    if len(obs_buf) == 0:
                        continue

                    if len(obs_buf) < int(horizon):
                        pad_n = int(horizon) - len(obs_buf)
                        obs_buf.extend([obs_buf[-1].copy() for _ in range(pad_n)])
                        next_obs_buf.extend([next_obs_buf[-1].copy() for _ in range(pad_n)])
                        act_buf.extend([np.zeros_like(act_buf[-1], dtype=np.float32) for _ in range(pad_n)])
                        logp_buf.extend([0.0 for _ in range(pad_n)])
                        rew_buf.extend([0.0 for _ in range(pad_n)])

                    out_paths.append(
                        {
                            "observations": np.asarray(obs_buf, dtype=np.float32),
                            "next_observations": np.asarray(next_obs_buf, dtype=np.float32),
                            "actions": np.asarray(act_buf, dtype=np.float32),
                            "log_probs": np.asarray(logp_buf, dtype=np.float32).reshape(-1, 1),
                            "env_rewards": np.asarray(rew_buf, dtype=np.float32),
                        }
                    )
            finally:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

            if len(out_paths) == 0:
                raise RuntimeError("PEMIRL rollout collector failed to collect any paths.")
            return out_paths

        if not inference_config.evaluation:
            print("Training PEMIRL baseline")
            train_rollouts, train_goals, used_rollouts = _apply_pemirl_rollout_budget(
                rollouts,
                gt_goals,
                pemirl_cfg.meta_train_rollout_budget,
                pemirl_cfg.seed,
            )
            print(
                "PEMIRL meta-train rollout budget",
                pemirl_cfg.meta_train_rollout_budget,
                "-> using",
                used_rollouts,
                "rollouts across",
                len(train_rollouts),
                "tasks",
            )
            wandb.run.summary.update(
                {
                    "pemirl_meta_train_rollout_budget": pemirl_cfg.meta_train_rollout_budget,
                    "pemirl_meta_train_rollouts_used": used_rollouts,
                    "pemirl_meta_train_tasks_used": len(train_rollouts),
                    "pemirl_on_policy_training": bool(pemirl_cfg.on_policy_training),
                    "pemirl_on_policy_adaptation": bool(pemirl_cfg.on_policy_adaptation),
                }
            )
            model = PEMIRLModel.from_rollouts(train_rollouts, pemirl_cfg, device=device)
            train_metrics = model.train_meta(
                train_rollouts,
                task_goals=train_goals,
                rollout_collector=_collect_pemirl_rollouts,
            )
            ckpt_path = os.path.join(pemirl_dir, "meta_checkpoint.pt")
            model.save_checkpoint(ckpt_path, train_metrics)

            with open(os.path.join(pemirl_dir, "adapt_config.yaml"), "w") as f:
                yaml.safe_dump(asdict(pemirl_cfg), f)
            np.savez(
                os.path.join(pemirl_dir, "metrics.npz"),
                cent_loss=np.array(train_metrics.cent_loss),
                info_loss=np.array(train_metrics.info_loss),
                info_surr_loss=np.array(train_metrics.info_surr_loss),
                imitation_loss=np.array(train_metrics.imitation_loss),
                total_loss=np.array(train_metrics.total_loss),
                trpo_kl=np.array(train_metrics.trpo_kl),
                trpo_accepted=np.array(train_metrics.trpo_accepted),
                avg_return=np.array(train_metrics.avg_return),
            )

            num_goals = len(gt_goals) if inference_config.num_goals == -1 else min(
                inference_config.num_goals, len(gt_goals)
            )
            for idx, task_rollouts in enumerate(rollouts[:num_goals]):
                adapted_policy, adapt_stats = model.adapt(
                    task_rollouts,
                    adapt_iters=pemirl_cfg.adapt_iters,
                    goal=gt_goals[idx],
                    rollout_collector=_collect_pemirl_rollouts,
                )
                save_path = os.path.join(pemirl_dir, "adapted_policies", f"{idx}.pt")
                save_adapted_policy(save_path, adapted_policy)
                if idx == 0:
                    save_adapted_policy(
                        os.path.join(pemirl_dir, "adapted_policy.pt"), adapted_policy
                    )
                    wandb.run.summary.update(
                        {
                            "pemirl_adapt_final_loss": adapt_stats.final_loss,
                            "pemirl_context_norm": adapt_stats.context_norm,
                        }
                    )
            print(f"Saved PEMIRL artifacts to {pemirl_dir}")
            return

        if policy_path is None:
            policy_path_local = os.path.join(
                "scratch", "runs", wandb.run.name, "pemirl", "meta_checkpoint.pt"
            )
        else:
            policy_path_local = policy_path
        if not os.path.exists(policy_path_local):
            raise FileNotFoundError(
                f"PEMIRL checkpoint not found at {policy_path_local}"
            )
        print(f"Loading PEMIRL checkpoint from {policy_path_local}")
        model = PEMIRLModel.load_checkpoint(policy_path_local, map_location=device)
        pemirl_cfg = model.config
        with open(os.path.join(pemirl_dir, "adapt_config.yaml"), "w") as f:
            yaml.safe_dump(asdict(pemirl_cfg), f)

        num_goals = len(gt_goals) if inference_config.num_goals == -1 else min(
            inference_config.num_goals, len(gt_goals)
        )
        indices = list(range(num_goals))
        if goal_to_use is not None:
            matching = [
                i for i in indices if np.allclose(goal_to_use, gt_goals[i], atol=1e-4)
            ]
            if matching:
                indices = [matching[0]]
        for idx in indices:
            task_rollouts = rollouts[idx]
            gt_goal = gt_goals[idx]
            adapted_policy, adapt_stats = model.adapt(
                task_rollouts,
                adapt_iters=pemirl_cfg.adapt_iters,
                goal=gt_goal,
                rollout_collector=_collect_pemirl_rollouts,
            )
            save_path = os.path.join(pemirl_dir, "adapted_policies", f"{idx}.pt")
            save_adapted_policy(save_path, adapted_policy)
            if len(indices) == 1:
                save_adapted_policy(
                    os.path.join(pemirl_dir, "adapted_policy.pt"), adapted_policy
                )
            (
                succeeded_at_end,
                succeeded_at_any_point,
                ave_success,
                ave_norm_tcp_dist,
                ave_norm_tcp_closeness,
                ave_norm_obj_dist,
                ave_norm_obj_closeness,
            ) = _eval_adapted_policy(adapted_policy, gt_goal)
            wandb.log(
                {
                    "succeeded_at_end": wandb.Table(dataframe=pd.DataFrame(succeeded_at_end)),
                    "succeeded_at_any_point": wandb.Table(
                        dataframe=pd.DataFrame(succeeded_at_any_point)
                    ),
                    "ave_success": ave_success,
                    "ave_norm_tcp_dist": ave_norm_tcp_dist,
                    "ave_norm_tcp_closeness": ave_norm_tcp_closeness,
                    "ave_norm_obj_dist": ave_norm_obj_dist,
                    "ave_norm_obj_closeness": ave_norm_obj_closeness,
                    "pemirl_adapt_final_loss": adapt_stats.final_loss,
                    "pemirl_context_norm": adapt_stats.context_norm,
                }
            )
            np.savez(
                os.path.join(pemirl_dir, "metrics.npz"),
                succeeded_at_end=succeeded_at_end,
                succeeded_at_any_point=succeeded_at_any_point,
                ave_success=ave_success,
                ave_norm_tcp_dist=ave_norm_tcp_dist,
                ave_norm_tcp_closeness=ave_norm_tcp_closeness,
                ave_norm_obj_dist=ave_norm_obj_dist,
                ave_norm_obj_closeness=ave_norm_obj_closeness,
            )
            os.makedirs(f"scratch/runs/{wandb.run.name}", exist_ok=True)
            np.save(
                f"scratch/runs/{wandb.run.name}/succeeded_at_end.npy", succeeded_at_end
            )
            np.save(
                f"scratch/runs/{wandb.run.name}/succeeded_at_any_point.npy",
                succeeded_at_any_point,
            )

    def run_imitation(method):
        succeeded_at_end, succeeded_at_any_point = [], []
        num_goals = inference_config.num_goals
        for task_rollouts, gt_goal in tqdm(
            zip(rollouts[:num_goals], gt_goals[:num_goals]),
            total=len(gt_goals[:num_goals]),
        ):
            if goal_to_use is not None:
                assert np.allclose(
                    goal_to_use, gt_goal, atol=1e-4
                ), "Goal to use must match the ground truth goal"
            print("Now using GT goal", gt_goal)
            wandb.run.summary.update({"goal": gt_goal})
            env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[inference_config.env]
            if not inference_config.evaluation:
                print(
                    "Training policy, but not evaluating (remember, we don't do both at once anymore)"
                )
                # env = DummyVecEnv([make_env(env_cls, args)])
                # env = VecMonitor(env)
                # env = make_env(
                #     env_cls,
                #     inference_config,
                #     use_gt_goal=False,
                #     include_extra_reward_info=inference_config.include_extra_reward_info,
                # )()
                env = SubprocVecEnv(
                    [
                        make_env(
                            env_cls,
                            inference_config,
                            use_gt_goal=False,
                            goals=np.expand_dims(gt_goal, 0),
                            include_extra_reward_info=inference_config.include_extra_reward_info,
                            include_partial_reward_info=inference_config.include_partial_reward_info,
                            mask_obj=inference_config.mask_obj,
                            reinit_obj_pos=inference_config.reinit_obj_pos,
                            random_hand_starts=inference_config.random_hand_starts,
                        )
                        for _ in range(baselines_config.num_envs)
                    ]
                )
                # env.set_attr("max_path_length", args.horizon)
                # env.set_attr("_freeze_rand_vec", True)
                # # set goal
                # env.env_method("_set_target_pos", gt_goal)
                env.set_attr("max_path_length", inference_config.horizon)
                if not inference_config.reinit_obj_pos:
                    env.set_attr("_freeze_rand_vec", True)
                else:
                    env.set_attr("_freeze_rand_vec", False)
                # for env_idx in range(env.num_envs):
                #     env.env_method(
                #         "_set_target_pos", gt_goal, indices=[env_idx]
                #     )
                env.env_method("_set_target_pos", gt_goal, set_last_rand_vec=True)
                print("Training policy")
                if method.lower() == "bc":
                    policy = train_bc(
                        task_rollouts, env, baselines_config.bc_epochs, rl_config
                    )
                elif method.lower() in ("gail", "airl"):
                    policy = train_adversarial(
                        task_rollouts,
                        env,
                        method,
                        baselines_config.adv_its,
                        False,
                        rl_config,
                    )
                else:
                    raise ValueError(f"Unknown method {method}")
                # save_path = f"runs/{wandb.run.name}/{method}/policy.zip"
                save_path = f"scratch/runs/{wandb.run.name}/{method}/policy.zip"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                policy.save(save_path)
                print(f"Policy saved to {save_path}")
            else:
                print("Evaluating policy")
                # still making a vecenv for this one though
                # don't want to change inference code
                # env = DummyVecEnv([
                #     make_env(
                #         env_cls,
                #         inference_config,
                #         use_gt_goal=False,
                #         include_extra_reward_info=inference_config.
                #         include_extra_reward_info,
                #         include_partial_reward_info=inference_config.
                #         include_partial_reward_info,
                #         mask_obj=inference_config.mask_obj,
                #         reinit_obj_pos=inference_config.reinit_obj_pos,
                #     )
                # ])
                env = SubprocVecEnv(
                    [
                        make_env(
                            env_cls,
                            inference_config,
                            use_gt_goal=False,
                            goals=np.expand_dims(gt_goal, 0),
                            include_extra_reward_info=inference_config.include_extra_reward_info,
                            include_partial_reward_info=inference_config.include_partial_reward_info,
                            mask_obj=inference_config.mask_obj,
                            reinit_obj_pos=inference_config.reinit_obj_pos,
                            random_hand_starts=inference_config.random_hand_starts,
                        )
                        for _ in range(inference_config.num_envs)
                    ]
                )
                env = VecMonitor(env)
                env.set_attr("max_path_length", inference_config.horizon)
                if not inference_config.reinit_obj_pos:
                    env.set_attr("_freeze_rand_vec", True)
                else:
                    env.set_attr("_freeze_rand_vec", False)
                env.env_method("_set_target_pos", gt_goal, set_last_rand_vec=True)
                #     se, sa = single_env_inference(env, policy, inference_config)
                #     se = se.squeeze()
                #     sa = sa.squeeze()
                #     succeeded_at_end.append(se)
                #     succeeded_at_any_point.append(sa)
                # succeeded_at_end = np.array(succeeded_at_end)
                # succeeded_at_any_point = np.array(succeeded_at_any_point)
                # succeeded_at_end_table = wandb.Table(
                #     dataframe=pd.DataFrame(succeeded_at_end))
                # succeeded_at_any_point_table = wandb.Table(
                #     dataframe=pd.DataFrame(succeeded_at_any_point))
                # wandb.log({
                #     f"{method}_succeeded_at_end":
                #     succeeded_at_end_table,
                #     f"{method}_succeeded_at_any_point":
                #     succeeded_at_any_point_table,
                # })
                # os.makedirs(f"runs/{wandb.run.name}/{method}", exist_ok=True)
                # np.save(f"runs/{wandb.run.name}/{method}/succeeded_at_end.npy",
                #         succeeded_at_end)
                # np.save(
                #     f"runs/{wandb.run.name}/{method}/succeeded_at_any_point.npy",
                #     succeeded_at_any_point,
                # )
                (
                    succeeded_at_end,
                    succeeded_at_any_point,
                    ave_success,
                    ave_norm_tcp_dist,
                    ave_norm_tcp_closeness,
                    ave_norm_obj_dist,
                    ave_norm_obj_closeness,
                ) = inference(
                    env,
                    policy_path,
                    general_config,
                    inference_config,
                    rl_config,
                )
                succeeded_at_end_table = wandb.Table(
                    dataframe=pd.DataFrame(
                        # succeeded_at_end.reshape(-1, succeeded_at_end.shape[-1])
                        succeeded_at_end
                    )
                )
                succeeded_at_any_point_table = wandb.Table(
                    dataframe=pd.DataFrame(
                        # succeeded_at_any_point.reshape(-1, succeeded_at_any_point.shape[-1])
                        succeeded_at_any_point
                    )
                )
                # wandb.run.summary.update({"succeeded_at_end": succeeded_at_end, "succeeded_at_any_point": succeeded_at_any_point})
                wandb.log(
                    {
                        "succeeded_at_end": succeeded_at_end_table,
                        "succeeded_at_any_point": succeeded_at_any_point_table,
                        "ave_success": ave_success,   
                        "ave_norm_tcp_dist": ave_norm_tcp_dist,
                        "ave_norm_tcp_closeness": ave_norm_tcp_closeness,
                        "ave_norm_obj_dist": ave_norm_obj_dist,
                        "ave_norm_obj_closeness": ave_norm_obj_closeness,
                    }
                )

                # also save to runs/wandb.run.name/succeeded_at_end.npy and succeeded_at_any_point.npy
                os.makedirs(f"scratch/runs/{wandb.run.name}", exist_ok=True)
                np.save(f"scratch/runs/{wandb.run.name}/succeeded_at_end.npy", succeeded_at_end)
                np.save(
                    f"scratch/runs/{wandb.run.name}/succeeded_at_any_point.npy",
                    succeeded_at_any_point,
                )

    if "bc" in baselines_config.baselines:
        print("Running BC baseline")
        run_imitation("bc")
    if "gail" in baselines_config.baselines:
        print("Running GAIL baseline")
        run_imitation("gail")
    if "airl" in baselines_config.baselines:
        print("Running AIRL baseline")
        run_imitation("airl")
    if "pemirl" in baselines_config.baselines:
        print("Running PEMIRL baseline")
        run_pemirl()


if __name__ == "__main__":
    (
        dataset_idx,
        obs_dataset_idx,
        obs_dataset_2_idx,
        orl_dataset_idx,
        model_idxs,
        policy_path_override,
        general_config,
        dataset_config,
        obs_dataset_config,
        model_config,
        train_config,
        rl_config,
        inference_config,
        baselines_config,
        orl_dataset_config,
        obs_dataset_config_2,
    ) = parse_args()

    # Initialize wandb and log the combined config dictionary
    wandb.init(
        entity=general_config.wandb_entity,
        project=general_config.wandb_project,
        sync_tensorboard=True,
    )
    namespaces_and_names = [
        (general_config, "general_config"),
        (dataset_config, "dataset_config"),
        (obs_dataset_config, "obs_dataset_config"),
        (obs_dataset_config_2, "obs_dataset_config_2"),
        (orl_dataset_config, "orl_dataset_config"),
        (model_config, "model_config"),
        (train_config, "train_config"),
        (rl_config, "rl_config"),
        (inference_config, "inference_config"),
        (baselines_config, "baselines_config"),
    ]
    update_wandb_with_namespaces_and_names(namespaces_and_names)
    # we need the inference dataloader no matter what
    # but we'll add this argument if we only want to run baselines:
    if inference_config.baselines_only:
        train_config.skip_reward_training = True
        model_config.load_model = False
        # this ensures that we just get the dataloader and don't train the reward model
    if inference_config.evaluation:
        assert (model_config.load_model or inference_config.baselines_only
                ), "Must load model to evaluate RL (need shuffle idxs)"
        print(
            "Setting skip_reward_training to True due to inference_config.evaluation being True"
        )
        train_config.skip_reward_training = True
    use_explicit_baseline_policy_path = (
        inference_config.evaluation
        and inference_config.baselines_only
        and policy_path_override is not None
    )
    if use_explicit_baseline_policy_path:
        print(f"Using explicit baseline policy path: {policy_path_override}")
    print("################## Training ##################")

    if not model_config.load_model and not inference_config.evaluation:
        # we need to load datasets manually, since we don't have a model to tell us their names
        saved_model_dirs = None
        saved_model_dir = None
        policy_run = None
        policy_run_name = None
        dataset_run_name = get_dataset_name(dataset_idx,
                                            general_config,
                                            dataset_config,
                                            include_goal_params=True)
        print(f"Found dataset run name {dataset_run_name}")
        obs_dataset_run_name = get_dataset_name(
            obs_dataset_idx,
            general_config,
            obs_dataset_config,
            include_goal_params=True,
        )
        print(f"Found obs dataset run name {obs_dataset_run_name}")
        if obs_dataset_config_2 is not None:
            obs_dataset_2_run_name = get_dataset_name(
                obs_dataset_2_idx,
                general_config,
                obs_dataset_config_2,
                include_goal_params=True,
            )
            print(f"Found obs dataset 2 run name {obs_dataset_2_run_name}")
        else:
            obs_dataset_2_run_name = None
        if rl_config.offline_rl and not inference_config.baselines_only:
            assert (
                orl_dataset_config is not None
            ), "Must provide ORL dataset config if you want to do offline RL"
            orl_dataset_run_name = get_dataset_name(
                orl_dataset_idx,
                general_config,
                orl_dataset_config,
                include_goal_params=False,
            )
        else:
            orl_dataset_run_name = None
            orl_dataset_path = None
    else:
        wandb_api = wandb.Api()
        if use_explicit_baseline_policy_path:
            assert (
                len(baselines_config.baselines) == 1
            ), "Can only eval one baseline at a time with --policy-path"
            policy_run = None
            policy_run_name = None
            saved_model_dirs = None
            saved_model_dir = None
            dataset_run_name = get_dataset_name(
                dataset_idx, general_config, dataset_config, include_goal_params=True
            )
            obs_dataset_run_name = get_dataset_name(
                obs_dataset_idx,
                general_config,
                obs_dataset_config,
                include_goal_params=True,
            )
            if obs_dataset_config_2 is not None:
                obs_dataset_2_run_name = get_dataset_name(
                    obs_dataset_2_idx,
                    general_config,
                    obs_dataset_config_2,
                    include_goal_params=True,
                )
            else:
                obs_dataset_2_run_name = None
            orl_dataset_run_name = None
            orl_dataset_path = None
        else:
            if inference_config.evaluation and model_config.output_type == "reward":
                # first load RL run, then load model from RL run
                # get RL run name
                algo_name = ("RL" if not inference_config.baselines_only else
                             baselines_config.baselines)
                print(f"Evaluating {algo_name}")
                print(f"Loading {algo_name} run name")
                run_names = get_model_names(
                    general_config,
                    model_config,
                    train_config,
                    dataset_config,
                    obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                    rl=True if not inference_config.baselines_only else False,
                    rl_config=rl_config,
                    baseline=inference_config.baselines_only,
                    baselines_config=baselines_config,
                    inference_config=inference_config,
                )
                policy_run_name = run_names[0]
                project = (train_config.wandb_project
                           if not inference_config.baselines_only else
                           baselines_config.wandb_project)
                policy_runs = wandb_api.runs(
                    f"{general_config.wandb_entity}/{project}",
                    {"display_name": policy_run_name},
                )
                policy_run = policy_runs[0]
                print(f"Loaded {algo_name} run name", policy_run_name)
                policy_run_config = policy_run.config
                # get dataset run names (just for sanity check)
                policy_dataset_run_name = policy_run_config["dataset_run_name"]
                # now get model run name
                if not inference_config.baselines_only:
                    model_run_names = policy_run_config["loaded_model_names"]
                    model_run_name = model_run_names[0]
                    saved_model_dirs = [
                        find_path(general_config.run_dirs, [run_name, "models"],
                                  ["model.parameters"]) for run_name in model_run_names
                    ]
                    saved_model_dir = saved_model_dirs[0]
                    wandb.config.update({"loaded_model_names": model_run_names})
                    print("Loaded models from run names", model_run_names)
                wandb.config.update({"loaded_rl_name": policy_run_name})
            else:
                # load model directly
                assert not inference_config.baselines_only, "Must be doing RL to load model"
                policy_run = None
                policy_run_name = None
                run_names = get_model_names(
                    general_config,
                    model_config,
                    train_config,
                    dataset_config,
                    obs_dataset_config,
                    most_recent_first=True,
                    model_idxs=model_idxs,
                )
                # saved_model_dir = os.path.join("runs", run_names, "models")
                # saved_model_dirs = [
                #     os.path.join("runs", run_name, "models") for run_name in run_names
                # ]
                saved_model_dirs = [
                    find_path(general_config.run_dirs, [run_name, "models"],
                              ["model.parameters"]) for run_name in run_names
                ]
                saved_model_dir = saved_model_dirs[0]
                model_run_name = run_names[0]
                wandb.config.update({"loaded_model_names": run_names})
                print("Loaded models from run names", run_names)
            if not inference_config.baselines_only:
                # now load dataset run name from model run
                model_runs = wandb_api.runs(
                    f"{general_config.wandb_entity}/{train_config.wandb_project}",
                    {"display_name": model_run_name},
                )
                model_run = model_runs[0]
                dataset_run_name = model_run.config["dataset_run_name"]
                print(
                    f"Model run {model_run_name} has dataset run name {dataset_run_name}"
                )
                if inference_config.evaluation and model_config.output_type == "reward":
                    # assert (dataset_run_name == policy_dataset_run_name
                    #         ), "Dataset run names don't match"
                    if dataset_run_name != policy_dataset_run_name:
                        print("#" * 50)
                        print("WARNING: Dataset run names don't match")
                        print(
                            f"Model dataset {dataset_run_name} != RL dataset {policy_dataset_run_name}"
                        )
                        print(
                            "Shuffle idxs will be loaded from model run, but applied to RL dataset, as this should be what happened during RL training"
                        )
                        print(
                            "This should not cause data leakage, but you should check why this is happening (probably use of legacy dataset loading code)"
                        )
                        print(
                            "Note that if you did single task RL, you're in trouble if the shuffle idxs aren't what were originally used"
                        )
                        print("Switching dataset run to RL dataset run")
                        print("#" * 50)
                        dataset_run_name = policy_dataset_run_name
                obs_dataset_run_name = model_run.config["obs_dataset_run_name"]
                print(
                    f"Model run {model_run_name} has obs dataset run name {obs_dataset_run_name}"
                )
                if obs_dataset_config_2 is not None:
                    obs_dataset_2_run_name = model_run.config[
                        "obs_dataset_run_name_2"]
                    print(
                        f"Model run {model_run_name} has obs dataset 2 run name {obs_dataset_2_run_name}"
                    )
                else:
                    obs_dataset_2_run_name = None
                if rl_config.offline_rl:
                    orl_dataset_run_name = model_run.config["orl_dataset_run_name"]
                    print(
                        f"Model run {model_run_name} has orl dataset run name {orl_dataset_run_name}"
                    )
                else:
                    orl_dataset_run_name = None
            else:
                dataset_run_name = policy_dataset_run_name
                obs_dataset_run_name = policy_run_config["obs_dataset_run_name"]
                obs_dataset_2_run_name = None
                orl_dataset_run_name = None
                saved_model_dir = None

    wandb.config.update({"dataset_run_name": dataset_run_name})
    dataset_path = find_path(
        general_config.run_dirs,
        [dataset_config.env, "rollouts", dataset_run_name],
        ["observations.npy", "goals.npy"],
        raise_on_missing=True)
    wandb.config.update({"obs_dataset_run_name": obs_dataset_run_name})
    obs_dataset_path = find_path(
        general_config.run_dirs,
        [obs_dataset_config.env, "rollouts", obs_dataset_run_name],
        ["observations.npy", "goals.npy"],
        raise_on_missing=True)
    if obs_dataset_2_run_name is not None:
        wandb.config.update({"obs_dataset_run_name_2": obs_dataset_2_run_name})
        obs_dataset_path_2 = find_path(
            general_config.run_dirs,
            [obs_dataset_config_2.env, "rollouts", obs_dataset_2_run_name],
            ["observations.npy", "goals.npy"],
            raise_on_missing=True)
    else:
        obs_dataset_path_2 = None
    if orl_dataset_run_name is not None:
        wandb.config.update({"orl_dataset_run_name": orl_dataset_run_name})
        orl_dataset_path = find_path(
            general_config.run_dirs,
            [orl_dataset_config.env, "rollouts", orl_dataset_run_name],
            ["observations.npy", "goals.npy"],
            raise_on_missing=True)
    else:
        orl_dataset_path = None

    if not train_config.skip_reward_training:
        print("Training BID model")
        model, inference_dataloader = train_bid_model(
            dataset_path,
            obs_dataset_path,
            saved_model_dir,
            general_config,
            model_config,
            train_config,
            inference_config,
            obs_dataset_path_2=obs_dataset_path_2,
        )
        models = [model]
    else:
        print("Skipping reward training")
        if saved_model_dir is not None:
            assert (not inference_config.baselines_only
                    ), "Baselines don't have reward models"
            print(f"Loading shuffle idxs from {saved_model_dir}")
            # note that if we have multiple models this is the first one
            assert (
                inference_config.evaluation or inference_config.only_rl
            ), "Sorry, these are the only times we can load models right now"
            assert os.path.exists(
                saved_model_dir), "Saved model directory does not exist"
            shuffle_idxs = np.load(
                os.path.join(saved_model_dir, "shuffle_idxs.npy"))
        else:
            print("No saved model directory, not loading shuffle idxs")
            assert inference_config.baselines_only, "Must load model to run RL"
            shuffle_idxs = None
        print("Loading data")
        if inference_config.baselines_only:
            assert (
                not train_config.shuffle
            ), "Can't shuffle data for baselines (need reproducibile inference)"
        _, _, inference_dataloader, obs_size, dem_obs_size, horizon, shuffle_idxs = (
            load_data(
                dataset_path,
                obs_dataset_path,
                general_config,
                model_config,
                train_config,
                inference_config,
                shuffle_idxs=shuffle_idxs,
                obs_dataset_path_2=None,
            ))
        models = []
        if not inference_config.baselines_only:
            for saved_model_dir in saved_model_dirs:
                # model = load_model(saved_model_dir, model_config, model_idx)
                # def load_model(saved_model_dir, model_config, train_config, obs_size, dem_obs_size, horizon):
                print(f"Loading model from {saved_model_dir}")
                model = load_model(
                    saved_model_dir,
                    model_config,
                    train_config,
                    obs_size,
                    dem_obs_size,
                    horizon,
                )
                models.append(model)

    if inference_config.skip_all_inference:
        print("Skipping all inference")
    else:
        # goal_to_use = inference_config.use_goal
        if policy_run is not None:
            print(f"Checking for goal from {algo_name}")
            policy_summary = policy_run.summary
            if "goal" in policy_summary:
                goal_to_use = policy_summary["goal"]
                assert (
                    not hasattr(inference_config, "use_goal")
                    or inference_config.use_goal is None
                ), "Can't use specific goal if {algo_name} run has goal"
                print(f"Using goal from {algo_name} run: {goal_to_use}")
            else:
                if (hasattr(inference_config, "use_goal")
                        and inference_config.use_goal is not None):
                    print(
                        "{algo_name} run does not have goal, using specified goal instead"
                    )
                    goal_to_use = inference_config.use_goal
                else:
                    print(
                        "{algo_name} run does not have goal specified, and no single goal specified, using random goal(s)"
                    )
                    goal_to_use = None
            if not inference_config.baselines_only:
                # policy_path = os.path.join(
                #     "runs", policy_run_name, "models", "final.zip"
                # )
                # policy_path = os.path.join(saved_model_dir, "final.zip")
                policy_path = find_path(
                    general_config.run_dirs,
                    [policy_run_name, "models"],
                    ["final.zip"],
                    raise_on_missing=True)
                policy_path = os.path.join(policy_path, "final.zip")
            else:
                assert (len(baselines_config.baselines) == 1
                        ), "Can only eval one baseline at a time"
                baseline_method = baselines_config.baselines[0]
                if baseline_method == "pemirl":
                    policy_path = (
                        f"scratch/runs/{policy_run_name}/pemirl/meta_checkpoint.pt"
                    )
                else:
                    policy_path = (
                        f"scratch/runs/{policy_run_name}/{baseline_method}/policy.zip"
                    )
            print(f"Will load {algo_name} policy from {policy_path}")
        else:
            print("Not using a policy run, so not loading a policy")
            goal_to_use = None
            policy_path = policy_path_override
            if policy_path is not None:
                if not os.path.exists(policy_path):
                    raise FileNotFoundError(
                        f"Explicit policy path does not exist: {policy_path}"
                    )
                print(f"Using explicit policy path: {policy_path}")

        if not inference_config.baselines_only:
            print(
                f"Doing SRI inference {'with RL' if policy_run is None else 'with existing policy'}"
            )
            sri(
                models,
                inference_dataloader,
                orl_dataset_path,
                general_config,
                model_config,
                train_config,
                rl_config,
                inference_config,
                policy_path=policy_path,
                goal_to_use=goal_to_use,
            )
        # run baselines
        if not inference_config.only_rl and "none" not in baselines_config.baselines:
            print("Running baselines")
            run_baselines(
                inference_dataloader,
                general_config,
                inference_config,
                baselines_config,
                rl_config,
                policy_path,
                goal_to_use,
            )
