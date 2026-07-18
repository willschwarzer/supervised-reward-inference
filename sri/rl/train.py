import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import random
import numpy as np
import torch
from PIL import Image
import argparse
from tqdm import tqdm
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
import pickle
import torch.nn as nn
import imageio
from sri.reward_inference.train import TaskDataset
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
)

from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import (
    compute_reward_batch as reach_compute_reward_batch,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import (
    compute_reward_batch as pick_place_compute_reward_batch,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import (
    compute_reward_batch as push_compute_reward_batch,
)
from sri.utils import scale, get_dataset_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-algorithm",
        choices=["PPO", "SAC", "TQC"],
        default="PPO",
        help="SB3 algorithm to use: PPO, SAC, TQC. Default is PPO.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pick-place",
        help="Environment to use. Default is pick-place-v2-goal-observable.",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the SB3 algorithm. Default is 3e-4.",
    )
    parser.add_argument(
        "--rl-batch-size",
        type=int,
        default=500,
        help="Batch size for RL algorithms. Default is 500. Should be multiple of max_steps.",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=10000000,
        help="Replay buffer size for offline RL. Default is 10000000 (10 million).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for the SB3 algorithm. Default is 0.99.",
    )
    parser.add_argument(
        "--rl-learning-steps",
        type=int,
        default=10000000,
        help="Number of learning steps. Default is 10000000.",
    )
    parser.add_argument(
        "--rl-gradient-steps",
        type=int,
        default=1,
        help="Number of gradient steps for offline RL. Default is 1.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Maximum number of steps per episode. Default is 500.",
    )
    parser.add_argument(
        "--rl-render-interval",
        type=int,
        default=10000,
        help="Number of steps between renders during training. Default is 100.",
    )
    parser.add_argument(
        "--rl-render-duration",
        type=int,
        default=-1,
        help="Duration of each render during training in seconds. Default is entire episode.",
    )
    parser.add_argument(
        "--policy-latent-dim",
        type=int,
        default=64,
        help="Number of neurons in each layer",
    )
    parser.add_argument(
        "--policy-num-layers",
        type=int,
        default=2,
        help="Number of layers in the network",
    )
    parser.add_argument(
        "--num-critics", type=int, default=2, help="Number of critics in TQC"
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="Number of steps between target network updates. Default is 1.",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=1,
        help="Number of steps between gradient updates. Default is 1.",
    )
    parser.add_argument(
        "--use-sde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use SDE (state-dependent exploration). Default is False.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel. Default is 1.",
    )
    parser.add_argument(
        "--rl-save-freq",
        type=int,
        default=100000,
        help="Number of steps between model saves. Default is 100000.",
    )
    parser.add_argument(
        "--rl-weight-resets",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to reset the policy after policy-reset-steps",
    )
    parser.add_argument(
        "--rl-weight-reset-interval",
        default=10000,
        type=int,
        help="Number of steps after which to reset the policy",
    )
    parser.add_argument(
        "--rl-weight-reset-ratio",
        default=0.5,
        type=float,
        help="How far towards random weights to reset weights. Default is 0.5 (halfway).",
    )
    # parser.add_argument('--rl-learning-starts', default=10000, type=int,
    #                     help="Number of steps before training TQC/SAC. Default is 10000.")
    parser.add_argument(
        "--load-rl-run",
        default=None,
        type=str,
        help="Name of wandb run's model to load. Default is None.",
    )
    parser.add_argument(
        "--gripped-start",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to start with a random saved position with the object gripped. Default is False.",
    )
    parser.add_argument(
        "--reinit-goals",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to reinitialize the goal after each episode. Default is True.",
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
        default=[0.05, 0.3],
        help="Bounds for the z-coordinate of the goal. Default is [0.05, 0.3].",
    )
    parser.add_argument(
        "--record-grips",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to save positions with the object gripped. Default is False.",
    )
    parser.add_argument(
        "--skip-rl-training",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to skip training the RL policy (only used from experiment scripts).",
    )
    parser.add_argument(
        "--extra-success-reward",
        type=float,
        default=0.0,
        help="Extra reward for reaching the goal. Default is 0.0.",
    )
    parser.add_argument(
        "--render-reward",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to render the current reward as text overlaid on the state (for SRI experiments). Default is False.",
    )
    parser.add_argument(
        "--limit-reward-obs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to limit the reward model to only observe relevant variables (only used from experiment scripts). Default is False.",
    )
    parser.add_argument(
        "--unscale-rl-rewards",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to unscale the rewards for RL training. Default is False.",
    )
    parser.add_argument(
        "--offline-rl-dataset",
        default=None,
        type=str,
        help="Path to dataset of states and actions to warm-start replay buffer. Default is None, which means no offline RL.",
    )
    parser.add_argument(
        "--offline-num-task-reps-per-obs",
        default=1,
        type=int,
        help="Number of (random) task representations to concatenate with each observation for offline RL. Default is 1.",
    )
    parser.add_argument(
        "--offline-num-transitions",
        default=1000000,
        type=int,
        help="Number of transitions to add to the warm-start offline dataset. Default is 1000000.",
    )
    parser.add_argument(
        "--offline-rejection-sampling-num-bins",
        default=None,
        type=int,
        help="Number of bins to use for rejection sampling of transitions with diverse rewards. Default is None (no rejection sampling).",
    )
    parser.add_argument(
        "--offline-rejection-sampling-bias-factor",
        default=1.0,
        type=float,
        help="How much to bias rejection sampling towards high-reward transitions. Default is 1.0 (no bias).",
    )
    parser.add_argument(
        "--offline-training-epochs",
        default=0,
        type=float,
        help="Number of training epochs for offline RL. Default is 0 (immediately start online RL).",
    )
    parser.add_argument(
        "--rl-use-gt-reward",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use the ground-truth reward for debugging RL. Default is False.",
    )
    parser.add_argument(
        "--rl-scale-gt-reward",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to scale the ground-truth reward of the RL environments like the inferred reward. Default is False.",
    )
    parser.add_argument(
        "--rl-use-gt-goal",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to include the ground-truth goal in the observation for debugging RL. Default is False.",
    )
    parser.add_argument(
        "--rl-no-task-rep",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use omit the task representation in the observation for debugging. Default is False (include task rep).",
    )
    parser.add_argument(
        "--num-discrete-reward-bins",
        default=None,
        type=int,
        help="Number of discrete reward bins to use for post-hoc discretized reward inference. Default is None (continuous rewards).",
    )
    parser.add_argument(
        "--proximity-reward",
        default=0.0,
        type=float,
        help="Extra reward for being close to the object. Default is 0.0. Only for pick-place.",
    )

    args = parser.parse_args()
    # if args.gripped_start:
    #     assert args.env == "pick-place", "Can only use gripped_start with pick-place"
    if args.env == "push":
        assert (
            args.goal_z_bounds[1] < 0.03
        ), "Push environment goals must be on the table"
    args.env = args.env + "-v2-goal-observable"
    if args.rl_render_duration < 0:
        args.rl_render_duration = args.horizon
    assert (
        args.rl_render_interval > args.rl_render_duration
    ), "render_interval must be greater than render_duration"
    return args


# def reset_weights(model, verbose=False):
#     print("Resetting weights...")
#     num_layers = 0
#     for name, layer in model.policy.named_modules():
#         if verbose:
#             print(name, layer)
#         if isinstance(layer, nn.Linear):
#             if verbose:
#                 print(name, "weight before", layer.weight)
#                 print(name, "bias before", layer.bias)
#             num_layers += 1
#             nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 layer.bias.data.fill_(0.01)
#             if verbose:
#                 print(name, "weight after", layer.weight)
#                 print(name, "bias after", layer.bias)
#     print(f"Reset {num_layers} layers")
#     print("Resetting optimizer...")
#     model.actor.optimizer = model.actor.optimizer.__class__(model.actor.parameters(), lr=model.lr_schedule(1))
#     model.critic.optimizer = model.critic.optimizer.__class__(model.critic.parameters(), lr=model.lr_schedule(1))


def reset_weights(model, reset_ratio=1.0, verbose=False):
    assert 0.0 <= reset_ratio <= 1.0, "Reset ratio must be between 0 and 1"

    print("Resetting weights...")
    num_layers = 0
    for name, layer in model.policy.named_modules():
        if verbose:
            print(name, layer)
        if isinstance(layer, nn.Linear):
            if verbose:
                print(name, "weight before", layer.weight)
                print(name, "bias before", layer.bias)
            num_layers += 1
            new_weight = nn.init.xavier_uniform_(torch.empty_like(layer.weight))
            new_bias = (
                torch.empty_like(layer.bias).fill_(0.01)
                if layer.bias is not None
                else None
            )

            layer.weight.data = (
                1 - reset_ratio
            ) * layer.weight.data + reset_ratio * new_weight
            if layer.bias is not None:
                layer.bias.data = (
                    1 - reset_ratio
                ) * layer.bias.data + reset_ratio * new_bias

            if verbose:
                print(name, "weight after", layer.weight)
                print(name, "bias after", layer.bias)
    print(f"Reset {num_layers} layers")
    print("Resetting optimizer...")
    model.actor.optimizer = model.actor.optimizer.__class__(
        model.actor.parameters(), lr=model.lr_schedule(1)
    )
    model.critic.optimizer = model.critic.optimizer.__class__(
        model.critic.parameters(), lr=model.lr_schedule(1)
    )


class RenderWandbCallback(WandbCallback):
    """
    This is a custom callback that renders an episode every certain number of steps
    and logs it to wandb.
    """

    def __init__(
        self,
        env,
        render_interval,
        render_duration,
        do_weight_resets,
        weight_reset_interval,
        weight_reset_ratio,
        max_steps,
        render_reward,
        # cmd_args,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env = env
        self.max_steps = max_steps
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1
        self.render_interval = render_interval
        self.render_duration = render_duration
        self.render_end = 0
        self.render_env_index = 0
        self.do_weight_resets = do_weight_resets
        self.weight_reset_interval = weight_reset_interval
        self.weight_reset_ratio = weight_reset_ratio
        self.render_reward = render_reward
        self.special_save_step = 2_000_000
        # self.cmd_args = cmd_args  # need this to save the arguments
        assert render_duration < render_interval

    def _on_step(self) -> bool:
        super()._on_step()
        if self.num_timesteps % self.special_save_step == 0:
            print(f"Saving model at step {self.num_timesteps}")
            # self.model.save(f"runs/{wandb.run.name}/models/{self.num_timesteps}")
            self.model.save(f"scratch/runs/{wandb.run.name}/models/{self.num_timesteps}")

        adjusted_timesteps = self.num_timesteps // self.num_envs

        if adjusted_timesteps % self.render_interval == 0:
            print(f"Rendering at step {self.num_timesteps}")
            self.render_end = adjusted_timesteps + self.render_duration
            self.frames = []
            # Render a specific environment using env_method
            if self.num_envs > 1:
                img_as_array = self.env.env_method(
                    "render",
                    render_reward=self.render_reward,
                    indices=[self.render_env_index],
                )[0]
            else:
                img_as_array = self.env.render(render_reward=self.render_reward)
            self.frames.append(img_as_array)

        elif adjusted_timesteps < self.render_end:
            if self.num_timesteps % 1 == 0:
                print(f"Rendering step {self.num_timesteps}")
                print(
                    f"Adjusted step {adjusted_timesteps}, render finished at {self.render_end}"
                )
            if self.num_envs > 1:
                img_as_array = self.env.env_method(
                    "render",
                    render_reward=self.render_reward,
                    indices=[self.render_env_index],
                )[0]
            else:
                img_as_array = self.env.render(render_reward=self.render_reward)
            self.frames.append(img_as_array)
        elif adjusted_timesteps == self.render_end:
            print(f"Saving video at step {self.num_timesteps}")
            # save_dir = f"runs/{wandb.run.name}/images/"
            save_dir = f"scratch/runs/{wandb.run.name}/images/"
            os.makedirs(save_dir, exist_ok=True)
            imageio.mimsave(f"{save_dir}/{self.num_timesteps}.mp4", self.frames, fps=24)

        if (
            self.do_weight_resets
            and self.num_timesteps % self.weight_reset_interval == 0
        ):
            reset_weights(self.model, reset_ratio=self.weight_reset_ratio)

        # also log info
        if self.num_timesteps // self.num_envs % 25 == 0:
            infos = self.locals["infos"]
            # wandb.log(infos[0])
            ave_infos = {}
            for info in infos:
                # first, delete a few things that we don't want to log
                del info["TimeLimit.truncated"]
                if "terminal_observation" in info:
                    del info["terminal_observation"]
                for key, value in info.items():
                    if key not in ave_infos:
                        if key == "episode":
                            ave_infos[key] = {"r": 0, "t": 0, "l": 0}
                        else:
                            ave_infos[key] = 0
                    if key == "episode":
                        ave_infos[key]["r"] += value["r"]
                        ave_infos[key]["t"] += value["t"]
                        ave_infos[key]["l"] += value["l"]
                    else:
                        ave_infos[key] += value
            for key, value in ave_infos.items():
                if key == "episode":
                    for k, v in value.items():
                        ave_infos[key][k] /= len(infos)
                else:
                    ave_infos[key] /= len(infos)
            wandb.log(ave_infos)
            # print(ave_infos)

        return True

    def save_model(self) -> None:
        super().save_model()


def make_env(
    env_cls,
    env_config,
    state_encoders=None,
    reward_mlps=None,
    task_reps=None,
    goals=None,
    limit_reward_obs=False,
    reinit_goals=False,  # need to see if this breaks stuff XXX
    record_grips=False,
    **rl_config_kwargs,  # Added to accept rl_config variables as kwargs
):
    # convert goal bounds to tuples
    goal_bounds = (
        (
            env_config.goal_x_bounds[0],
            env_config.goal_y_bounds[0],
            env_config.goal_z_bounds[0],
        ),
        (
            env_config.goal_x_bounds[1],
            env_config.goal_y_bounds[1],
            env_config.goal_z_bounds[1],
        ),
    )
    random_hand_starts = rl_config_kwargs.get("random_hand_starts", False)
    if random_hand_starts:
        hand_starts_path = os.path.join("artifacts", "hand_starts", "hand_starts.npy")
        hand_starts = np.load(hand_starts_path)
    else:
        hand_starts = None

    assert not rl_config_kwargs.get(
        "no_init_success", False
    ), "no_init_success not implemented yet"

    def _env():
        return env_cls(
            reinit_goals=reinit_goals,
            record_grips=record_grips,
            goal_bounds=goal_bounds,
            render_mode="rgb_array",
            state_encoders=state_encoders,
            reward_mlps=reward_mlps,
            task_reps=task_reps,
            goals=goals,
            gripped_start=env_config.gripped_start,
            horizon=env_config.horizon,
            extra_success_reward=rl_config_kwargs.get(
                "extra_success_reward", 0
            ),  # Default value as example
            limit_reward_obs=limit_reward_obs,
            unscale_reward=rl_config_kwargs.get(
                "unscale_rewards", True
            ),  # Default value as example
            use_gt_reward=rl_config_kwargs.get(
                "use_gt_reward", True
            ),  # Default value as example
            scale_gt_reward=rl_config_kwargs.get(
                "scale_gt_reward", False
            ),  # Default value as example
            use_gt_goal=rl_config_kwargs.get(
                "use_gt_goal", True
            ),  # Default value as example
            no_task_rep=rl_config_kwargs.get(
                "no_task_rep", True
            ),  # Default value as example
            proximity_reward=rl_config_kwargs.get(
                "proximity_reward", 0.0
            ),  # Default value as example
            # no_init_success=rl_config_kwargs.get(
            #     "no_init_success", False
            # ),  # Default value as example
            success_requires_touch=rl_config_kwargs.get(
                "success_requires_touch", False
            ),  # Default value as example
            third_gt=rl_config_kwargs.get(
                "third_gt", False
            ),  # Default value as example
            half_gt=rl_config_kwargs.get("half_gt", False),  # Default value as example
            ensembling=rl_config_kwargs.get(
                "ensembling", None
            ),  # Default value as example
            include_extra_reward_info=rl_config_kwargs.get(
                "include_extra_reward_info", False
            ),  # Default value as example
            include_partial_reward_info=rl_config_kwargs.get(
                "include_partial_reward_info", False
            ),  # Default value as example
            mask_obj=rl_config_kwargs.get(
                "mask_obj", False
            ),  # Default value as example
            reinit_obj_pos=rl_config_kwargs.get(
                "reinit_obj_pos", False
            ),  # Default value as example
            hand_starts=hand_starts,
            multi_obj_pos=rl_config_kwargs.get(
                "multi_obj_pos", False
            ),  # Default value as example
            grasp_rew_only=rl_config_kwargs.get(
                "grasp_rew_only", False
            ),  # Default value as example
        )

    return _env


def train(
    general_config,
    # model_config,
    inference_config,
    rl_config,
    state_encoders=None,
    reward_mlps=None,
    task_reps=None,
    goals=None,
    orl_dataset_path=None,
    limit_reward_obs=False,
    goal_bounds=None,
):
    # project = (
    #     f"metaworld_{inference_config.env}" if inference_config.env != "pick-place-v2-goal-observable" else "metaworld"
    # )
    # run = wandb.init(project=project, sync_tensorboard=True)
    # wandb.config.update(args)
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[inference_config.env]
    # pick_place_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["pick-place-v2-goal-hidden"]
    reinit_goals = rl_config.full_legacy and rl_config.full_legacy_reinit_goals
    if rl_config.num_envs == 1:
        print("Using single environment")
        env = make_env(
            env_cls,
            inference_config,
            # rl_config,
            # model_config,
            state_encoders,
            reward_mlps,
            task_reps,
            goals,
            limit_reward_obs,
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
            multi_obj_pos=inference_config.multi_obj_pos,
            grasp_rew_only=rl_config.grasp_rew_only,
        )()
    else:
        print(f"Initializing {rl_config.num_envs} environments")
        env = SubprocVecEnv(
            [
                make_env(
                    env_cls,
                    inference_config,
                    # rl_config,
                    # model_config,
                    state_encoders,
                    reward_mlps,
                    task_reps,
                    goals,
                    limit_reward_obs,
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
                    multi_obj_pos=inference_config.multi_obj_pos,
                    grasp_rew_only=rl_config.grasp_rew_only,
                )
                for _ in range(rl_config.num_envs)
            ]
        )
        print(f"Using SubprocVecEnv with {env.num_envs} environments")
        print("Wrapping in VecMonitor")
        env = VecMonitor(env)
    env.reset()
    name = wandb.run.name
    # os.makedirs(f"runs/{name}", exist_ok=True)
    # os.makedirs(f"runs/{name}/models", exist_ok=True)
    # os.makedirs(f"runs/{name}/images", exist_ok=True)
    os.makedirs(f"scratch/runs/{name}", exist_ok=True)
    os.makedirs(f"scratch/runs/{name}/models", exist_ok=True)
    os.makedirs(f"scratch/runs/{name}/images", exist_ok=True)


    # Getting our agent
    # first, check if we're loading a model
    if rl_config.load_run is not None:
        raise NotImplementedError("Loading models not implemented yet")
        # load the model
        base_path = os.path.join("runs", rl_config.load_run, "models")
        args_path = f"{base_path}/model.args"
        model_path = f"{base_path}/model.zip"
        with open(args_path, "rb") as f:
            env_args = pickle.load(f)
        if env_args.rl_algorithm != args.rl_algorithm:
            raise ValueError("Algorithm must be the same as the loaded model")
        elif env_args.env != args.env:
            raise ValueError("Environment must be the same as the loaded model")
        if args.rl_algorithm == "PPO":
            model = PPO.load(model_path, env=env)
        elif args.rl_algorithm == "SAC":
            model = SAC.load(model_path, env=env)
        elif args.rl_algorithm == "TQC":
            model = TQC.load(model_path, env=env)
        # need to update tensorboard log location
        # model.tensorboard_log = f"runs/{name}/tensorboard/"
        model.tensorboard_log = f"scratch/runs/{name}/tensorboard/"
    else:
        if rl_config.algorithm == "PPO":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                    vf=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                ),
                activation_fn=nn.ReLU,
            )
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=rl_config.ppo_n_steps,
                # tensorboard_log=f"runs/{name}/tensorboard/",
                tensorboard_log=f"scratch/runs/{name}/tensorboard/",
                gamma=rl_config.gamma,
                policy_kwargs=policy_kwargs,
                use_sde=rl_config.use_sde,
                batch_size=rl_config.batch_size,
                learning_rate=rl_config.learning_rate,
            )
        elif rl_config.algorithm == "SAC":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                    qf=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                ),
                activation_fn=nn.ReLU,
            )
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                # tensorboard_log=f"runs/{name}/tensorboard/",
                tensorboard_log=f"scratch/runs/{name}/tensorboard/",
                gamma=rl_config.gamma,
                policy_kwargs=policy_kwargs,
                use_sde=rl_config.use_sde,
                target_update_interval=rl_config.target_update_interval,
                train_freq=rl_config.train_frequency,
                batch_size=rl_config.batch_size,
                learning_rate=rl_config.learning_rate,
                gradient_steps=rl_config.gradient_steps,
                buffer_size=rl_config.replay_buffer_size,
            )
        elif rl_config.algorithm == "TQC":
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                    qf=[rl_config.policy_latent_dim] * rl_config.policy_num_layers,
                    n_critics=rl_config.num_critics,
                ),
                activation_fn=nn.ReLU,
            )
            model = TQC(
                "MlpPolicy",
                env,
                verbose=1,
                # tensorboard_log=f"runs/{name}/tensorboard/",
                tensorboard_log=f"scratch/runs/{name}/tensorboard/",
                gamma=rl_config.gamma,
                policy_kwargs=policy_kwargs,
                use_sde=rl_config.use_sde,
                target_update_interval=rl_config.target_update_interval,
                train_freq=rl_config.train_frequency,
                batch_size=rl_config.batch_size,
                learning_rate=rl_config.learning_rate,
                gradient_steps=rl_config.gradient_steps,
                buffer_size=rl_config.replay_buffer_size,
            )
        else:
            raise ValueError("Algorithm must be one of PPO, SAC or TQC")

    if inference_config.skip_rl:
        return model

    # mwahaha, now we can use our custom callback >:D
    # we're unstoppably on our way to a gif/video
    wandb_callback = RenderWandbCallback(
        env,
        rl_config.render_interval,
        rl_config.render_duration,
        rl_config.weight_resets,
        rl_config.weight_reset_interval,
        rl_config.weight_reset_ratio,
        inference_config.horizon,  # TODO maybe let both inference and rl have horizon later
        rl_config.render_reward,
        gradient_save_freq=rl_config.save_freq,
        model_save_freq=rl_config.save_freq,
        # model_save_path=f"runs/{name}/models/",
        model_save_path=f"scratch/runs/{name}/models/",
        # cmd_args=args,  # need this to save the arguments
        verbose=1,
    )

    # now for offline RL if we're using it
    # if orl_config.dataset_path is not None:
    if rl_config.offline_rl:
        # can only do this if we have state_encoder and task_reps
        assert state_encoder is not None, "Need state_encoder for offline RL"
        assert task_reps is not None, "Need task_reps for offline RL"
        assert orl_dataset_path is not None, "Need dataset path for offline RL"
        # should have "observations.npy" and "actions.npy"
        obs_path = os.path.join(orl_dataset_path, "observations.npy")
        action_path = os.path.join(orl_dataset_path, "actions.npy")
        assert os.path.exists(obs_path), "Observations file not found"
        assert os.path.exists(action_path), "Actions file not found"
        obs = np.load(obs_path)
        actions = np.load(action_path)
        print("Offline RL dataset loaded")
        if rl_config.offline_rejection_sampling_num_bins is not None:
            print(
                f"Doing rejection sampling with {rl_config.offline_rejection_sampling_num_bins} bins"
            )
            if "reach" in inference_config.env:
                reward_bounds = (
                    1,
                    10,
                )  # don't include extra_success_reward because there should only
                # be one bin for 10 to 10+extra_success_reward
            elif "pick-place" in inference_config.env or "push" in inference_config.env:
                reward_bounds = (
                    0.5,
                    6 + rl_config.proximity_reward + max(1, rl_config.proximity_reward),
                )
            else:
                raise NotImplementedError(
                    "Rejection sampling not implemented for this environment"
                )
            if (not rl_config.use_gt_reward and not rl_config.unscale_rewards) or (
                rl_config.scale_gt_reward and rl_config.use_gt_reward
            ):
                # reward_bounds = (-3, 4)
                new_low = scale(
                    reward_bounds[0], 0, 10 + rl_config.extra_success_reward, -3, 3
                )
                new_high = scale(
                    reward_bounds[1], 0, 10 + rl_config.extra_success_reward, -3, 3
                )
                reward_bounds = (new_low, new_high)
            reward_bins = np.linspace(
                reward_bounds[0],
                reward_bounds[1],
                rl_config.offline_rejection_sampling_num_bins - 1,
            )
            print(f"Reward bins: {reward_bins}")
            target_frequencies = (
                rl_config.offline_rejection_sampling_bias_factor
                ** np.arange(rl_config.offline_rejection_sampling_num_bins)
            )
            target_frequencies /= target_frequencies.sum()
            print(f"Target frequencies: {target_frequencies}")
        else:
            reward_bins = None
            target_frequencies = None
        num_transitions = offline_warm_start(
            model,
            inference_config.env,
            obs,
            actions,
            goals,
            state_encoder,
            reward_mlp,
            task_reps,
            rl_config.offline_num_task_reps_per_obs,
            rl_config.offline_num_transitions,
            reward_bins,
            target_frequencies,
            rl_config.use_gt_goal,
            rl_config.no_task_rep,
            rl_config.use_gt_reward,
            rl_config.num_discrete_reward_bins,
            rl_config.unscale_rewards,
            rl_config.scale_gt_reward,
            rl_config.extra_success_reward,
            rl_config.proximity_reward,
        )

        # do offline training, if any
        if rl_config.offline_training_epochs > 0:
            # def print_parameters(model, title="Model Parameters"):
            #     print(title)
            #     for name, param in model.policy.state_dict().items():
            #         print(f"{name}: {param}")
            num_steps = int(
                rl_config.offline_training_epochs
                * (num_transitions // rl_config.batch_size)
            )
            print(
                f"Doing offline training for {rl_config.offline_training_epochs} epochs, {num_steps} steps"
            )
            model.learn(total_timesteps=1, callback=wandb_callback)  # setup
            # print_parameters(model, "Weights before training")
            for _ in tqdm(range(num_steps)):
                model.train(1)  # 1 is the number of gradient steps
            # print_parameters(model, "Weights after training")
    else:
        print("Not doing offline RL")
        # assert rl_config.offline_training_epochs == 0, "Cannot do offline training without a dataset"

    print("Starting online RL training for", rl_config.learning_steps, "steps")
    model.learn(total_timesteps=rl_config.learning_steps, callback=wandb_callback)

    # print("Finished online RL training, saving model to", f"runs/{name}/models/final")
    # model.save(f"runs/{name}/models/final")
    print("Finished online RL training, saving model to", f"scratch/runs/{name}/models/final")
    model.save(f"scratch/runs/{name}/models/final")

    return model


def offline_warm_start(
    model,
    env,
    obs,
    actions,
    goals,
    state_encoder,
    reward_mlp,
    task_reps,
    num_task_reps_per_obs,
    num_transitions,
    reward_bins,
    target_frequencies,
    use_gt_goal,
    no_task_rep,
    use_gt_reward,
    num_discrete_reward_bins,
    unscale_rewards,
    scale_gt_reward,
    extra_success_reward,
    proximity_reward,
):
    replay_buffer = model.replay_buffer
    device = next(state_encoder.parameters()).device
    task_reps = task_reps.to(device)

    num_transitions_added = 0
    if reward_bins is not None:
        num_bins = len(reward_bins) + 1  # Total bins
        reward_counts = np.zeros(num_bins)
        epsilon = 1e-5  # Small constant to prevent division by zero
    residual_states = None
    residual_next_states = None
    residual_actions = None
    residual_rewards = None
    max_loops = len(task_reps)
    cur_loop = 0

    # if num_discrete_reward_bins is not None:
    #     if (not unscale_rl_rewards and not use_gt_reward) or (scale_gt_reward and use_gt_reward):
    #         bounds = (-3, 3) # will be modified later
    #     elif (unscale_rl_rewards and not use_gt_reward) or (not scale_gt_reward and use_gt_reward):
    #         bounds = (0, 10+extra_success_reward) # will be modified later
    #     else:
    #         raise ValueError("how did we get here")
    #     # shift bottom bound up and top bound down according to number of bins
    #     length = bounds[1] - bounds[0]
    #     bounds = (bounds[0] + length / num_discrete_reward_bins, bounds[1] - length / num_discrete_reward_bins)
    #     bins = np.linspace(bounds[0], bounds[1], num_discrete_reward_bins - 1)
    #     rewards = np.digitize(rewards, bins, right=True)
    #     bin_centers = (bins[1:] + bins[:-1]) / 2
    # inference_dataset = TaskDataset(
    #     obs,
    #     actions,
    #     goals,
    #     args.n,
    #     args.rand_n,
    #     args.num_obs,
    #     demonstration_type=args.demonstration_type,
    #     synthesize_obs=False,
    #     args=args,
    # )
    while num_transitions_added < num_transitions and cur_loop < max_loops:
        # loop through dataset, continually sampling new random task representations
        for task_idx in tqdm(range(obs.shape[0])):
            states_full = obs[task_idx, :, :-1]
            actions_full = actions[task_idx]
            next_states_full = obs[task_idx, :, 1:]

            # Flatten the arrays to simplify batch processing
            states_full = states_full.reshape(-1, states_full.shape[-1])
            actions_full = actions_full.reshape(-1, actions_full.shape[-1])
            next_states_full = next_states_full.reshape(-1, next_states_full.shape[-1])

            indices = torch.randint(0, len(task_reps), size=(len(states_full),))
            random_task_reps = task_reps[indices]
            random_goals = goals[indices]

            state_torch = torch.tensor(states_full, dtype=torch.float32).to(device)
            state_rep = state_encoder(state_torch)

            if use_gt_reward:
                # need to create a batch dimension
                # (and it needs to be the second dimension because the reward functions expect one task rep per batch element)
                state_torch = state_torch.unsqueeze(1)
                goals_torch = torch.tensor(random_goals, dtype=torch.float32).to(device)
                if "reach" in env.lower():
                    rewards = (
                        reach_compute_reward_batch(
                            state_torch,
                            goals_torch,
                            extra_success_reward=extra_success_reward,
                        )
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                elif "pick-place" in env.lower():
                    rewards = (
                        pick_place_compute_reward_batch(
                            state_torch,
                            goals_torch,
                            extra_success_reward=extra_success_reward,
                            proximity_reward=proximity_reward,
                        )
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                elif "push" in env.lower():
                    rewards = (
                        push_compute_reward_batch(
                            state_torch,
                            goals_torch,
                            extra_success_reward=extra_success_reward,
                        )
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    raise ValueError(
                        "Environment not recognized for ground-truth reward inference"
                    )

                if scale_gt_reward:
                    rewards = scale(rewards, 0, 10 + extra_success_reward, -3, 3)
            else:
                if reward_mlp is not None:
                    rewards = (
                        reward_mlp(random_task_reps, state_rep).detach().cpu().numpy()
                    )
                else:
                    rewards = (
                        torch.matmul(state_rep, random_task_reps.T)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                if unscale_rewards:
                    rewards = scale(rewards, -3, 3, 0, 10 + extra_success_reward)

            # if num_discrete_reward_bins is not None:
            #     rewards = bin_centers[rewards]
            # Augment states and next_states with task representations
            states_augmented = states_full
            next_states_augmented = next_states_full
            if not no_task_rep:
                random_task_reps_np = random_task_reps.detach().cpu().numpy()
                states_augmented = np.concatenate(
                    (states_augmented, random_task_reps_np), axis=1
                )
                next_states_augmented = np.concatenate(
                    (next_states_augmented, random_task_reps_np), axis=1
                )
            if use_gt_goal:
                states_augmented = np.concatenate(
                    (states_augmented, random_goals), axis=1
                )
                next_states_augmented = np.concatenate(
                    (next_states_augmented, random_goals), axis=1
                )
            states_augmented = states_augmented[
                ..., 21:
            ]  # first 21 dims are reward model inputs
            next_states_augmented = next_states_augmented[..., 21:]
            # states_augmented = np.concatenate((states_full, random_task_reps_np), axis=1)[..., 21:] # first 21 dims are reward model inputs
            # next_states_augmented = np.concatenate((next_states_full, random_task_reps_np), axis=1)[..., 21:]

            accepted = np.ones_like(
                rewards, dtype=bool
            )  # Default to accepting all if no bins

            if reward_bins is not None:
                bin_indices = np.digitize(rewards, reward_bins, right=False)
                current_counts = reward_counts[bin_indices]
                total_counts = reward_counts.sum()
                target_counts = target_frequencies * total_counts
                diff_from_target = reward_counts - target_counts
                # bin_accept_probs = np.where(diff_from_target > 0, 0, 1)
                # need to do logsumexp trick to avoid numerical instability
                # bin_accept_probs = np.exp(-reward_counts - np.max(-reward_counts))
                # bin_accept_probs /= np.max(bin_accept_probs)
                # accept_probs = bin_accept_probs[bin_indices]
                # accepted = np.random.rand(len(accept_probs)) < accept_probs
                bin_accept = np.where(diff_from_target > 0, False, True)
                # print(f"diff_from_target: {diff_from_target}")
                # print(f"bin_accept: {bin_accept}")
                accepted = bin_accept[bin_indices]
                # if task_idx > 5:
                # update reward counts: add number of transitions for each bin that were accepted
                reward_counts += np.bincount(
                    bin_indices[accepted], minlength=len(reward_counts)
                )

            # Handle transition addition respecting environment batch size
            num_accepted = accepted.sum()
            if num_accepted > 0:
                # Incorporate residuals
                if residual_states is not None:
                    states_augmented = np.concatenate(
                        [residual_states, states_augmented]
                    )
                    next_states_augmented = np.concatenate(
                        [residual_next_states, next_states_augmented]
                    )
                    actions_full = np.concatenate([residual_actions, actions_full])
                    rewards = np.concatenate([residual_rewards, rewards])
                    accepted = np.concatenate(
                        [np.ones(len(residual_rewards), dtype=bool), accepted]
                    )
                    residual_states = None
                    residual_next_states = None
                    residual_actions = None
                    residual_rewards = None

                num_accepted = accepted.sum()
                if num_accepted % model.n_envs != 0:
                    num_residual = num_accepted % model.n_envs
                    num_accepted -= num_residual
                    accepted_indices = np.where(accepted)[0]
                    residual_indices = accepted_indices[-num_residual:]
                    accepted_indices = accepted_indices[:-num_residual]

                    residual_states = states_augmented[residual_indices]
                    residual_next_states = next_states_augmented[residual_indices]
                    residual_actions = actions_full[residual_indices]
                    residual_rewards = rewards[residual_indices]
                    accepted = np.zeros_like(rewards, dtype=bool)
                    accepted[accepted_indices] = True

                accepted_states_augmented = states_augmented[accepted]
                accepted_next_states_augmented = next_states_augmented[accepted]
                accepted_actions_full = actions_full[accepted]
                accepted_rewards = rewards[accepted]
                for i in range(0, num_accepted, model.n_envs):
                    replay_buffer.add(
                        accepted_states_augmented[i : i + model.n_envs],
                        accepted_next_states_augmented[i : i + model.n_envs],
                        accepted_actions_full[i : i + model.n_envs],
                        accepted_rewards[i : i + model.n_envs],
                        np.zeros(model.n_envs, dtype=bool),
                        [{}] * model.n_envs,
                    )
                num_transitions_added += num_accepted

            if num_transitions_added >= num_transitions:
                break
        if reward_bins is not None:
            print(f"Reward counts so far: {reward_counts}")
        cur_loop += 1

    print(f"Added {num_transitions_added} transitions to the replay buffer")
    if reward_bins is not None:
        print(
            f"Reward bins and counts: {list(zip(list(reward_bins) + ['edge'], reward_counts))}"
        )

    return num_transitions_added


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly")
    args = parse_args()
    train(args)


# def offline_warm_start(model, obs, actions, state_encoder, reward_mlp, task_reps, num_task_reps_per_obs, num_transitions, reward_bins):
#     replay_buffer = model.replay_buffer
#     print("Filling replay buffer...")

#     # Determine the device of the neural networks
#     device = next(state_encoder.parameters()).device

#     # Preprocess the task representations to the correct device
#     task_reps = task_reps.to(device)

#     all_rewards = np.array([])

#     # num_tasks = num_transitions // (obs.shape[1] * obs.shape[2])
#     # print(f"Data has {obs.shape[0]} tasks, {obs.shape[1]} trajectories, and {obs.shape[2]} steps")
#     # print(f"Total transitions: {obs.shape[0] * obs.shape[1] * obs.shape[2]}")
#     # print(f"Subsampling {num_tasks} tasks for {num_tasks*obs.shape[1]*obs.shape[2]} ~= {num_transitions} transitions")

#     if reward_bins is not None:
#         reward_counts = np.zeros(len(reward_bins)+1) # n-1 internal bins, 2 edge bins

#     # for task_idx in tqdm(range(obs.shape[0])):
#     # for task_idx in tqdm(range(num_tasks)):
#     # for task_idx in tqdm(range(2)):
#     num_transitions_added = 0
#     while num_transitions_added < num_transitions:
#         for traj_idx in range(obs.shape[1]):
#             max_transitions = (obs.shape[2]-1) * num_task_reps_per_obs
#             states = np.zeros((max_transitions, obs.shape[3] + task_reps.shape[1] - 21))
#             actions_batch = np.zeros((max_transitions, actions.shape[3]))
#             rewards_batch = np.zeros((max_transitions,))
#             next_states = np.zeros((max_transitions, obs.shape[3] + task_reps.shape[1] - 21))
#             transition_count = 0

#             states_full = obs[task_idx, traj_idx, :-1]
#             actions_full = actions[task_idx, traj_idx]
#             next_states_full = obs[task_idx, traj_idx, 1:]

#             indices = torch.randint(0, len(task_reps), size=(max_transitions,))
#             random_task_reps = task_reps[indices]

#             state_torch = torch.tensor(states_full, dtype=torch.float32).to(device)
#             state_rep = state_encoder(state_torch)

#             if reward_mlp is not None:
#                 state_rep_repeated = state_rep.repeat_interleave(num_task_reps_per_obs, dim=0)
#                 rewards = reward_mlp(state_rep_repeated, random_task_reps).detach().cpu().numpy()
#             else:
#                 rewards = torch.matmul(state_rep, random_task_reps.T).detach().cpu().numpy()

#             # all_rewards = np.append(all_rewards, rewards)

#             states_repeated = np.repeat(states_full, num_task_reps_per_obs, axis=0)
#             next_states_repeated = np.repeat(next_states_full, num_task_reps_per_obs, axis=0)
#             actions_repeated = np.repeat(actions_full, num_task_reps_per_obs, axis=0)
#             random_task_reps_np = random_task_reps.detach().cpu().numpy()

#             states_batch = np.concatenate((states_repeated, random_task_reps_np), axis=1)[..., 21:]
#             next_states_batch = np.concatenate((next_states_repeated, random_task_reps_np), axis=1)[..., 21:]

#             # We need to ensure the replay buffer can handle batches divisible by model.n_envs
#             if len(states_batch) % model.n_envs != 0:
#                 num_transitions_to_keep = (len(states_batch) // model.n_envs) * model.n_envs
#                 indices = np.random.choice(len(states_batch), num_transitions_to_keep, replace=False)
#                 states_batch = states_batch[indices]
#                 next_states_batch = next_states_batch[indices]
#                 actions_repeated = actions_repeated[indices]
#                 rewards = rewards[indices]

#             for i in range(0, len(states_batch), model.n_envs):
#                 try:
#                     replay_buffer.add(states_batch[i:i+model.n_envs], next_states_batch[i:i+model.n_envs], actions_repeated[i:i+model.n_envs], rewards[i:i+model.n_envs], np.zeros(model.n_envs, dtype=bool), [{}]*model.n_envs)
#                 except:
#

#     print(f"Added {replay_buffer.pos} transitions to the replay buffer")

#     # for decile in range(10, 100, 10):
#     #     print(f"{decile}th percentile: {np.percentile(all_rewards, decile)}")

# def offline_warm_start(model, obs, actions, state_encoder, reward_mlp, task_reps, num_task_reps_per_obs, num_transitions, reward_bins):
#     replay_buffer = model.replay_buffer
#     device = next(state_encoder.parameters()).device
#     task_reps = task_reps.to(device)

#     if reward_bins is not None:
#         reward_counts = np.zeros(len(reward_bins) + 1)
#         num_bins = len(reward_bins) + 1  # Total bins
#         epsilon = 1e-5  # Small constant to prevent division by zero

#     num_transitions_added = 0
#     total_count = num_transitions  # Total number of transitions to add

#     for task_idx in range(obs.shape[0]):
#         states_full = obs[task_idx, :, :-1]
#         actions_full = actions[task_idx]
#         next_states_full = obs[task_idx, :, 1:]

#         # Flatten the arrays to simplify batch processing
#         states_full = states_full.reshape(-1, states_full.shape[-1])
#         actions_full = actions_full.reshape(-1, actions_full.shape[-1])
#         next_states_full = next_states_full.reshape(-1, next_states_full.shape[-1])

#         indices = torch.randint(0, len(task_reps), size=(len(states_full),))
#         random_task_reps = task_reps[indices]

#         state_torch = torch.tensor(states_full, dtype=torch.float32).to(device)
#         state_rep = state_encoder(state_torch)

#         if reward_mlp is not None:
#             rewards = reward_mlp(state_rep, random_task_reps).detach().cpu().numpy()
#         else:
#             rewards = torch.matmul(state_rep, random_task_reps.T).detach().cpu().numpy()

#         if reward_bins is not None:
#             # Batching reward bin assignment
#             bin_indices = np.digitize(rewards, reward_bins, right=True)
#             current_counts = reward_counts[bin_indices]
#             accept_probs = np.minimum((1 / num_bins) / ((current_counts + epsilon) / total_count), 1)

#             # Determine which transitions to accept
#             accepted = np.random.rand(len(accept_probs)) < accept_probs
#             num_accepted = accepted.sum()

#             # Add accepted transitions to the buffer
#             if num_accepted > 0:
#                 replay_buffer.add_batch(states_full[accepted], next_states_full[accepted], actions_full[accepted], rewards[accepted], np.zeros(num_accepted, dtype=bool), [{}] * num_accepted)
#                 reward_counts[bin_indices[accepted]] += 1
#                 num_transitions_added += num_accepted
#         else:
#             replay_buffer.add_batch(states_full, next_states_full, actions_full, rewards, np.zeros(len(states_full), dtype=bool), [{}] * len(states_full))
#             num_transitions_added += len(states_full)

#         if num_transitions_added >= num_transitions:
#             break

#     print(f"Added {num_transitions_added} transitions to the replay buffer")

#     print(f"Reward bins and counts: {list(zip(list(reward_bins) + ['edge'], reward_counts))}")

# def offline_warm_start(model, obs, actions, state_encoder, reward_mlp, task_reps, num_task_reps_per_obs, num_transitions, reward_bins):
#     replay_buffer = model.replay_buffer
#     device = next(state_encoder.parameters()).device
#     task_reps = task_reps.to(device)

#     if reward_bins is not None:
#         reward_counts = np.zeros(len(reward_bins) + 1)
#         num_bins = len(reward_bins) + 1  # Total bins
#         epsilon = 1e-5  # Small constant to prevent division by zero

#     num_transitions_added = 0
#     total_count = num_transitions  # Total number of transitions to add

#     for task_idx in range(obs.shape[0]):
#         states_full = obs[task_idx, :, :-1]
#         actions_full = actions[task_idx]
#         next_states_full = obs[task_idx, :, 1:]

#         # Flatten the arrays to simplify batch processing
#         states_full = states_full.reshape(-1, states_full.shape[-1])
#         actions_full = actions_full.reshape(-1, actions_full.shape[-1])
#         next_states_full = next_states_full.reshape(-1, next_states_full.shape[-1])

#         indices = torch.randint(0, len(task_reps), size=(len(states_full),))
#         random_task_reps = task_reps[indices]

#         state_torch = torch.tensor(states_full, dtype=torch.float32).to(device)
#         state_rep = state_encoder(state_torch)

#         if reward_mlp is not None:
#             rewards = reward_mlp(state_rep, random_task_reps).detach().cpu().numpy()
#         else:
#             rewards = torch.matmul(state_rep, random_task_reps.T).detach().cpu().numpy()

#         if reward_bins is not None:
#             # Batching reward bin assignment
#             bin_indices = np.digitize(rewards, reward_bins, right=True)
#             current_counts = reward_counts[bin_indices]
#             accept_probs = np.minimum((1 / num_bins) / ((current_counts + epsilon) / total_count), 1)

#             # Determine which transitions to accept
#             accepted = np.random.rand(len(accept_probs)) < accept_probs
#             num_accepted = accepted.sum()

#             # Add accepted transitions to the buffer
#             if num_accepted > 0:
#                 # Ensure the number of transitions is a multiple of n_envs
#                 if num_accepted % model.n_envs != 0:
#                     num_accepted = (num_accepted // model.n_envs) * model.n_envs
#                     accepted_indices = np.where(accepted)[0]
#                     accepted_indices = np.random.choice(accepted_indices, num_accepted, replace=False)
#                     accepted = np.zeros_like(accepted, dtype=bool)
#                     accepted[accepted_indices] = True

#                 for i in range(0, num_accepted, model.n_envs):
#                     replay_buffer.add(states_full[accepted][i:i+model.n_envs], next_states_full[accepted][i:i+model.n_envs], actions_full[accepted][i:i+model.n_envs], rewards[accepted][i:i+model.n_envs], np.zeros(model.n_envs, dtype=bool), [{}]*model.n_envs)
#                 reward_counts[bin_indices[accepted]] += 1
#                 num_transitions_added += num_accepted
#         else:
#             # Ensure the number of transitions is a multiple of n_envs
#             if len(states_full) % model.n_envs != 0:
#                 num_transitions_to_keep = (len(states_full) // model.n_envs) * model.n_envs
#                 indices = np.random.choice(len(states_full), num_transitions_to_keep, replace=False)
#                 states_full = states_full[indices]
#                 next_states_full = next_states_full[indices]
#                 actions_full = actions_full[indices]
#                 rewards = rewards[indices]

#             for i in range(0, len(states_full), model.n_envs):
#                 replay_buffer.add(states_full[i:i+model.n_envs], next_states_full[i:i+model.n_envs], actions_full[i:i+model.n_envs], rewards[i:i+model.n_envs], np.zeros(model.n_envs, dtype=bool), [{}]*model.n_envs)
#             num_transitions_added += len(states_full)

#         if num_transitions_added >= num_transitions:
#             break

#     print(f"Added {num_transitions_added} transitions to the replay buffer")

#     print(f"Reward bins and counts: {list(zip(list(reward_bins) + ['edge'], reward_counts))}")
