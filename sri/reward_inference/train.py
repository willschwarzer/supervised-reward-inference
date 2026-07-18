import os
from sri.utils import generate_gadget_demonstration, scale, get_reward_bins
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from sri.reward_inference.models import *
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import (
    compute_reward as reach_compute_reward,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import (
    compute_reward_batch as reach_compute_reward_batch,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import (
    REWARD_DIMS as REACH_REWARD_DIMS,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import (
    compute_reward as pick_place_compute_reward,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import (
    compute_reward_batch as pick_place_compute_reward_batch,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import (
    REWARD_DIMS as PICK_PLACE_REWARD_DIMS,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import (
    compute_reward as push_compute_reward,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import (
    compute_reward_batch as push_compute_reward_batch,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import (
    REWARD_DIMS as PUSH_REWARD_DIMS,
)
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import HAND_INIT_POS
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from metaworld.policies.sawyer_pick_place_v2_policy import (
    SawyerPickPlaceV2Policy
)
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_DEM_PER_REWARD = 100
NUM_AGENTS_PER_GPU = 8
rng = np.random.default_rng()


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised IRL models")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument(
        "--save-model",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to save the model after every epoch",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="use mlp for prediction from dem rep and state rep; not for TST",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="proportion of data to be in train split",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=str,
        default=None,
        help="Location of saved model to load. Must also have shuffle_idxs.npy in this directory to prevent data leakage",
    )
    parser.add_argument(
        "--skip-reward-training",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to skip training the reward model (only used when being called from run_experiment.py)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="sirl")
    parser.add_argument("--env", default="reach", type=str)
    parser.add_argument(
        "--direct-goal-inference", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num-demonstration-layers", default=2, type=int)
    parser.add_argument("--num-state-layers", default=2, type=int)
    parser.add_argument("--state-hidden-size", default=256, type=int)
    parser.add_argument("--demonstration-hidden-size", default=256, type=int)
    parser.add_argument("--reward-hidden-size", default=64, type=int)
    parser.add_argument(
        "--dem-encoder-type",
        type=str,
        default="transformer",
        help="transformer or lstm or set_transformer",
    )
    parser.add_argument("--demonstration-rep-dim", default=100, type=int)
    parser.add_argument(
        "--internal-tst-dim",
        default=100,
        type=int,
        help="dimension of the demonstration encoding before passing to set transformer",
    )
    parser.add_argument("--state-rep-dim", default=100, type=int)
    parser.add_argument(
        "--use-shuffled", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--num-rings", default=5, type=int, help="For ring env only")
    parser.add_argument("--data-location", type=str, default="rings_multimove")
    parser.add_argument(
        "--n",
        default=100,
        type=int,
        help="How many demonstrations to train on (if not batching by task, must be 1)",
    )
    parser.add_argument(
        "--rand-n",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Train on a random number of demonstrations within each task and epoch",
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        help="Number of iid observations to predict rewards of per task (defaults to all)",
    )
    parser.add_argument(
        "--demonstration-type", type=str, default="normal", help="normal or gadget"
    )
    parser.add_argument(
        "--shuffle",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If false, deterministically uses the first train_split*100%% of the data for training (not sure why you'd want this)",
    )
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument(
        "--batch-reward-computation",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--scale-rewards", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--include-actions", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--synthesize-obs", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--synth-prop", default=0.5, type=float)
    parser.add_argument(
        "--x-bounds",
        nargs=2,
        type=float,
        default=[-0.5, 0.5],
        help="Bounds for the x-coordinate of the synthesized observations. Default is [-0.4, 0.4].",
    )
    parser.add_argument(
        "--y-bounds",
        nargs=2,
        type=float,
        default=[0.2, 0.9],
        help="Bounds for the y-coordinate of the synthesized observations. Default is [0.4, 0.8].",
    )
    parser.add_argument(
        "--z-bounds",
        nargs=2,
        type=float,
        default=[0.05, 0.5],
        help="Bounds for the z-coordinate of the synthesized observations. Default is [0.05, 0.3].",
    )
    parser.add_argument(
        "--synth-on-grid",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to synthesize observations on a grid (or, if not, synthesize randomly). Default is False.",
    )
    parser.add_argument(
        "--synth-grid-size",
        default=10,
        type=int,
        help="Number of points to synthesize observations at on each axis. Default is 10.",
    )
    parser.add_argument(
        "--synth-edges-only",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to only synthesize observations at extreme points. Default is False.",
    )
    parser.add_argument(
        "--only-synth-tcp", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--dem-horizon",
        default=50,
        type=int,
        help="Number of steps in each demonstration",
    )
    parser.add_argument(
        "--only-synthetic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to only use synthetic state data for training. Default is False.",
    )
    parser.add_argument(
        "--extra-success-reward",
        type=float,
        default=0.0,
        help="Extra reward for reaching the goal. Default is 0.0.",
    )
    parser.add_argument(
        "--gt-hand-init-pos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use ground-truth hand initial positions for synthetic obs. Default is True.",
    )
    parser.add_argument(
        "--synth-frame-stacking",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If False, synthetic obs use synthesized obs as previous frame as well. Default is False.",
    )
    parser.add_argument(
        "--limit-reward-obs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to limit the reward model to only observe relevant variables. Default is False.",
    )
    parser.add_argument(
        "--limit-dem-obs",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to limit the demonstration encoder to only observe relevant variables. Default is False.",
    )
    parser.add_argument(
        "--proximity-reward",
        default=0.0,
        type=float,
        help="Extra reward for being close to the object. Default is 0.0. Only for pick-place.",
    )
    args = parser.parse_args()

    max_grid_points = args.synth_grid_size**3
    if args.synth_edges_only:
        max_grid_points = 6 * args.synth_grid_size**2 - 12 * args.synth_grid_size + 8
    num_obs = (
        args.num_obs if args.num_obs is not None else args.n * (args.dem_horizon + 1)
    )
    max_synth_prop = max_grid_points / num_obs
    adjusted_synth_prop = min(args.synth_prop, max_synth_prop)
    if adjusted_synth_prop < args.synth_prop:
        print(
            f"Reducing synth_prop from {args.synth_prop} to {adjusted_synth_prop} to fit within grid size"
        )
    args.synth_prop = adjusted_synth_prop

    assert args.env in [
        "rings",
        "reach",
        "pick-place",
    ], "Only rings and metaworld environments supported atm"
    return args


def custom_collate_fn(batch):
    # useful for padding when we have variable length demonstrations
    max_n = max(len(item[0]) for item in batch)  # Assuming item[0] is 'return_dems'
    padded_batch = []

    for dems, obs, goals in batch:
        padding_size = max_n - len(dems)
        padding_tensor = torch.zeros(padding_size, dems.shape[1], dems.shape[2]).cuda()
        padded_dems = torch.cat([dems, padding_tensor], dim=0)

        padded_batch.append((padded_dems, obs, goals))

    dems_batch, obs_batch, goals_batch = zip(*padded_batch)
    # states_batch, rewards_batch, weights_batch are all currently tuples of tensors
    # we need to convert them to tensors
    states_batch = torch.stack(states_batch)
    # if type(weights_batch[0]) != int:
    #     weights_batch = torch.stack(weights_batch)
    # not sure why this is here?
    assert type(goals_batch[0]) == int
    dems_batch = torch.stack(dems_batch)

    return dems_batch, obs_batch, goals_batch


# Dataset for data organized by goal
# Allows for us to do n-shot, as well as using iid states separate from the
# demonstration trajectory for inference
class TaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dems,
        obs1,
        obs2,
        actions,
        goals,
        n=1,
        rand_n=False,
        num_obs=None,
        demonstration_type="normal",
        weights_only_for_gadget=True,
        synthesize_obs=False,
        limit_reward_obs=False,
        limit_dem_obs=False,
        config=None,
        start_prop_obs_1=1.0,
        end_prop_obs_1=1.0,
        num_epochs=1,
        dem_obs_share_goals=False,
    ):
        self.dems = torch.Tensor(dems).to(torch.float)
        self.actions = (
            torch.Tensor(actions).to(torch.float) if actions is not None else None
        )
        self.goals = torch.Tensor(goals).to(torch.float)
        self.n = n
        self.rand_n = rand_n
        self.demonstration_type = demonstration_type.split(",")
        assert (
            "gadget" not in self.demonstration_type
        ), "Gadget hasn't been supported for a while"
        self.num_obs = num_obs
        self.synthesize_obs = synthesize_obs
        self.limit_reward_obs = limit_reward_obs
        self.limit_dem_obs = limit_dem_obs
        self.start_prop_obs_1 = start_prop_obs_1
        self.end_prop_obs_1 = end_prop_obs_1
        self.num_epochs = num_epochs
        self.cur_epoch = 0
        self.config = config
        self.dem_obs_share_goals = dem_obs_share_goals

        self.have_obs_1 = obs1 is not None
        self.have_obs_2 = obs2 is not None
        self.have_obs = self.have_obs_1 or self.have_obs_2

        if dem_obs_share_goals:
            # Observations are per goal
            self.obs1 = (
                [torch.Tensor(o).to(torch.float) for o in obs1]
                if obs1 is not None
                else None
            )
            self.obs2 = (
                [torch.Tensor(o).to(torch.float) for o in obs2]
                if obs2 is not None
                else None
            )
        else:
            # Observations are global
            self.obs_flat_1 = (
                torch.Tensor(obs1).to(torch.float) if obs1 is not None else None
            )
            self.obs_flat_2 = (
                torch.Tensor(obs2).to(torch.float) if obs2 is not None else None
            )

        self.process_auto_prop_obs()

        print("Pre-sampling observations")
        # if obs1 is not None and obs2 is not None:
        self.pre_sample_observations()
        print(f"Dataset initialized with {len(self)} tasks")

    def process_auto_prop_obs(self):
        if not self.have_obs_1:
            self.start_prop_obs_1 = 0
            self.end_prop_obs_1 = 0
            return
        elif not self.have_obs_2:
            self.start_prop_obs_1 = 1
            self.end_prop_obs_1 = 1
            return
        if self.start_prop_obs_1 == "auto":
            # self.start_prop_obs_1 = len(self.obs1) / (len(self.obs1) + len(self.obs2))
            if self.dem_obs_share_goals:
                self.start_prop_obs_1 = len(self.obs1[0]) / (
                    len(self.obs1[0]) + len(self.obs2[0])
                )
            else:
                self.start_prop_obs_1 = len(self.obs_flat_1) / (
                    len(self.obs_flat_1) + len(self.obs_flat_2)
                )
        if self.end_prop_obs_1 == "auto":
            # self.end_prop_obs_1 = len(self.obs1) / (len(self.obs1) + len(self.obs2))
            if self.dem_obs_share_goals:
                self.end_prop_obs_1 = len(self.obs1[0]) / (
                    len(self.obs1[0]) + len(self.obs2[0])
                )
            else:
                self.end_prop_obs_1 = len(self.obs_flat_1) / (
                    len(self.obs_flat_1) + len(self.obs_flat_2)
                )

    def pre_sample_observations(self):
        # set obs_1 sample proportion: if both start and end are less than or greater than 0.5,
        # set to the one closer to 0.5
        # otherwise, they disagree, so set to 0.5
        if self.start_prop_obs_1 > 0.5 and self.end_prop_obs_1 > 0.5:
            sample_prop_obs_1 = min(self.start_prop_obs_1, self.end_prop_obs_1)
        elif self.start_prop_obs_1 < 0.5 and self.end_prop_obs_1 < 0.5:
            sample_prop_obs_1 = max(self.start_prop_obs_1, self.end_prop_obs_1)
        else:
            sample_prop_obs_1 = 0.5
        if self.dem_obs_share_goals:
            # Observations are per goal
            # sample sample_prop_obs_1*num_obs observations from obs1 and the rest from obs2
            if self.have_obs_1:
                self.obs1 = [
                    o[torch.randperm(len(o))[: int(sample_prop_obs_1 * self.num_obs)]]
                    for o in self.obs1
                ]
            if self.have_obs_2:
                self.obs2 = [
                    o[
                        torch.randperm(len(o))[
                            : int((1 - sample_prop_obs_1) * self.num_obs)
                        ]
                    ]
                    for o in self.obs2
                ]
        else:
            # Observations are global
            # sample sample_prop_obs_1*num_obs observations from obs1 and the rest from obs2
            # but in this case, we create a non-flat list of tensors
            if self.have_obs_1:
                # self.obs1 = [
                #     self.obs_flat_1[
                #         torch.randperm(len(self.obs_flat_1))[
                #             : int(sample_prop_obs_1 * self.num_obs)
                #         ]
                #     ]
                #     for _ in range(len(self.dems))
                # ]
                # self.obs1 = []
                # for _ in tqdm(range(len(self.dems))):
                #     obs1 = self.obs_flat_1[
                #         torch.randperm(len(self.obs_flat_1))[
                #             : int(sample_prop_obs_1 * self.num_obs)
                #         ]
                #     ]
                #     self.obs1.append(obs1)
                num_samples = int(sample_prop_obs_1 * self.num_obs)
                indices = np.random.choice(
                    len(self.obs_flat_1), num_samples * len(self.dems), replace=False
                )
                obs1_samples = self.obs_flat_1[indices].view(
                    len(self.dems), num_samples, -1
                )
                self.obs1 = [obs1_samples[i] for i in range(len(self.dems))]
            if self.have_obs_2:
                # self.obs2 = [
                #     self.obs_flat_2[
                #         torch.randperm(len(self.obs_flat_2))[
                #             : int((1 - sample_prop_obs_1) * self.num_obs)
                #         ]
                #     ]
                #     for _ in range(len(self.dems))
                # ]
                num_samples = int((1 - sample_prop_obs_1) * self.num_obs)
                indices = np.random.choice(
                    len(self.obs_flat_2), num_samples * len(self.dems), replace=False
                )
                obs2_samples = self.obs_flat_2[indices].view(
                    len(self.dems), num_samples, -1
                )
                self.obs2 = [obs2_samples[i] for i in range(len(self.dems))]

    def update_epoch(self, epoch):
        self.cur_epoch = epoch

    def __len__(self):
        return len(self.dems)

    def __getitem__(self, index):
        dems = self.dems[index]
        return_goals = self.goals[index]

        if self.rand_n:
            # n = rng.integers(0, self.n, endpoint=True) # zero-shot!
            # we'll do geometric distribution instead, to bias towards smaller n
            p = 0.005  # bias parameter, larger means more likely to be small
            # should eventually compute this based on n instead
            n = self.n
            while n >= self.n:
                n = (
                    rng.geometric(p) - 1
                )  # subtract 1 because geometric distribution starts from 1
        else:
            n = self.n
        shuffled_dems = dems[torch.randperm(len(dems))]
        return_dems = shuffled_dems[:n]

        # Apply synthesis and observation limiting if required

        if self.have_obs:
            # Compute the current proportion
            if self.num_epochs > 1:
                prop_obs_1 = self.start_prop_obs_1 + (
                    (self.end_prop_obs_1 - self.start_prop_obs_1)
                    * (self.cur_epoch / (self.num_epochs - 1))
                )
            else:
                prop_obs_1 = self.end_prop_obs_1

            assert (
                0 <= prop_obs_1 <= 1
            ), f"prop_obs_1 is {prop_obs_1}, that doesn't make sense"

            num_obs_from_obs1 = int(round(self.num_obs * prop_obs_1))
            num_obs_from_obs2 = self.num_obs - num_obs_from_obs1

            # Adjust based on availability
            if not self.have_obs_1:
                num_obs_from_obs1 = 0
                num_obs_from_obs2 = self.num_obs
            if not self.have_obs_2:
                num_obs_from_obs2 = 0
                num_obs_from_obs1 = self.num_obs
            if not self.have_obs_1 and not self.have_obs_2:
                raise ValueError("No observations available.")

            # Sample from obs1
            if self.have_obs_1:
                num_obs1_available = len(self.obs1[index])
                num_obs1_without_replacement = min(
                    num_obs1_available, num_obs_from_obs1
                )
                obs1_samples = self.obs1[index][
                    torch.randperm(num_obs1_available)[:num_obs1_without_replacement]
                ]
                if num_obs_from_obs1 > num_obs1_without_replacement:
                    additional_obs1 = self.obs1[index][
                        torch.randint(
                            num_obs1_available,
                            (num_obs_from_obs1 - num_obs1_without_replacement,),
                        )
                    ]
                    obs1 = torch.cat([obs1_samples, additional_obs1], dim=0)
                else:
                    obs1 = obs1_samples
            else:
                obs1 = torch.empty((0, self.obs2[index].shape[-1]))

            # Sample from obs2
            if self.have_obs_2:
                num_obs2_available = len(self.obs2[index])
                num_obs2_without_replacement = min(
                    num_obs2_available, num_obs_from_obs2
                )
                obs2_samples = self.obs2[index][
                    torch.randperm(num_obs2_available)[:num_obs2_without_replacement]
                ]
                if num_obs_from_obs2 > num_obs2_without_replacement:
                    additional_obs2 = self.obs2[index][
                        torch.randint(
                            num_obs2_available,
                            (num_obs_from_obs2 - num_obs2_without_replacement,),
                        )
                    ]
                    obs2 = torch.cat([obs2_samples, additional_obs2], dim=0)
                else:
                    obs2 = obs2_samples
            else:
                obs2 = torch.empty((0, self.obs1[index].shape[-1]))

            # Combine and shuffle
            return_obs = torch.cat([obs1, obs2], dim=0)
            return_obs = return_obs[torch.randperm(len(return_obs))]
            if self.synthesize_obs:
                return_obs = self.get_synthetic_obs(return_obs, index)
            if self.limit_reward_obs:
                return_obs = self.filter_obs_dims(return_obs)
        if self.limit_dem_obs:
            # dems = self.filter_obs_dims(dems)
            return_dems = self.filter_obs_dims(return_dems)

        if torch.cuda.is_available():
            # dems = dems.cuda()
            return_dems = return_dems.cuda()
            return_goals = return_goals.cuda()
            if self.have_obs:
                return_obs = return_obs.cuda()
        if self.have_obs:
            return return_dems, return_obs, return_goals
        else:
            return return_dems, return_goals

        # # Apply synthesis and observation limiting if required
        # if self.synthesize_obs:
        #     return_obs = self.get_synthetic_obs(return_obs, index)
        # if self.limit_reward_obs:
        #     return_obs = self.filter_obs_dims(return_obs)
        # if self.limit_dem_obs:
        #     dems = self.filter_obs_dims(dems)

        # if torch.cuda.is_available():
        #     dems = dems.cuda()
        #     return_obs = return_obs.cuda()
        #     return_goals = return_goals.cuda()
        # return dems, return_obs, return_goals

    def get_synthetic_obs(self, obs, index):
        if not self.only_synthetic:
            synthesis_idxs = rng.choice(
                len(obs), int(self.synth_prop * len(obs)), replace=False
            )  # sample without replacement
        else:
            synthesis_idxs = np.arange(int(self.synth_prop * len(obs)))
        if self.synth_on_grid:
            num_points = self.synth_grid_size**3
            synthesis_idxs = synthesis_idxs[:num_points]

        if not self.only_synthetic:
            # in this case we're including some real observations, so we need to copy them over
            new_obs = obs.clone()
        else:
            # in this case we're just using as many synthetic observations as we have
            new_obs = torch.zeros_like(obs)[synthesis_idxs]

        if self.minimal_synth:
            assert not self.only_synthetic
            # just put the object in init_obj_pos
            # (also need to do this with the previous frame)
            # not going to worry about the obj quats for now
            # (we might also want the model to learn that those don't matter)
            assert len(new_obs.shape) == 2
            if not self.minimal_rand_pos:
                new_obj_pos = new_obs[:, 15:18]
            else:
                new_obj_pos = self.rand_obj_pos[index]
            if self.minimal_only:
                new_obs[synthesis_idxs, 25:28] = new_obj_pos[synthesis_idxs]
                new_obs[synthesis_idxs, 43:46] = new_obj_pos[synthesis_idxs]
                return new_obs
            else:
                minimal_idxs = synthesis_idxs[: len(synthesis_idxs) // 2]
                synthesis_idxs = synthesis_idxs[len(synthesis_idxs) // 2 :]
                new_obs[minimal_idxs, 25:28] = new_obj_pos[minimal_idxs]
                new_obs[minimal_idxs, 43:46] = new_obj_pos[minimal_idxs]

        # now we need to synthesize these observations
        # in particular, we will synthesize the left and right finger positions, the hand position, and the "tcp_center" position
        # we will do this by sampling tcp_center uniformly from the bounds, and then setting the other positions
        # based on their average displacement in the data from the tcp_center
        # left and right finger positions: 0:3, 3:6
        # tcp_center: 12:15
        # hand position: 21:24
        # we also need to synthesize the frame-stacked previous observation's hand position, which is at 39:42
        # we'll do this by figuring out the average absolute displacement of the two frames' hand positions
        # and then sampling a new hand position in a circle around the new hand position with that radius
        if self.synth_on_grid:
            x = torch.linspace(self.x_bounds[0], self.x_bounds[1], self.synth_grid_size)
            y = torch.linspace(self.y_bounds[0], self.y_bounds[1], self.synth_grid_size)
            z = torch.linspace(self.z_bounds[0], self.z_bounds[1], self.synth_grid_size)
            grid = torch.stack(torch.meshgrid(x, y, z), -1).view(-1, 3)

            if self.synth_edges_only:
                mask = (
                    (grid == self.x_bounds[0])
                    | (grid == self.x_bounds[1])
                    | (grid == self.y_bounds[0])
                    | (grid == self.y_bounds[1])
                    | (grid == self.z_bounds[0])
                    | (grid == self.z_bounds[1])
                )
                grid = grid[mask.any(dim=-1)]

            if len(grid) <= len(synthesis_idxs):
                new_tcp_centers = grid
            else:
                new_tcp_centers = grid[
                    rng.choice(len(grid), len(synthesis_idxs), replace=False)
                ]
        else:
            new_tcp_centers = torch.stack(
                [
                    torch.empty(len(synthesis_idxs)).uniform_(
                        self.x_bounds[0], self.x_bounds[1]
                    ),
                    torch.empty(len(synthesis_idxs)).uniform_(
                        self.y_bounds[0], self.y_bounds[1]
                    ),
                    torch.empty(len(synthesis_idxs)).uniform_(
                        self.z_bounds[0], self.z_bounds[1]
                    ),
                ],
                dim=-1,
            )
        new_obs[synthesis_idxs, 12:15] = new_tcp_centers
        if not self.only_synth_tcp:
            # synthesize new end effector positions (just offset from tcp)
            # also left and right finger positions, also just offsets
            # XXX this is not prepared for push or pick-place
            ave_left_displacement = torch.mean(obs[:, 0:3] - obs[:, 12:15], dim=0)
            ave_right_displacement = torch.mean(obs[:, 3:6] - obs[:, 12:15], dim=0)
            ave_hand_displacement = torch.mean(obs[:, 21:24] - obs[:, 12:15], dim=0)
            if not self.synth_random_grip:
                new_left_positions = new_tcp_centers + ave_left_displacement
                new_right_positions = new_tcp_centers + ave_right_displacement
            else:
                # take the average zx displacement, but randomize the y
                # specifically, take random uniform left displacement between 0 and 0.05, then right = -left
                # (obviously, we shouldn't expect these numbers to be realistic, but as long as
                # they contain the realistic grips we're okay)
                # ave_right_zy_displacement = ave_right_displacement[1:]
                # ave_left_zy_displacement = ave_left_displacement[1:]
                # ave_right_zy_displacement_repeated = ave_right_zy_displacement.repeat(len(synthesis_idxs), 1)
                # ave_left_zy_displacement_repeated = ave_left_zy_displacement.repeat(len(synthesis_idxs), 1)
                # right_x_displacement = torch.empty(len(synthesis_idxs)).uniform_(0, 0.025)
                # new_right_positions = new_tcp_centers + torch.cat(
                #     [right_x_displacement.unsqueeze(-1), ave_right_zy_displacement_repeated],
                #     dim=-1,
                # )
                # new_left_positions = new_tcp_centers + torch.cat(
                #     [-right_x_displacement.unsqueeze(-1), ave_left_zy_displacement_repeated],
                #     dim=-1,
                # )
                ave_right_zx_displacement = ave_right_displacement[[0, 2]]
                ave_left_zx_displacement = ave_left_displacement[[0, 2]]
                ave_right_zx_displacement_repeated = ave_right_zx_displacement.repeat(
                    len(synthesis_idxs), 1
                )
                ave_left_zx_displacement_repeated = ave_left_zx_displacement.repeat(
                    len(synthesis_idxs), 1
                )
                left_y_displacement = torch.empty(len(synthesis_idxs)).uniform_(0, 0.05)
                new_left_positions = new_tcp_centers + torch.cat(
                    [
                        ave_left_zx_displacement_repeated,
                        left_y_displacement.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                new_right_positions = new_tcp_centers + torch.cat(
                    [
                        ave_right_zx_displacement_repeated,
                        -left_y_displacement.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                new_gripper_distances_apart = torch.norm(
                    new_left_positions - new_right_positions, dim=-1
                )
                new_obs[synthesis_idxs, 24] = new_gripper_distances_apart
            new_hand_positions = new_tcp_centers + ave_hand_displacement
            new_obs[synthesis_idxs, 21:24] = new_hand_positions
            new_obs[synthesis_idxs, 0:3] = new_left_positions
            new_obs[synthesis_idxs, 3:6] = new_right_positions

            if self.synth_frame_stacking:
                # ave_prev_hand_displacement_norm = torch.mean(torch.norm(obs[1:, 21:24] - obs[:-1, 21:24], dim=-1))
                ave_prev_hand_displacement_norm = torch.mean(
                    torch.norm(obs[:, 21:24] - obs[:, 39:42], dim=-1)
                )

                # Generate random angles (azimuthal and polar) for new hand positions
                theta = torch.empty(len(synthesis_idxs)).uniform_(0, 2 * np.pi)
                phi = torch.empty(len(synthesis_idxs)).uniform_(0, np.pi)

                # Generate new previous hand positions
                # new_prev_hand_positions = (
                #     new_hand_positions
                #     + ave_prev_hand_displacement_norm
                #     * torch.stack(
                #         [
                #             torch.sin(phi) * torch.cos(theta),
                #             torch.sin(phi) * torch.sin(theta),
                #             torch.cos(phi),
                #         ],
                #         dim=-1,
                #     )
                # )
                displacement_vec = ave_prev_hand_displacement_norm * torch.stack(
                    [
                        torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi),
                    ],
                    dim=-1,
                )
                new_prev_hand_positions = new_hand_positions + displacement_vec
                new_obs[synthesis_idxs, 39:42] = new_prev_hand_positions

                obj_is_gripped = (
                    torch.norm(
                        new_obs[synthesis_idxs, 12:15] - new_obs[synthesis_idxs, 25:28],
                        dim=-1,
                    )
                    < 0.03
                )
                new_obs[synthesis_idxs][obj_is_gripped, 43:46] = (
                    new_obs[synthesis_idxs][obj_is_gripped, 25:28]
                    + displacement_vec[obj_is_gripped]
                )

                new_obs[synthesis_idxs, 42] = new_obs[synthesis_idxs, 24]
            else:
                new_obs[synthesis_idxs, 39:] = new_obs[synthesis_idxs, 21:39]
        else:
            new_obs[synthesis_idxs, 12:15] = new_tcp_centers

        return_obs = new_obs
        return return_obs

    def filter_obs_dims(self, obs):
        if "pick-place" in self.env or "push" in self.env:
            raise NotImplementedError
        elif "reach" in self.env:
            return obs[..., np.r_[12:15, 18:21]]


def get_distributed_obs(
    goals,
    obs,
    batch_rew_func,
    env,
    target_bin_counts,
    extra_success_reward,
    device,
    grasp_rew_only,
):
    # goals: (n_tasks, goal_dim)
    # obs: (total_num_obs, obs_dim)
    # returns: (n_tasks, num_obs, obs_dim)
    bins = get_reward_bins(env, len(target_bin_counts), extra_success_reward, False)
    batch_size = 10000
    return_obs = np.zeros((len(goals), sum(target_bin_counts), obs.shape[-1]))

    print("Gathering observations matching target reward distribution")
    for goal_idx, goal in enumerate(tqdm(goals, desc="Gathering observations")):
        current_counts = np.zeros(len(target_bin_counts))
        collected_obs = []

        for batch_idx in range(0, len(obs), batch_size):
            batch_obs = np.expand_dims(obs[batch_idx : batch_idx + batch_size], axis=0)
            batch_goal = np.expand_dims(goal, axis=0)
            batch_obs_torch = torch.from_numpy(batch_obs).to(device)
            batch_goal_torch = torch.from_numpy(batch_goal).to(device)
            if "pick-place" in env:
                batch_rew = batch_rew_func(
                    batch_obs_torch,
                    batch_goal_torch,
                    extra_success_reward,
                    False,
                    grasp_rew_only=grasp_rew_only,
                ).squeeze(0)
            else:
                batch_rew = batch_rew_func(
                    batch_obs_torch, batch_goal_torch, extra_success_reward, False
                ).squeeze(0)
            batch_rew = batch_rew.cpu().detach().numpy()
            rew_bins = np.digitize(batch_rew, bins) - 1

            # Create a mask that allows observations up to the target_bin_counts
            mask = np.zeros_like(rew_bins, dtype=bool)
            for bin_idx in range(len(target_bin_counts)):
                bin_mask = rew_bins == bin_idx
                num_to_accept = int(
                    target_bin_counts[bin_idx] - current_counts[bin_idx]
                )
                if num_to_accept > 0:
                    bin_indices = np.where(bin_mask)[0][:num_to_accept]
                    mask[bin_indices] = True
                    current_counts[bin_idx] += len(bin_indices)

            # Collect the masked observations
            if mask.any():
                collected_obs.append(batch_obs[0, mask, :])

            if np.all(current_counts >= target_bin_counts):
                break

            # Update the progress bar with current_counts
            tqdm.write(f"Goal {goal_idx}: current_counts = {current_counts}")
        else:
            print(
                "Warning: not enough observations to match target reward distribution for goal",
                goal_idx,
            )

        collected_obs = np.concatenate(collected_obs, axis=0)
        return_obs[goal_idx, : collected_obs.shape[0], :] = collected_obs

    return return_obs


def load_model(
    saved_model_dir, model_config, train_config, obs_size, dem_obs_size, horizon
):
    net = NonLinearNet(
        model_config.demonstration_rep_dim,
        model_config.state_rep_dim,
        model_config.internal_tst_dim,
        model_config.state_hidden_size,
        model_config.reward_hidden_size,  # note that this isn't always used if mlp is false
        model_config.demonstration_hidden_size,
        obs_size,
        dem_obs_size,
        horizon,
        model_config.num_demonstration_layers,
        model_config.num_state_layers,
        mlp=model_config.mlp,
        dem_encoder_type=model_config.dem_encoder_type,
        output_type=model_config.output_type,
    )
    if torch.cuda.is_available():
        net = net.cuda()

    # if model_config.saved_model_dir is not None:
    if saved_model_dir is not None:
        saved_model_path = os.path.join(saved_model_dir, "model.parameters")
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        net.load_state_dict(torch.load(saved_model_path, map_location=map_location))
        print(f"Loaded model from {saved_model_path}")
    return net


def get_splits(
    dem_by_goal,
    obs_by_goal_1,
    obs_by_goal_2,
    actions_by_goal,
    goals,
    general_config,
    model_config,
    train_config,
    inference_config,
    shuffle_idxs=None,
):
    if general_config.verbose:
        print(f"¯\_(ツ)_/¯ Dems shape: {dem_by_goal.shape}")
        if obs_by_goal_1 is not None:
            print(f"¯\_(ツ)_/¯ Obs1 shape: {obs_by_goal_1.shape}")
        if obs_by_goal_2 is not None:
            print(f"¯\_(ツ)_/¯ Obs2 shape: {obs_by_goal_2.shape}")
        print(f"¯\_(ツ)_/¯ Goals shape: {goals.shape}")
    data_horizon = dem_by_goal.shape[-2]
    num_goals = len(goals)
    if shuffle_idxs is None:
        goal_idx_list = np.arange(num_goals)
        if train_config.shuffle:
            rng.shuffle(goal_idx_list)
    else:
        goal_idx_list = shuffle_idxs

    train_length = int(num_goals * train_config.train_split)
    train_idxs = goal_idx_list[:train_length]
    val_idxs = goal_idx_list[train_length:]
    train_dems = dem_by_goal[train_idxs]
    val_dems = dem_by_goal[val_idxs]
    if actions_by_goal is not None:
        train_actions = actions_by_goal[train_idxs]
        val_actions = actions_by_goal[val_idxs]
    else:
        train_actions = None
        val_actions = None
    train_goals = goals[train_idxs]
    val_goals = goals[val_idxs]

    if model_config.output_type != "goal":
        if not train_config.dem_obs_share_goals:
            assert (
                train_config.num_obs is not None
            ), "Must specify num_obs when not sharing goals"
            # Flatten the observations along the first axis
            obs_flat_1 = obs_by_goal_1.reshape(-1, obs_by_goal_1.shape[-1])
            if obs_by_goal_2 is not None:
                obs_flat_2 = obs_by_goal_2.reshape(-1, obs_by_goal_2.shape[-1])
            else:
                obs_flat_2 = None
            # Optionally, handle repeating observations if needed
            # For simplicity, let's assume the datasets are large enough
        else:
            # When dem_obs_share_goals is True, we need to split obs_by_goal_1 and obs_by_goal_2 according to train_idxs and val_idxs
            train_obs1 = obs_by_goal_1[train_idxs]
            val_obs1 = obs_by_goal_1[val_idxs]
            if obs_by_goal_2 is not None:
                train_obs2 = obs_by_goal_2[train_idxs]
                val_obs2 = obs_by_goal_2[val_idxs]
            else:
                train_obs2 = None
                val_obs2 = None
    else:
        # If the model output type is "goal", we don't even use the observations
        obs_flat_1 = None
        obs_flat_2 = None
        train_obs1 = None
        val_obs1 = None
        train_obs2 = None
        val_obs2 = None

    if not train_config.skip_reward_training:
        if not train_config.dem_obs_share_goals:
            train_dataset = TaskDataset(
                train_dems,
                obs_flat_1,
                obs_flat_2,
                train_actions,
                train_goals,
                train_config.n,
                train_config.rand_n,
                train_config.num_obs,
                demonstration_type=train_config.demonstration_type,
                synthesize_obs=train_config.synthesize_obs,
                limit_reward_obs=model_config.limit_reward_obs,
                limit_dem_obs=model_config.limit_dem_obs,
                config=train_config,
                start_prop_obs_1=train_config.start_prop_obs_1,
                end_prop_obs_1=train_config.end_prop_obs_1,
                num_epochs=train_config.num_epochs,
                dem_obs_share_goals=False,
            )
            val_dataset = TaskDataset(
                val_dems,
                obs_flat_1,
                obs_flat_2,
                val_actions,
                val_goals,
                train_config.n,
                train_config.rand_n,
                train_config.num_obs,
                demonstration_type=train_config.demonstration_type,
                synthesize_obs=train_config.synthesize_obs,
                limit_reward_obs=model_config.limit_reward_obs,
                limit_dem_obs=model_config.limit_dem_obs,
                config=train_config,
                start_prop_obs_1=train_config.start_prop_obs_1,
                end_prop_obs_1=train_config.end_prop_obs_1,
                num_epochs=train_config.num_epochs,
                dem_obs_share_goals=False,
            )
        else:
            train_dataset = TaskDataset(
                train_dems,
                train_obs1,
                train_obs2,
                train_actions,
                train_goals,
                train_config.n,
                train_config.rand_n,
                train_config.num_obs,
                demonstration_type=train_config.demonstration_type,
                synthesize_obs=train_config.synthesize_obs,
                limit_reward_obs=model_config.limit_reward_obs,
                limit_dem_obs=model_config.limit_dem_obs,
                config=train_config,
                start_prop_obs_1=train_config.start_prop_obs_1,
                end_prop_obs_1=train_config.end_prop_obs_1,
                num_epochs=train_config.num_epochs,
                dem_obs_share_goals=True,
            )
            val_dataset = TaskDataset(
                val_dems,
                val_obs1,
                val_obs2,
                val_actions,
                val_goals,
                train_config.n,
                train_config.rand_n,
                train_config.num_obs,
                demonstration_type=train_config.demonstration_type,
                synthesize_obs=train_config.synthesize_obs,
                limit_reward_obs=model_config.limit_reward_obs,
                limit_dem_obs=model_config.limit_dem_obs,
                config=train_config,
                start_prop_obs_1=train_config.start_prop_obs_1,
                end_prop_obs_1=train_config.end_prop_obs_1,
                num_epochs=train_config.num_epochs,
                dem_obs_share_goals=True,
            )
    else:
        train_dataset = None
        val_dataset = None

    # Create data loaders as before
    if not train_config.skip_reward_training:
        if train_config.rand_n:
            if len(train_dataset) > 0:
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=train_config.batch_size,
                    shuffle=train_config.shuffle,
                    collate_fn=custom_collate_fn,
                )
            else:
                assert (
                    train_config.skip_reward_training
                ), "If there are no training examples, we must be doing inference"
                train_dataloader = None
            validation_dataloader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=train_config.shuffle,
                collate_fn=custom_collate_fn,
            )
        else:
            if len(train_dataset) > 0:
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=train_config.batch_size,
                    shuffle=train_config.shuffle,
                )
            else:
                assert (
                    train_config.skip_reward_training
                ), "If there are no training examples, we must be doing inference"
                train_dataloader = None
            validation_dataloader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=train_config.shuffle,
            )
    else:
        train_dataloader = None
        validation_dataloader = None

    # Inference dataset and dataloader
    inference_dataset = TaskDataset(
        val_dems,
        None,
        None,
        val_actions,
        val_goals,
        inference_config.n,
        inference_config.rand_n,
        inference_config.num_obs,
        demonstration_type=inference_config.demonstration_type,
        synthesize_obs=False,
        limit_reward_obs=model_config.limit_reward_obs,
        limit_dem_obs=model_config.limit_dem_obs,
        config=inference_config,
        num_epochs=train_config.num_epochs,
        dem_obs_share_goals=train_config.dem_obs_share_goals,
    )

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_config.batch_size,
        shuffle=train_config.shuffle,
    )

    return train_dataloader, validation_dataloader, inference_dataloader, goal_idx_list


def get_batch_loss(
    net,
    loss_func,
    dem_batch,
    obs_batch,
    goal_batch,
    general_config,
    model_config,
    train_config,
    disp=False,
    policy=None,
):
    if model_config.output_type == "goal":
        prediction = net.forward(dem_batch)  # (batch_size, goal_dim)
        loss = loss_func(prediction, goal_batch)
        if (
            general_config.verbose and disp and len(dem_batch) > 1
        ):  # this breaks if batch size is 1, and I don't want to deal with that right now
            print(
                "Predicted vs actual goals: \n",
                np.array(
                    list(
                        zip(
                            prediction.cpu().detach().numpy(),
                            goal_batch.cpu().detach().numpy(),
                        )
                    )
                ),
            )
            print("Maximum error: \n", torch.max((prediction - goal_batch) ** 2))
            worst = torch.argmax((prediction - goal_batch) ** 2)
            print(
                "Argmax of error: \n",
                prediction.flatten()[worst],
                goal_batch.flatten()[worst],
            )
    elif model_config.output_type == "reward":
        prediction = net.forward(dem_batch, obs_batch)  # (batch_size, num_obs)
        rewards_batch = torch.zeros(prediction.shape).cuda()

        # Prepare common kwargs for reward functions
        compute_reward_kwargs = {
            "extra_success_reward": train_config.extra_success_reward,
            "limit_reward_obs": model_config.limit_reward_obs,
        }

        # Select the reward computation function based on environment and batch setting
        if train_config.batch_reward_computation:
            if "reach" in train_config.env:
                compute_reward_func = reach_compute_reward_batch
            elif "pick-place" in train_config.env:
                compute_reward_func = pick_place_compute_reward_batch
                compute_reward_kwargs["proximity_reward"] = (
                    train_config.proximity_reward
                )
                compute_reward_kwargs["grasp_rew_only"] = train_config.grasp_rew_only
            elif "push" in train_config.env:
                compute_reward_func = push_compute_reward_batch
            else:
                raise NotImplementedError
            # Compute rewards in batch mode
            rewards_batch = compute_reward_func(
                obs_batch,
                goal_batch,
                **compute_reward_kwargs,
            )
        else:
            if "reach" in train_config.env:
                compute_reward_func = reach_compute_reward
            elif "pick-place" in train_config.env:
                compute_reward_func = pick_place_compute_reward
            elif "push" in train_config.env:
                compute_reward_func = push_compute_reward
            else:
                raise NotImplementedError
            # Compute rewards in non-batch mode
            rewards_batch = torch.zeros_like(obs_batch[..., 0])
            for i in range(len(obs_batch)):
                for j in range(len(obs_batch[i])):
                    rewards_batch[i, j] = compute_reward_func(
                        obs_batch[i, j].cpu(),
                        goal_batch[i].cpu(),
                        **compute_reward_kwargs,
                    )
        if train_config.scale_rewards:
            if train_config.grasp_rew_only:
                maximum = 2.0
            else:
                maximum = 10.0 + train_config.extra_success_reward
            rewards_batch = scale(rewards_batch, 0, maximum)

        if train_config.pessimism_factor > 0.0:
            # shift up predicted rewards by sqrt(1+pf) wherever predicted rewards are greater than actual rewards
            # this increases the loss for these examples by a factor of (1+pf)
            pessimism_factor = train_config.pessimism_factor
            pessimism_mask = prediction > rewards_batch
            pessimism_factor = np.sqrt(1 + pessimism_factor)
            prediction[pessimism_mask] = (
                rewards_batch[pessimism_mask]
                + (prediction[pessimism_mask] - rewards_batch[pessimism_mask])
                * pessimism_factor
            )

        loss = loss_func(prediction, rewards_batch)
        if general_config.verbose and disp:
            predictions_np = prediction.cpu().detach().numpy()
            rewards_np = rewards_batch.cpu().detach().numpy()
            print(
                "Predicted vs actual rewards: \n",
                np.array(list(zip(predictions_np, rewards_np))),
            )
            print(
                "Maximum error: \n", torch.max((prediction - rewards_batch) ** 2)
            )  # XXX doesn't take into account pessimism factor
            worst = torch.argmax((prediction - rewards_batch) ** 2)
            print(
                "Argmax of error: \n",
                prediction.flatten()[worst],
                rewards_batch.flatten()[worst],
            )
            print_histograms(
                rewards_np,
                predictions_np,
                train_config.scale_rewards,
                train_config.extra_success_reward,
                train_config.grasp_rew_only,
            )
    elif model_config.output_type == "action":
        # obs_batch = obs_batch.view(
        #     obs_batch.shape[0] * obs_batch.shape[1], obs_batch.shape[-1]
        # )
        prediction = net.forward(dem_batch, obs_batch)
        prediction = prediction.view(
            obs_batch.shape[0] * obs_batch.shape[1], -1
        )  # (batch_size * num_obs, action_dim)
        # (batch_size, num_obs, action_dim)
        assert "push" not in train_config.env, "Action output not implemented for push"
        # Now we need to get the oracle actions
        # First, we need to append the goals to the observations
        # They're expected to be the last three dimensions of each observation
        obs_batch_with_goal = torch.cat(
            [obs_batch, goal_batch.unsqueeze(1).expand(-1, obs_batch.shape[1], -1)],
            dim=-1,
        )
        # Now reshape to (batch_size * num_obs, obs_dim + goal_dim)
        obs_batch_with_goal = obs_batch_with_goal.view(
            -1, obs_batch_with_goal.shape[-1]
        )
        actions_batch = policy.get_action_batch(obs_batch_with_goal)
        # # Reshape to (batch_size, num_obs, action_dim)
        # actions_batch = actions_batch.view(
        #     obs_batch.shape[0], obs_batch.shape[1], -1
        # )
        # Now we need to compute the loss
        loss = loss_func(prediction, actions_batch)
        if general_config.verbose and disp:
            predictions_np = prediction.cpu().detach().numpy()
            actions_np = actions_batch.cpu().detach().numpy()
            print(
                "Predicted vs actual actions: \n",
                # Wrapping in a list for potentially cleaner display if action_dim is large
                # For small action_dim, np.array(list(zip(predictions_np, actions_np))) is also fine
                list(zip(predictions_np, actions_np))[:10], # Print first 10 for brevity
            )
            squared_error = (prediction - actions_batch) ** 2
            print(
                "Maximum error (element-wise squared): \n", torch.max(squared_error)
            )
            # For multi-dimensional actions, argmax on flattened squared_error might be less intuitive
            # than showing the error for the specific sample and action dimension.
            # Here, we find the index of the max error in the flattened tensor.
            worst_flat_idx = torch.argmax(squared_error)
            # Convert flat index to 2D index
            worst_sample_idx = worst_flat_idx // prediction.shape[1]
            worst_dim_idx = worst_flat_idx % prediction.shape[1]
            print(
                f"Argmax of error (sample {worst_sample_idx}, dim {worst_dim_idx}): \n",
                f"Predicted: {prediction[worst_sample_idx, worst_dim_idx].item()}",
                f"Actual: {actions_batch[worst_sample_idx, worst_dim_idx].item()}",
            )
            # Histograms for multi-dimensional actions might require more specific implementation
            # depending on what aspects you want to visualize.
            # print_action_histograms(actions_np, predictions_np) # If you implement this
    else:
        raise NotImplementedError(
            f"Output type {model_config.output_type} not implemented"
        )
    wandb.log({"train loss": loss})
    return loss


def print_histograms(
    ground_truth_rewards,
    predicted_rewards,
    scale_rewards,
    extra_success_reward,
    grasp_rew_only,
):
    if not scale_rewards:
        bins = [0, 2, 4, 6, 8, 10]
    else:
        # upper_bound = last bin before maximum
        if not grasp_rew_only:
            upper_bound = 6 * (10 / (10 + extra_success_reward)) - 3
            bin_width = (upper_bound + 3) / 4
            bins = [-3 + i * bin_width for i in range(5)] + [3]
        else:
            bins = np.linspace(-3, 3, 6)

    gt_hist, gt_bin_edges = np.histogram(ground_truth_rewards, bins=bins)
    pred_hist, pred_bin_edges = np.histogram(predicted_rewards, bins=bins)

    print("Ground-truth rewards histogram:")
    for count, edge in zip(gt_hist, gt_bin_edges[:-1]):
        print(f"Bin {edge}: {count}")

    print("Predicted rewards histogram:")
    for count, edge in zip(pred_hist, pred_bin_edges[:-1]):
        print(f"Bin {edge}: {count}")


def load_data(
    dataset_path,
    obs_dataset_path,
    general_config,
    model_config,
    train_config,
    inference_config,
    shuffle_idxs=None,
    obs_dataset_path_2=None,
):
    dem_path = os.path.join(dataset_path, "observations.npy")
    goal_path = os.path.join(dataset_path, "goals.npy")
    if model_config.output_type != "goal":
        obs_path = os.path.join(obs_dataset_path, "observations.npy")
    else:
        obs_path = None
    # if model_config.output_type == "action":
    #     act_path = os.path.join(obs_dataset_path, "actions.npy")
    # else:
    #     act_path = None
    if obs_dataset_path_2 is not None and model_config.output_type != "goal":
        obs_path_2 = os.path.join(obs_dataset_path_2, "observations.npy")
        # if model_config.output_type == "action":
        #     act_path_2 = os.path.join(obs_dataset_path_2, "actions.npy")
        # else:
        #     act_path_2 = None
    else:
        obs_path_2 = None
        # act_path_2 = None
    if train_config.dem_obs_share_goals:
        assert (obs_dataset_path_2 is None
                ), "Two observation dataset goal sharing not yet implemented"
        obs_goal_path = os.path.join(obs_dataset_path, "goals.npy")

    # we fixed this in calling code, so don't need try/except
    assert os.path.exists(dem_path), "Demonstration file does not exist"
    dem_by_goal = np.load(dem_path, allow_pickle=True)
    assert os.path.exists(goal_path), "Goal file does not exist"
    if model_config.output_type != "goal":
        assert os.path.exists(obs_path), "Observation file does not exist"
        obs_by_goal_1 = np.load(obs_path, allow_pickle=True)
    else:
        obs_by_goal_1 = None
    # if model_config.output_type == "action":
    #     assert os.path.exists(act_path), "Action file does not exist"
    #     acts_by_goal_1 = np.load(act_path, allow_pickle=True)
    if obs_path_2 is not None and model_config.output_type != "goal":
        assert os.path.exists(obs_path_2), "Second obs path does not exist"
        obs_by_goal_2 = np.load(obs_path_2, allow_pickle=True)
        # if model_config.output_type == "action":
        #     assert os.path.exists(act_path_2), "Second action path does not exist"
        #     acts_by_goal_2 = np.load(act_path_2, allow_pickle=True)
    else:
        obs_by_goal_2 = None
        # acts_by_goal_2 = None

    # if model_config.limit_reward_obs:
    #     if "reach" in train_config.env:
    #         obs_size = len(REACH_REWARD_DIMS)
    #     elif "pick-place" in train_config.env:
    #         obs_size = len(PICK_PLACE_REWARD_DIMS)
    #     elif "push" in train_config.env:
    #         obs_size = len(PUSH_REWARD_DIMS)
    #     else:
    #         raise NotImplementedError
    # else:
    assert not model_config.limit_reward_obs, (
        "Limit reward obs is not implemented right now, please set to False"
    )
    assert not model_config.limit_dem_obs, (
        "Limit dem obs is not implemented right now, please set to False"
    )
    # dem_obs_size = obs_by_goal_1.shape[-1]
    dem_obs_size = dem_by_goal.shape[-1]
    if obs_by_goal_1 is not None:
        obs_size = obs_by_goal_1.shape[-1]
    else:
        obs_size = None

    # if model_config.limit_dem_obs:
    #     if "reach" in train_config.env:
    #         dem_obs_size = len(REACH_REWARD_DIMS)
    #     elif "pick-place" in train_config.env:
    #         dem_obs_size = len(PICK_PLACE_REWARD_DIMS)
    #     elif "push" in train_config.env:
    #         dem_obs_size = len(PUSH_REWARD_DIMS)
    #     else:
    #         raise NotImplementedError
    # else:

    goals = np.load(goal_path, allow_pickle=True)
    # if train_config.dem_obs_share_goals:
    #     obs_goals = np.load(obs_goal_path, allow_pickle=True)
    #     assert np.array_equal(
    #         goals,
    #         obs_goals), "Goals must be the same for obs and dem datasets"
    assert not train_config.dem_obs_share_goals, (
        "Observation sharing not implemented anymore, please set to False"
    )
    if train_config.clip_push_goals_z and "push" in train_config.env:
        goals[:, 2] = np.minimum(goals[:, 2], 0.02)
    horizon = dem_by_goal.shape[-2]

    if (not train_config.skip_reward_training and
            train_config.include_actions) or inference_config.include_actions:
        act_path = os.path.join(dataset_path, "actions.npy")
        try:
            actions_by_goal = np.load(act_path, allow_pickle=True)
        except OSError:
            act_path = os.path.join("scratch", act_path)
            try:
                actions_by_goal = np.load(act_path, allow_pickle=True)
            except OSError:
                print(f"File not found: {act_path}")
    else:
        actions_by_goal = None

    train_dataloader, validation_dataloader, inference_dataloader, shuffle_idxs = (
        get_splits(
            dem_by_goal,
            obs_by_goal_1,
            obs_by_goal_2,
            actions_by_goal,
            goals,
            general_config,
            model_config,
            train_config,
            inference_config,
            shuffle_idxs,
        ))
    return (
        train_dataloader,
        validation_dataloader,
        inference_dataloader,
        obs_size,
        dem_obs_size,
        horizon,
        shuffle_idxs,
    )


def train(
    dataset_path,
    obs_dataset_path,
    saved_model_dir,
    general_config,
    model_config,
    train_config,
    inference_config,
    obs_dataset_path_2=None,
):

    if saved_model_dir is not None:
        assert os.path.exists(saved_model_dir), "Saved model directory does not exist"
        shuffle_idxs = np.load(os.path.join(saved_model_dir, "shuffle_idxs.npy"))
        if (
            not train_config.skip_reward_training
            and not train_config.dem_obs_share_goals
        ):
            raise NotImplementedError(
                "Need to implement observation saving for non-shared goals"
            )
        # if not train_config.skip_reward_training and not train_config.dem_obs_share_goals:
        #     print("WARNING: Loading shuffle_idxs from saved model directory, observation choices are not saved, there will be data leakage")
        #     print("Bad researcher! No cookie!")
    else:
        shuffle_idxs = None

    (
        train_dataloader,
        validation_dataloader,
        inference_dataloader,
        obs_size,
        dem_obs_size,
        horizon,
        shuffle_idxs,
    ) = load_data(
        dataset_path,
        obs_dataset_path,
        general_config,
        model_config,
        train_config,
        inference_config,
        shuffle_idxs,
        obs_dataset_path_2=obs_dataset_path_2,
    )

    net = load_model(
        saved_model_dir, model_config, train_config, obs_size, dem_obs_size, horizon
    )

    if train_config.skip_reward_training:
        return net, inference_dataloader

    optimizer = torch.optim.Adam(net.parameters(), lr=train_config.lr)
    loss_func = nn.MSELoss()

    if model_config.output_type == "action":
        if "pick-place" in train_config.env:
            policy = SawyerPickPlaceV2Policy(
                circle_around=False,
                circle_radius=None,
                horizon=None,
                mirror_goal=False,
                x_bounds=None,
                y_bounds=None,
                z_bounds=None,
                hand_speed=30.0,
                noise_coeff=0.0,
                random_gripping=False,
                num_envs=1,
                goal_pos_adjustment_factor=1.0,
            )
        elif "reach" in train_config.env:
            policy = SawyerReachV2Policy(
                circle_around=False,
                circle_radius=None,
                horizon=None,
                mirror_goal=False,
                x_bounds=None,
                y_bounds=None,
                z_bounds=None,
                hand_speed=30.0,
                noise_coeff=0.0,
                random_gripping=False,
                num_envs=1,
                goal_pos_adjustment_factor=1.0,
            )
        else:
            raise NotImplementedError(
                "Policy not implemented for this environment, please implement"
            )
    else:
        policy = None

    # TQDM inspiration from https://towardsdatascience.com/training-models-with-a-progress-a-bar-2b664de3e13e

    for epoch in range(train_config.num_epochs):
        if train_dataloader.dataset is not None and hasattr(train_dataloader.dataset, 'update_epoch'):
            train_dataloader.dataset.update_epoch(epoch)
        total_loss = 0
        n_samples = 0
        net.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch_data in tepoch:
                obs_batch_for_loss = None
                if model_config.output_type == "goal":
                    dem_batch, goal_batch = batch_data
                else:
                    dem_batch, obs_batch, goal_batch = batch_data
                    obs_batch_for_loss = obs_batch

                optimizer.zero_grad()
                loss = get_batch_loss(
                    net,
                    loss_func,
                    dem_batch,
                    obs_batch_for_loss, # Pass obs_batch or None
                    goal_batch,
                    general_config,
                    model_config,
                    train_config,
                    disp=tepoch.n % 100 == 0,
                    policy=policy,
                )
                loss.backward()
                optimizer.step()

                tepoch.set_description(f"Epoch {epoch}")
                
                actual_batch_size = dem_batch.size(0)
                items_in_batch_loss_calculation = actual_batch_size
                if model_config.output_type != "goal" and obs_batch_for_loss is not None:
                    items_in_batch_loss_calculation *= obs_batch_for_loss.size(1)
                
                total_loss += loss.item() * items_in_batch_loss_calculation
                n_samples += items_in_batch_loss_calculation
                if n_samples > 0:
                    tepoch.set_postfix({"avg loss": total_loss / n_samples})
                else:
                    tepoch.set_postfix({"avg loss": 0})


        avg_training_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        if train_config.save_model:
            # save_dir = os.path.join("runs", wandb.run.name, "models")
            save_dir = os.path.join("scratch", "runs", wandb.run.name, "models")
            os.makedirs(save_dir, exist_ok=True)
            # Construct the filename with wandb run name
            save_filename = os.path.join(save_dir, f"model.parameters")
            # Save the model's state dictionary
            torch.save(net.state_dict(), save_filename)
            print(f"Model saved to {save_filename}")
            # also save shuffle_idxs
            if shuffle_idxs is not None:
                np.save(os.path.join(save_dir, "shuffle_idxs.npy"), shuffle_idxs)

        print(f"Average training loss: {avg_training_loss}")
        wandb.log({"average train loss": avg_training_loss})

        # Validation
        total_loss = 0
        n_samples = 0
        net.eval()
        limit = 320  # Define the limit for the total number of items
        task_counter = 0  # Initialize the counter

        if validation_dataloader is not None:
            if hasattr(validation_dataloader.dataset, 'update_epoch'): # Optional: if val dataset also needs epoch updates
                validation_dataloader.dataset.update_epoch(epoch)
            with torch.no_grad():
                with tqdm(validation_dataloader, unit="batch") as tepoch:
                    for batch_data in tepoch:
                        if task_counter >= limit:
                            break  # Break the loop if the limit is reached

                        obs_batch_for_loss = None
                        if model_config.output_type == "goal":
                            dem_batch, goal_batch = batch_data
                        else:
                            dem_batch, obs_batch, goal_batch = batch_data
                            obs_batch_for_loss = obs_batch

                        loss = get_batch_loss(
                            net,
                            loss_func,
                            dem_batch,
                            obs_batch_for_loss, # Pass obs_batch or None
                            goal_batch,
                            general_config,
                            model_config,
                            train_config, # Using train_config for params like extra_success_reward
                            disp=tepoch.n % 100 == 0,
                            policy=policy,
                        )

                        actual_batch_size = dem_batch.size(0)
                        items_in_batch_loss_calculation = actual_batch_size
                        if model_config.output_type != "goal" and obs_batch_for_loss is not None:
                             items_in_batch_loss_calculation *= obs_batch_for_loss.size(1)
                        
                        total_loss += loss.item() * items_in_batch_loss_calculation
                        n_samples += items_in_batch_loss_calculation
                        task_counter += actual_batch_size  # task_counter counts tasks/goals
                        if n_samples > 0:
                            tepoch.set_postfix({"avg validation loss": total_loss / n_samples})
                        else:
                            tepoch.set_postfix({"avg validation loss": 0})


        avg_val_loss = total_loss / n_samples if n_samples > 0 else 0.0
        wandb.log({"average validation loss": avg_val_loss})

    # return net and inference dataloader for inference
    # (not used in this script)
    return net, inference_dataloader


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly")
    args = parse_args()
    wandb.init(project=args.wandb_project)
    wandb.config.update(args)
    train(args)
