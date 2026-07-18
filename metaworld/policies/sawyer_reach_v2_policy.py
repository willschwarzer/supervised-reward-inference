import numpy as np
import torch

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerReachV2Policy(Policy):

    def __init__(
        self,
        circle_around=False,
        circle_radius=None,
        horizon=None,
        mirror_goal=False,
        x_bounds=None,
        y_bounds=None,
        z_bounds=None,
        hand_speed=10.0,
        noise_coeff=0.0,
        random_gripping=False,
        num_envs=1,
        goal_pos_adjustment_factor=1.0,
    ):
        self.circle_around = circle_around
        self.circle_radius = circle_radius
        self.horizon = horizon
        if horizon is not None:
            self.angle_increment = 2 * np.pi / horizon
        self.mirror_goal = mirror_goal
        if self.mirror_goal and goal_pos_adjustment_factor != 1.0:
            raise ValueError(
                "Cannot have mirror_goal and goal_pos_adjustment_factor at the same time"
            )
        self.goal_pos_adjustment_factor = goal_pos_adjustment_factor
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.hand_speed = hand_speed
        self.noise_coeff = noise_coeff
        self.num_envs = num_envs
        self.random_gripping = random_gripping
        if self.random_gripping:
            self.random_grip_targets = np.random.uniform(0.0, 1.0, size=(num_envs,))
            self.random_grip_efforts = np.random.uniform(0.5, 1.0, size=(num_envs,))

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "puck_pos": obs[4:7],
            "unused_2": obs[7:-3],
            "goal_pos": obs[-3:],
        }

    @staticmethod
    def _parse_obs_batch(obs):
        expanded_rep = obs.shape[-1] > 39
        assert not expanded_rep or obs.shape[-1] == 60, "just a sanity check"
        if not expanded_rep:
            return {
                "hand_pos": obs[..., :3],
                "gripper_distance_apart": obs[..., 3],
                "puck_pos": obs[..., 4:7],
                "unused_2": obs[..., 7:-3],
                "goal_pos": obs[..., -3:],
            }
        
        else:
            # Note: this is the case where
            # the first 21 dimensions are additional information used for computing
            # the reward. Therefore, we need to add 21 to the indices
            # of the observations we are interested in.
            return {
                "hand_pos": obs[..., 21:24],
                "gripper_distance_apart": obs[..., 24],
                "puck_pos": obs[..., 25:28],
                "unused_2": obs[..., 28:-3],
                "goal_pos": obs[..., -3:],
            }

    def mirror_goal_position(self, goal_pos):
        x_midpoint = (self.x_bounds[0] + self.x_bounds[1]) / 2
        y_midpoint = (self.y_bounds[0] + self.y_bounds[1]) / 2
        z_midpoint = (self.z_bounds[0] + self.z_bounds[1]) / 2

        mirrored_goal = np.array(
            [
                x_midpoint + (x_midpoint - goal_pos[0]),
                y_midpoint + (y_midpoint - goal_pos[1]),
                z_midpoint + (z_midpoint - goal_pos[2]),
            ]
        )

        return mirrored_goal
    
    def adjust_goal_position(self, goal_pos):
        x_midpoint = (self.x_bounds[0] + self.x_bounds[1]) / 2
        y_midpoint = (self.y_bounds[0] + self.y_bounds[1]) / 2

        adjusted_goal = np.array(
            [
                x_midpoint + self.goal_pos_adjustment_factor * (goal_pos[0] - x_midpoint),
                y_midpoint + self.goal_pos_adjustment_factor * (goal_pos[1] - y_midpoint),
                goal_pos[2],
            ]
        )
        return adjusted_goal

    def get_action(self, obs, current_step, env_idx):
        if current_step == 0 and self.random_gripping:
            self.random_grip_targets = np.random.uniform(
                0.0, 1.0, size=(self.num_envs,)
            )
            self.random_grip_efforts = np.random.uniform(
                0.5, 1.0, size=(self.num_envs,)
            )
        o_d = self._parse_obs(obs)

        if self.mirror_goal:
            o_d["goal_pos"] = self.mirror_goal_position(o_d["goal_pos"])
        elif self.goal_pos_adjustment_factor != 1.0:
            o_d["goal_pos"] = self.adjust_goal_position(o_d["goal_pos"])

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        if self.circle_around and current_step < self.horizon:
            circle_point = self.calculate_circle_point(o_d["goal_pos"], current_step)
            action["delta_pos"] = move(
                o_d["hand_pos"], to_xyz=circle_point, p=self.hand_speed
            )
        else:
            action["delta_pos"] = move(
                o_d["hand_pos"], to_xyz=o_d["goal_pos"], p=self.hand_speed
            )

        if self.noise_coeff > 0:
            noisy_action = np.random.uniform(0, 1)
            if noisy_action < self.noise_coeff:
                action["delta_pos"] = np.random.uniform(-100, 100, size=3)

        if self.random_gripping:
            if o_d["gripper_distance_apart"] < self.random_grip_targets[env_idx]:
                action["grab_effort"] = self.random_grip_efforts[env_idx]
            else:
                action["grab_effort"] = 1.0 - self.random_grip_efforts[env_idx]
            # print(
            #     f"Env idx: {env_idx}, Step: {current_step}, random_grip_target: {self.random_grip_targets[env_idx]}, random_grip_effort: {self.random_grip_efforts[env_idx]}, gripper_distance_apart: {o_d['gripper_distance_apart']}, grab_effort: {action['grab_effort']}"
            # )
        else:
            action["grab_effort"] = 0.0

        return action.array

    def get_action_batch(self, obs):
        assert not (
            self.random_gripping or
            self.circle_around or
            self.noise_coeff > 0 or
            self.mirror_goal or
            self.goal_pos_adjustment_factor != 1.0
        ), "Batch action not implemented for random gripping, circle around, noise coeff, mirror goal or goal pos adjustment factor"
        o_d = self._parse_obs_batch(obs)
        # action = Action({"delta_pos": torch.arange(3), "grab_effort": 3})
        action = torch.zeros(
            (o_d["hand_pos"].shape[0], 4), dtype=torch.float32
        ).to(obs.device)
        action[..., 0:3] = move(
            o_d["hand_pos"], to_xyz=o_d["goal_pos"], p=self.hand_speed
        )
        action[..., -1] = 0.0
        return action

    def calculate_circle_point(self, goal_pos, current_step):
        angle = current_step * self.angle_increment
        x = goal_pos[0] + self.circle_radius * np.cos(angle)
        y = goal_pos[1] + self.circle_radius * np.sin(angle)
        # print(f"Step: {current_step},\n\
        #       Angle: {angle},\n\
        #       Angle Increment: {self.angle_increment},\n\
        #     Goal: {goal_pos},\n\
        #     Circle Point: {np.array([x, y, goal_pos[2]])}\n")
        return np.array([x, y, goal_pos[2]])
