import numpy as np
import torch

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPickPlaceV2Policy(Policy):

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
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.hand_speed = hand_speed
        self.num_envs = num_envs
        self.random_gripping = random_gripping
        if self.random_gripping:
            self.random_grip_targets = np.random.uniform(0.0,
                                                        1.0,
                                                        size=(num_envs, ))
            self.random_grip_efforts = np.random.uniform(0.5,
                                                        1.0,
                                                        size=(num_envs, ))

        assert circle_around == False, "circle_around not implemented for pick-place-v2"
        assert mirror_goal == False, "mirror_goal not implemented for pick-place-v2"

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "puck_pos": obs[4:7],
            "puck_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
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
                "puck_rot": obs[..., 7:11],
                "goal_pos": obs[..., -3:],
                "unused_info_curr_obs": obs[..., 11:18],
                "_prev_obs": obs[..., 18:36],
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
                "puck_rot": obs[..., 28:32],
                "goal_pos": obs[..., -3:],
                # "unused_info_curr_obs": obs[..., 11:18],
                "_prev_obs": obs[..., 39:57],
            }

    def get_action(self, obs, current_step, env_idx):
        if current_step == 0 and self.random_gripping:
            self.random_grip_targets = np.random.uniform(
                0.0, 1.0, size=(self.num_envs, ))
            self.random_grip_efforts = np.random.uniform(
                0.5, 1.0, size=(self.num_envs, ))
        # current_step not used here, only in reach
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(o_d["hand_pos"],
                                   to_xyz=self._desired_pos(o_d),
                                   p=self.hand_speed)
        # clip action to be within 0.3
        action["delta_pos"] = np.clip(action["delta_pos"], -0.3, 0.3)
        action["grab_effort"] = self._grab_effort(o_d, env_idx)

        return action.array

    def get_action_batch(self, obs):
        assert not self.random_gripping, "random gripping not implemented for batch"
        # current_step not used here, only in reach
        o_d = self._parse_obs_batch(obs)

        delta_pos = move(o_d["hand_pos"],
                         to_xyz=self._desired_pos_batch(o_d),
                         p=self.hand_speed)
        # breakpoint()
        # clip action to be within 0.5
        delta_pos = torch.clip(delta_pos, -0.3, 0.3)
        grab_effort = self._grab_effort_batch(o_d)
        # print(torch.sum(grab_effort))
        # breakpoint()

        # Construct the action array
        # Assuming delta_pos is (batch_size, 3) and grab_effort is (batch_size,)
        # We want to create an action array of shape (batch_size, 4)
        # where the first 3 elements are delta_pos and the last is grab_effort
        action_batch = torch.concatenate((delta_pos, grab_effort[..., None]), axis=-1)

        return action_batch

    # @staticmethod
    def _desired_pos(self, o_d):
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"] + np.array([-0.005, 0, 0])
        pos_goal = o_d["goal_pos"]
        gripper_separation = o_d["gripper_distance_apart"]
        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.04:
            # return pos_puck + np.array([0.0, 0.0, 0.03])
            return pos_puck + np.array([0.0, 0.0, 0.01])
        # Wait for gripper to close before continuing to move
        elif gripper_separation > 0.73:
            return pos_curr
        # Move to goal
        else:
            return pos_goal

    def _desired_pos_batch(self, o_d):
        pos_curr = o_d["hand_pos"]  # (batch_size, 3)
        device = pos_curr.device
        dtype = pos_curr.dtype
        pos_puck = o_d["puck_pos"] + torch.tensor([-0.005, 0, 0], device=device, dtype=dtype)  # (batch_size, 3)
        pos_goal = o_d["goal_pos"]  # (batch_size, 3)
        gripper_separation = o_d["gripper_distance_apart"]  # (batch_size,)

        # Initialize desired_pos with a default value (e.g., move to goal)
        desired_pos = pos_goal.clone()  # (batch_size, 3)

        # Condition 1: If error in the XY plane is greater than 0.02, place end effector above the puck
        xy_error_cond = torch.linalg.norm(pos_curr[..., :2] - pos_puck[..., :2], dim=1) > 0.02
        desired_pos[xy_error_cond] = pos_puck[xy_error_cond] + torch.tensor([0.0, 0.0, 0.1], device=device, dtype=dtype)

        # Condition 2: Once XY error is low enough, drop end effector down on top of puck
        # This condition applies only if condition 1 is false
        z_error_cond = torch.logical_and(
            ~xy_error_cond,
            torch.abs(pos_curr[..., 2] - pos_puck[..., 2]) > 0.05
        )
        puck_height_cond = pos_puck[..., -1] < 0.04
        cond2_combined = torch.logical_and(z_error_cond, puck_height_cond)
        desired_pos[cond2_combined] = pos_puck[cond2_combined] + torch.tensor([0.0, 0.0, 0.01], device=device, dtype=dtype)
        
        # Condition 3: Wait for gripper to close before continuing to move
        # This condition applies if neither cond1 nor cond2 (for dropping) is true
        # and gripper is not closed enough
        gripper_not_closed_cond = gripper_separation > 0.73
        # Apply this condition where previous conditions were false
        cond3_combined = torch.logical_and(
            torch.logical_and(~xy_error_cond, ~cond2_combined),
            gripper_not_closed_cond
        )
        desired_pos[cond3_combined] = pos_curr[cond3_combined]

        # Else (implicitly): Move to goal (already set as default)

        return desired_pos

    # @staticmethod
    def _grab_effort(self, o_d, env_idx):
        if self.random_gripping:
            if o_d["gripper_distance_apart"] < self.random_grip_targets[
                    env_idx]:
                return self.random_grip_efforts[env_idx]
            else:
                return 1.0 - self.random_grip_efforts[env_idx]
        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"]
        if np.linalg.norm(pos_curr - pos_puck) < 0.07:
            return 1.0
        else:
            return 0.0

    def _grab_effort_batch(self, o_d):
        assert not self.random_gripping, "random gripping not implemented for batch"
        pos_curr = o_d["hand_pos"] # (batch_size, 3)
        pos_puck = o_d["puck_pos"] # (batch_size, 3)
        # (batch_size, )
        # if np.linalg.norm(pos_curr - pos_puck) < 0.07:
        #     return 1.0
        # else:
        #     return 0.0
        return torch.where(torch.linalg.norm(pos_curr - pos_puck, axis=-1) < 0.07,
                        1.0, 0.0).to(pos_curr.device)

