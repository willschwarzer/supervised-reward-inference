import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushV2Policy(Policy):

    def __init__(self,
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
                 num_envs=1):
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

        assert circle_around == False, "circle_around not implemented for push-v2"
        assert mirror_goal == False, "mirror_goal not implemented for push-v2"
        assert random_gripping == False, "random_gripping not implemented for push-v2"
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            # "hand_pos": obs[:3],
            # "unused_1": obs[3],
            # "puck_pos": obs[4:7],
            # "unused_2": obs[7:-3],
            # "goal_pos": obs[-3:],
            "hand_pos": obs[:3],
            "gripper_distance_apart": obs[3],
            "puck_pos": obs[4:7],
            "puck_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
        }

    def get_action(self, obs, step, env_idx):
        # step not used
        # o_d = self._parse_obs(obs)

        # action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        # action["delta_pos"] = move(
        #     o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=20.0
        # )
        # action["grab_effort"] = self._grab_effort(o_d)

        # return action.array

        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(o_d["hand_pos"],
                                   to_xyz=self._desired_pos(o_d),
                                   p=self.hand_speed)
        # clip action to be within 0.5
        action["delta_pos"] = np.clip(action["delta_pos"], -0.3, 0.3)
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        # pos_curr = o_d["hand_pos"]
        # pos_puck = o_d["puck_pos"] + np.array([-0.005, 0, 0])
        # pos_goal = o_d["goal_pos"]

        # # If error in the XY plane is greater than 0.02, place end effector above the puck
        # if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
        #     return pos_puck + np.array([0.0, 0.0, 0.2])
        # # Once XY error is low enough, drop end effector down on top of puck
        # elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
        #     return pos_puck + np.array([0.0, 0.0, 0.03])
        # # Move to the goal
        # else:
        #     return pos_goal

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

    @staticmethod
    def _grab_effort(o_d):
        # pos_curr = o_d["hand_pos"]
        # pos_puck = o_d["puck_pos"]

        # if (
        #     np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02
        #     or abs(pos_curr[2] - pos_puck[2]) > 0.10
        # ):
        #     return 0.0
        # # While end effector is moving down toward the puck, begin closing the grabber
        # else:
        #     return 0.6

        pos_curr = o_d["hand_pos"]
        pos_puck = o_d["puck_pos"]
        if np.linalg.norm(pos_curr - pos_puck) < 0.07:
            return 1.0
        else:
            return 0.0
