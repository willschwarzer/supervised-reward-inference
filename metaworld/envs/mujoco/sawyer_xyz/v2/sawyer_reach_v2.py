import mujoco
import numpy as np
import torch
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerReachEnvV2(SawyerXYZEnv):
    """SawyerReachEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    def __init__(
        self,
        tasks=None,
        render_mode=None,
        goal_bounds=None,
        state_encoders=None,
        reward_mlps=None,
        task_reps=None,
        goals=None,
        extra_success_reward=0.0,
        limit_reward_obs=False,
        unscale_reward=False,
        use_gt_reward=False,
        scale_gt_reward=False,
        use_gt_goal=False,
        no_task_rep=False,
        proximity_reward=0.0,
        success_requires_touch=False,
        third_gt=False,
        half_gt=False,
        ensembling=None,
        include_extra_reward_info=False,
        include_partial_reward_info=False,
        mask_obj=False,
        hand_starts=None,
        grasp_rew_only=False,
    ):
        assert not (third_gt or half_gt), "Third and half ground truth not implemented for reach"
        if goal_bounds is None:
            goal_low = (-0.1, 0.8, 0.05)
            goal_high = (0.1, 0.9, 0.3)
        else:
            goal_low, goal_high = goal_bounds
        hand_low = (-0.5, 0.2, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            state_encoders=state_encoders,
            reward_mlps=reward_mlps,
            task_reps=task_reps,
            goals=goals,
            extra_success_reward=extra_success_reward,
            unscale_reward=unscale_reward,
            use_gt_reward=use_gt_reward,
            scale_gt_reward=scale_gt_reward,
            use_gt_goal=use_gt_goal,
            no_task_rep=no_task_rep,
            ensembling=ensembling,
            include_extra_reward_info=include_extra_reward_info,
            include_partial_reward_info=include_partial_reward_info,
            mask_obj=mask_obj,
            hand_starts=hand_starts,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": self.
            hand_init_pos,  # Added by SRI authors; this var is in superclass
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.extra_success_reward = extra_success_reward

        if limit_reward_obs:
            self.reward_dims = REWARD_DIMS
        else:
            self.reward_dims = None

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_reach_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        # collecting info we need to calculate rewards later
        tcp_center = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        init_left_pad = self.init_left_pad
        init_right_pad = self.init_right_pad
        obj_init_pos = self.obj_init_pos
        hand_init_pos = self.hand_init_pos
        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {
            "success": success,
            "near_object": reach_dist,
            "grasp_success": 1.0,
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
            "unscaled_reward": reward,
            "tcp_center": tcp_center,
            "left_pad": left_pad,
            "right_pad": right_pad,
            "init_left_pad": init_left_pad,
            "init_right_pad": init_right_pad,
            "obj_init_pos": obj_init_pos,
            "hand_init_pos": hand_init_pos,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        # self._target_pos = self.goal.copy() # SRI authors have no idea why this was here
        self.obj_init_pos = self.fix_extreme_obj_pos(
            self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_and_obj = self._get_state_rand_vec()
        if self.goals is None:
            self._target_pos = goal_and_obj[3:]
        # while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
        #     goal_pos = self._get_state_rand_vec()
        #     self._target_pos = goal_pos[3:]
        self.obj_init_pos = goal_and_obj[:3]
        # change the MuJoCo simulation to reflect the new goal
        self.update_goal_site(self._target_pos)
        self._set_obj_xyz(self.obj_init_pos)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center  # need tcp_center
        # obj = obs[4:7]
        # tcp_opened = obs[3]
        target = self._target_pos  # need _target_pos if we don't have that already

        tcp_to_target = np.linalg.norm(tcp - target)
        # obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = np.linalg.norm(self.hand_init_pos -
                                         target)  # need hand_init_pos
        in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        success = float(tcp_to_target <= _TARGET_RADIUS)
        reward = 10 * in_place + success * self.extra_success_reward

        return [reward, tcp_to_target, in_place]


class TrainReachv2(SawyerReachEnvV2):
    tasks = None

    def __init__(self):
        SawyerReachEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestReachv2(SawyerReachEnvV2):
    tasks = None

    def __init__(self):
        SawyerReachEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


# REWARD FUNCTION #
# Used in SRI training code to compute reward without needing to run the environment
# Signature takes more arguments than they need just for consistency with pick-place

REWARD_DIMS = np.r_[12:15, 18:21]


def compute_reward(obs, target_pos, action=None, extra_success_reward=0.0):
    left_pad = obs[0:3]
    right_pad = obs[3:6]
    init_left_pad = obs[6:9]
    init_right_pad = obs[9:12]
    tcp_center = obs[12:15]
    obj_init_pos = obs[15:18]
    hand_init_pos = obs[18:21]
    obs_cut = obs[21:]
    _TARGET_RADIUS = 0.05
    tcp = tcp_center  # Use tcp_center from arguments
    target = target_pos  # Use target_pos from arguments

    tcp_to_target = np.linalg.norm(tcp - target)

    in_place_margin = np.linalg.norm(hand_init_pos - target)  # Use hand_init_pos from arguments
    in_place = reward_utils.tolerance(
        tcp_to_target,
        bounds=(0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    success = float(tcp_to_target <= _TARGET_RADIUS)
    reward = 10 * in_place + success * extra_success_reward

    # The return value is a list containing the scaled in_place reward, the distance to the target, and the in_place value itself.
    # This aligns with the original function's output, providing compatibility with the expected environment interface.
    return reward


def compute_reward_batch(
    obs, target_pos, action=None, extra_success_reward=0.0, limit_reward_obs=False
):
    if limit_reward_obs:
        assert obs.shape[-1] == 6, "Obs shape must be (bsize, n, 6) for limited observation"
        tcp_center = obs[:, :, 0:3]
        hand_init_pos = obs[:, :, 3:6]
    else:
        left_pad = obs[:, :, 0:3]
        right_pad = obs[:, :, 3:6]
        init_left_pad = obs[:, :, 6:9]
        init_right_pad = obs[:, :, 9:12]
        tcp_center = obs[:, :, 12:15]
        obj_init_pos = obs[:, :, 15:18]
        hand_init_pos = obs[:, :, 18:21]
    _TARGET_RADIUS = 0.05
    batch_size, n = tcp_center.shape[:2]

    # need to expand the target_pos to match the shape of the tcp_center
    # it starts out only (bsize, 3), but we need it to be (bsize, n, 3)
    target = target_pos.unsqueeze(1).expand(-1, n, -1)
    tcp_to_target = torch.norm(tcp_center - target, dim=-1)

    in_place_margin = torch.norm(hand_init_pos - target, dim=-1)  # Use hand_init_pos from arguments
    in_place = reward_utils.tolerance_batch(
        tcp_to_target,
        bounds=(0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    success = (tcp_to_target <= _TARGET_RADIUS).float()
    reward = 10 * in_place + success * extra_success_reward

    # The return value is a list containing the scaled in_place reward, the distance to the target, and the in_place value itself.
    # This aligns with the original function's output, providing compatibility with the expected environment interface.
    return reward
