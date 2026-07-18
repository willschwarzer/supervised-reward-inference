import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)
import mujoco
import os
import torch


class SawyerPickPlaceEnvV2(SawyerXYZEnv):
    """SawyerPickPlaceEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
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
        if goal_bounds is None:
            goal_low = (-0.1, 0.8, 0.05)
            goal_high = (0.1, 0.9, 0.3)
        else:
            goal_low, goal_high = goal_bounds
        hand_low = (-0.5, 0.2, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        # SRI change
        # obj_low = (-0.05, 0.6, 0.02)
        # obj_high = (0.05, 0.65, 0.02)

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
            success_requires_touch=success_requires_touch,
            third_gt=third_gt,
            half_gt=half_gt,
            ensembling=ensembling,
            include_extra_reward_info=include_extra_reward_info,
            include_partial_reward_info=include_partial_reward_info,
            mask_obj=mask_obj,
            hand_starts=hand_starts,
            grasp_rew_only=grasp_rew_only,
        )

        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": self.hand_init_pos,  # Added by SRI authors; this var is in superclass
        }

        self.goal = np.array([0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0
        self.obj_init_pos = None
        self.recorded_grasp_pos = False
        self.extra_success_reward = extra_success_reward
        self.grasp_rew_only = grasp_rew_only
        # assert not (self.reinit_goals and self.gripped_start), "Can't have both random_init and gripped_start"
        # if self.reinit_goals and self.gripped_start:
        #     print("Both reinit_goals and gripped_start are true, will only reinitialize goals, not the object")

        if limit_reward_obs:
            self.reward_dims = REWARD_DIMS
        else:
            self.reward_dims = None

        self.proximity_reward = proximity_reward
        self.success_requires_touch = success_requires_touch
        assert (
            not self.success_requires_touch
        ), "success_requires_touch not implemented for PickPlace"

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_pick_place_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        # collecting the info we need for computing reward later now
        # while it's still correct
        tcp_center = self.tcp_center
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")
        init_left_pad = self.init_left_pad
        init_right_pad = self.init_right_pad
        obj_init_pos = self.obj_init_pos
        hand_init_pos = self.hand_init_pos
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(
            self.touching_main_object
            and (tcp_open > 0)
            and (obj[2] - 0.02 > self.obj_init_pos[2])
        )
        if self.record_grips and grasp_success and not self.recorded_grasp_pos:
            # saving the episode's first grasp position
            self.recorded_grasp_pos = True
            # save all the positions, orientations and velocities
            # so we can restore the state for other agents to learn from
            qpos = self.data.qpos.ravel().copy()
            qvel = self.data.qvel.ravel().copy()
            # now save them to the first i such that grasp_pos/{i}.npy doesn't exist
            # make dirs if they don't exist
            os.makedirs("grasp_pos", exist_ok=True)
            os.makedirs("grasp_vel", exist_ok=True)
            for i in range(10000):
                if not os.path.exists(f"grasp_pos/pick_place_{i}.npy"):
                    print(
                        f"Saving grasp_pos/pick_place_{i}.npy and grasp_vel/pick_place_{i}.npy"
                    )
                    np.save(f"grasp_pos/pick_place_{i}.npy", qpos)
                    np.save(f"grasp_vel/pick_place_{i}.npy", qvel)
                    break
            else:
                raise Exception("Too many grasps, you don't need that many")

        info = {
            "success": success,
            "near_object": near_object,
            "tcp_to_obj": tcp_to_obj,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
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

    @property
    def _get_id_main_object(self):
        return self.data.geom("objGeom").id

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.geom("objGeom").xmat.reshape(3, 3)
        ).as_quat()

    def update_goal_site(self, new_goal_position):
        """
        Update the position of the 'goal' site in the MuJoCo environment.

        Args:
        new_goal_position (np.ndarray): A 3D numpy array specifying the new position of the goal.
        """
        # Find the ID of the 'goal' site in the MuJoCo model
        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")

        # Update the position of the 'goal' site
        self.model.site_pos[goal_site_id] = new_goal_position

        # Optionally, if immediate visual update in the simulation is required
        mujoco.mj_forward(self.model, self.data)

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
        # SRI note: if using manually set goals/tasks, this should be called
        # *after* the goal is set
        # So therefore the target_pos here should be correct,
        # and under no circumstances should be overwritten
        # Will add assert statements to ensure this
        self._reset_hand()
        self.recorded_grasp_pos = False

        # goal_pos = self._get_state_rand_vec()
        # self._target_pos = goal_pos[3:]
        goal_and_obj = self._get_state_rand_vec()
        if self.goals is None:
            assert self.task_reps is None, "Task reps should be None if goals is None"
            self._target_pos = goal_and_obj[-3:]

        if self.gripped_start:
            # Load the saved gripped state
            # grasp_pos_qpos = np.load("grasp_pos/pick_place_0.npy")
            # grasp_pos_qvel = np.load("grasp_vel/pick_place_0.npy")
            # Load a random gripped state
            idx = np.random.randint(10000)
            # make sure the file exists, otherwise sample again
            while not os.path.exists(f"grasp_pos/pick_place_{idx}.npy"):
                idx = np.random.randint(10000)
            grasp_pos_qpos = np.load(f"grasp_pos/pick_place_{idx}.npy")
            grasp_pos_qvel = np.load(f"grasp_vel/pick_place_{idx}.npy")
            # XXX zeroing out qvel for now
            grasp_pos_qvel = np.zeros_like(grasp_pos_qvel)

            # Set the simulation's state to the saved gripped position
            self.set_state(grasp_pos_qpos, grasp_pos_qvel)

            # Adjust the object's initial position based on the gripped state
            self.obj_init_pos = grasp_pos_qpos[
                9:12
            ]  # Assuming these indices correspond to the object position
        else:
            # Set object to its initial position from goal_pos
            self.obj_init_pos = goal_and_obj[:3]
            self._set_obj_xyz(self.obj_init_pos)

        self.obj_init_angle = self.init_config["obj_init_angle"]
        if not self._freeze_rand_vec:
            # Ensure the goal is sufficiently far from the initial object position
            max_its = 1000
            its = 0
            # if self.goals is not None:
            #     # in this case we're reinitializing the object, not the goal
            #     goal_pos = self.goals[0]
            #     self._target_pos = goal_pos
            while np.linalg.norm(goal_and_obj[:2] - self._target_pos[:2]) < 0.15:
                goal_and_obj = self._get_state_rand_vec()
                if self.goals is None:
                    self._target_pos = goal_and_obj[-3:]
                its += 1
                if its >= max_its:
                    raise ValueError(
                        "Could not find a goal position far enough from the object"
                    )
            if self.goals is None:
                assert (
                    self.task_reps is None
                ), "Task reps should be None if goals is None"
                self._target_pos = goal_and_obj[-3:]
            else:
                self.obj_init_pos = goal_and_obj[:3]
                self._set_obj_xyz(self.obj_init_pos)
        else:
            goal_and_obj = self._get_state_rand_vec()
            # if not np.allclose(goal_and_obj[:3], np.array([0, 0.6, 0.02])):
            #     print("Warning: object position is not at the default position. This should only happen on the first reset")

        # Update the MuJoCo simulation to reflect the new goal
        if self.goals is None:
            self.update_goal_site(self._target_pos)
        else:
            pass  # do nothing, the goal was set in the super class, they should handle it

        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com("leftpad")  # need left_pad
        right_pad = self.get_body_com("rightpad")  # need right_pad
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(
            abs(obj_position[1] - self.init_right_pad[1])
            - pad_success_margin  # need init_right_pad
        )
        left_caging_margin = abs(
            abs(obj_position[1] - self.init_left_pad[1])
            - pad_success_margin  # need init_left_pad
        )

        right_caging = reward_utils.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = reward_utils.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        y_caging = reward_utils.hamacher_product(left_caging, right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0.0, -obj_position[1], 0.0]
        )
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

        # used for computing the tcp to object object margin in the x_z plane
        init_obj_x_z = self.obj_init_pos + np.array(
            [0.0, -self.obj_init_pos[1], 0.0]
        )  # need obj_init_pos
        init_tcp_x_z = self.init_tcp + np.array([0.0, -self.init_tcp[1], 0.0])
        tcp_obj_x_z_margin = (
            np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        )
        # hot fix: margin is sometimes negative when using gripped_start
        tcp_obj_x_z_margin = max(0.0, tcp_obj_x_z_margin)

        x_z_caging = reward_utils.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        caging_and_gripping = (caging_and_gripping + caging) / 2
        return caging_and_gripping

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center  ### need tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = (
            self._target_pos
        )  ### need target_pos if we don't have that already (we do have it already)

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(action, obj)
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, in_place
        )
        if not self.grasp_rew_only:
            reward = in_place_and_object_grasped
        else:
            reward = object_grasped
        if self.proximity_reward > 0.0:
            # init_dist_to_obj = torch.norm(hand_init_pos - obj_init_pos, dim=2)
            init_dist_to_obj = np.linalg.norm(self.hand_init_pos - self.obj_init_pos)
            # Additional reward conditions
            reward *= max(
                self.proximity_reward, 1
            )  # need to double this as well to make sure proximity reward
            # doesn't overshadow it
            close_to_obj = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, 0.02),
                margin=init_dist_to_obj,
                sigmoid="long_tail",
            )
            reward += self.proximity_reward * close_to_obj

        if (
            tcp_to_obj < 0.02
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        ):
            reward += 1.0 + 5.0 * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.0 + self.extra_success_reward
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]


class TrainPickPlacev2(SawyerPickPlaceEnvV2):
    tasks = None

    def __init__(self):
        SawyerPickPlaceEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestPickPlacev2(SawyerPickPlaceEnvV2):
    tasks = None

    def __init__(self):
        SawyerPickPlaceEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


# REWARD FUNCTIONS #
# Used in SRI training code to compute reward without needing to run the environment


# def compute_reward(tcp_center, left_pad, right_pad, init_left_pad, init_right_pad, obj_init_pos, target_pos, action, obs):
def compute_reward(obs, target_pos, action=None, extra_success_reward=0.0):
    # we actually need to get all of the values from obs
    # this is how it was generated:
    # augmented[i] = np.concatenate((left_pad, right_pad, init_left_pad, init_right_pad, tcp_center, obj_init_pos, obs[i]))
    # so:
    left_pad = obs[0:3]
    right_pad = obs[3:6]
    init_left_pad = obs[6:9]
    init_right_pad = obs[9:12]
    tcp_center = obs[12:15]
    obj_init_pos = obs[15:18]
    obs = obs[18:]
    _TARGET_RADIUS = 0.05
    tcp = tcp_center  # Use tcp_center from arguments
    obj = obs[4:7]
    tcp_opened = obs[3]
    target = target_pos  # Use target_pos from arguments

    obj_to_target = np.linalg.norm(obj - target)
    tcp_to_obj = np.linalg.norm(obj - tcp)
    in_place_margin = np.linalg.norm(
        obj_init_pos - target
    )  # Use obj_init_pos from arguments

    in_place = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, _TARGET_RADIUS),
        margin=in_place_margin,
        sigmoid="long_tail",
    )

    object_grasped = _gripper_caging_reward(
        tcp_center,
        left_pad,
        right_pad,
        init_left_pad,
        init_right_pad,
        obj_init_pos,
        obj,
        action,
    )
    in_place_and_object_grasped = reward_utils.hamacher_product(
        object_grasped, in_place
    )
    reward = in_place_and_object_grasped

    if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > obj_init_pos[2]):
        reward += 1.0 + 5.0 * in_place
    if obj_to_target < _TARGET_RADIUS:
        reward = 10.0 + extra_success_reward
    # return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
    return reward


def _gripper_caging_reward(
    tcp_center,
    left_pad,
    right_pad,
    init_left_pad,
    init_right_pad,
    obj_init_pos,
    obj_position,
    action=None,
):
    pad_success_margin = 0.05
    x_z_success_margin = 0.005
    obj_radius = 0.015
    tcp = tcp_center  # Use tcp_center from arguments
    delta_object_y_left_pad = left_pad[1] - obj_position[1]
    delta_object_y_right_pad = obj_position[1] - right_pad[1]
    right_caging_margin = abs(
        abs(obj_position[1] - init_right_pad[1]) - pad_success_margin
    )  # Use init_right_pad from arguments
    left_caging_margin = abs(
        abs(obj_position[1] - init_left_pad[1]) - pad_success_margin
    )  # Use init_left_pad from arguments

    right_caging = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )

    y_caging = reward_utils.hamacher_product(left_caging, right_caging)

    # compute the tcp_obj distance in the x_z plane
    tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
    obj_position_x_z = np.copy(obj_position) + np.array([0.0, -obj_position[1], 0.0])
    tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

    # used for computing the tcp to object object margin in the x_z plane
    init_obj_x_z = obj_init_pos + np.array(
        [0.0, -obj_init_pos[1], 0.0]
    )  # Use obj_init_pos from arguments
    init_tcp_x_z = tcp_center + np.array(
        [0.0, -tcp_center[1], 0.0]
    )  # Use tcp_center from arguments
    tcp_obj_x_z_margin = (
        np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
    )
    tcp_obj_x_z_margin = max(0.0, tcp_obj_x_z_margin)

    x_z_caging = reward_utils.tolerance(
        tcp_obj_norm_x_z,
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_x_z_margin,
        sigmoid="long_tail",
    )

    if action is not None:
        gripper_closed = min(max(0, action[-1]), 1)
    else:
        gripper_closed = 1
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)

    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
    caging_and_gripping = (caging_and_gripping + caging) / 2
    return caging_and_gripping


REWARD_DIMS = np.r_[0:3, 3:6, 6:9, 9:12, 12:15, 15:18, 24, 25:28]


def compute_reward_batch(
    obs,
    target_pos,
    extra_success_reward=0.0,
    limit_reward_obs=False,
    action=None,
    proximity_reward=0.0,
    grasp_rew_only=False,
):
    assert (
        not limit_reward_obs
    ), "limit_reward_obs not implemented for compute_reward_batch"
    # Assuming obs shape is (batch_size, n, obs_size)
    # and target_pos shape is (batch_size, 3)
    batch_size, n, _ = obs.shape
    _TARGET_RADIUS = 0.05

    # Split observation into its components
    left_pad = obs[:, :, 0:3]
    right_pad = obs[:, :, 3:6]
    init_left_pad = obs[:, :, 6:9]
    init_right_pad = obs[:, :, 9:12]
    tcp_center = obs[:, :, 12:15]
    obj_init_pos = obs[:, :, 15:18]
    hand_init_pos = obs[:, :, 18:21]
    obs_cut = obs[:, :, 21:]
    obj = obs_cut[:, :, 4:7]
    tcp_opened = obs_cut[:, :, 3]

    # Expand target_pos for broadcasting
    target = target_pos.unsqueeze(1).expand(-1, n, -1)

    obj_to_target = torch.norm(obj - target, dim=2)
    tcp_to_obj = torch.norm(obj - tcp_center, dim=2)
    in_place_margin = torch.norm(obj_init_pos - target_pos.unsqueeze(1), dim=2)

    # Placeholder for in_place computation - Replace with actual function
    in_place = reward_utils.tolerance_batch(
        obj_to_target.flatten(),
        bounds=(0, _TARGET_RADIUS),
        margin=in_place_margin.flatten(),
        sigmoid="long_tail",
    )
    in_place = in_place.view(batch_size, n)

    object_grasped = _gripper_caging_reward_batch(
        tcp_center,
        left_pad,
        right_pad,
        init_left_pad,
        init_right_pad,
        obj_init_pos,
        obj,
        action,
    )
    # Placeholder for in_place_and_object_grasped computation - Replace with actual function
    in_place_and_object_grasped = reward_utils.hamacher_product_batch(
        object_grasped, in_place
    )
    if not grasp_rew_only:
        reward = in_place_and_object_grasped
    else:
        reward = object_grasped

    # breakpoint()
    if proximity_reward > 0.0:
        init_dist_to_obj = torch.norm(hand_init_pos - obj_init_pos, dim=2)
        # Additional reward conditions
        reward *= max(
            proximity_reward, 1
        )  # need to double this as well to make sure proximity reward
        # doesn't overshadow it
        close_to_obj = reward_utils.tolerance_batch(
            tcp_to_obj.flatten(),
            bounds=(0, 0.02),
            margin=init_dist_to_obj.flatten(),
            sigmoid="long_tail",
        )
        close_to_obj = close_to_obj.view(batch_size, n)
        reward += proximity_reward * close_to_obj

    # Additional reward conditions
    lift_condition = (
        (tcp_to_obj < 0.02)
        & (tcp_opened > 0)
        & ((obj[:, :, 2] - 0.01) > obj_init_pos[:, :, 2])
    )
    if not grasp_rew_only:
        reward[lift_condition] += 1.0 + 5.0 * in_place[lift_condition]
        close_target_condition = obj_to_target < _TARGET_RADIUS
        reward[close_target_condition] = 10.0 + extra_success_reward
    else:
        reward[lift_condition] += 1

    return reward


def _gripper_caging_reward_batch(
    tcp_center,
    left_pad,
    right_pad,
    init_left_pad,
    init_right_pad,
    obj_init_pos,
    obj_position,
    action=None,
):
    pad_success_margin = 0.05
    x_z_success_margin = 0.005
    obj_radius = 0.015
    batch_size = obj_position.shape[0]
    device = obj_position.device

    delta_object_y_left_pad = left_pad[:, :, 1] - obj_position[:, :, 1]
    delta_object_y_right_pad = obj_position[:, :, 1] - right_pad[:, :, 1]

    right_caging_margin = torch.abs(
        torch.abs(obj_position[:, :, 1] - init_right_pad[:, :, 1]) - pad_success_margin
    )
    left_caging_margin = torch.abs(
        torch.abs(obj_position[:, :, 1] - init_left_pad[:, :, 1]) - pad_success_margin
    )

    # Placeholder for y_caging computation - Replace with actual function
    right_caging = reward_utils.tolerance_batch(
        delta_object_y_right_pad.flatten(),
        bounds=(obj_radius, pad_success_margin),
        margin=right_caging_margin.flatten(),
        sigmoid="long_tail",
    )

    left_caging = reward_utils.tolerance_batch(
        delta_object_y_left_pad.flatten(),
        bounds=(obj_radius, pad_success_margin),
        margin=left_caging_margin.flatten(),
        sigmoid="long_tail",
    )

    y_caging = reward_utils.hamacher_product_batch(left_caging, right_caging)
    y_caging = y_caging.view(batch_size, -1)

    # tcp_xz = tcp_center + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(0)
    # need to send to the right device
    tcp_xz = tcp_center + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(0).to(
        device
    )
    # obj_position_x_z = obj_position + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(0)
    obj_position_x_z = obj_position + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(
        0
    ).unsqueeze(0).to(device)
    tcp_obj_norm_x_z = torch.norm(tcp_xz - obj_position_x_z, dim=2)

    # init_obj_x_z = obj_init_pos + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(0)
    init_obj_x_z = obj_init_pos + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(
        0
    ).to(device)
    init_tcp_x_z = tcp_center + torch.tensor([0.0, -1.0, 0.0]).unsqueeze(0).unsqueeze(
        0
    ).to(device)
    tcp_obj_x_z_margin = (
        torch.norm(init_obj_x_z - init_tcp_x_z, dim=2) - x_z_success_margin
    )
    tcp_obj_x_z_margin = torch.max(torch.tensor(0.0), tcp_obj_x_z_margin)

    # Placeholder for x_z_caging computation - Replace with actual function
    x_z_caging = reward_utils.tolerance_batch(
        tcp_obj_norm_x_z.flatten(),
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_x_z_margin.flatten(),
        sigmoid="long_tail",
    )
    x_z_caging = x_z_caging.view(batch_size, -1)

    # Gripping logic assuming action is provided
    if action is not None:
        gripper_closed = torch.clamp(action[:, -1], min=0, max=1)
    else:
        gripper_closed = torch.ones_like(y_caging).to(device)

    # Placeholder for caging_and_gripping computation - Replace with actual function
    caging = reward_utils.hamacher_product_batch(
        y_caging, x_z_caging
    )  # I guess this works without flattening
    # looking at the code, that seems to be the case
    gripping = torch.where(
        caging > 0.97, gripper_closed, torch.zeros_like(gripper_closed)
    )
    caging_and_gripping = reward_utils.hamacher_product_batch(caging, gripping)
    caging_and_gripping = (caging_and_gripping + caging) / 2
    return caging_and_gripping
