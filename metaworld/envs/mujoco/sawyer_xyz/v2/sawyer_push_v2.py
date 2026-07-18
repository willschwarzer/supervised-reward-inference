import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import _gripper_caging_reward_batch
import mujoco
import os
import torch


class SawyerPushEnvV2(SawyerXYZEnv):
    """SawyerPushEnv.

    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after reaching the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """

    TARGET_RADIUS = 0.05

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
        success_requires_touch=True,
        third_gt=False,
        half_gt=False,
        ensembling=None,
        include_extra_reward_info=False,
        hand_starts=None,
        grasp_rew_only=False,
    ):
        if goal_bounds is None:
            goal_low = (-0.1, 0.8, 0.02)
            goal_high = (0.1, 0.9, 0.01)
        else:
            goal_low, goal_high = goal_bounds
        hand_low = (-0.5, 0.40, 0.05)
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
            third_gt=third_gt,
            half_gt=half_gt,
            ensembling=ensembling,
            include_extra_reward_info=include_extra_reward_info,
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

        self.goal = np.array([0.1, 0.8, 0.02])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.num_resets = 0
        self.extra_success_reward = extra_success_reward
        self.success_requires_touch = success_requires_touch

        if limit_reward_obs:
            self.reward_dims = REWARD_DIMS
        else:
            self.reward_dims = None

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_push_v2.xml")

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
        # success = float(obj_to_target <= 0.07)
        success = float(obj_to_target <= self.TARGET_RADIUS)
        if self.success_requires_touch:
            invalid_success = tcp_to_obj > 0.05
            success = success and not invalid_success
        if success:
            print("Obj position:", obj)
            print("Target position:", self._target_pos)
            print("obj_to_target:", obj_to_target)
            print("tcp_center:", tcp_center)
            print("tcp_to_obj:", tcp_to_obj)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_main_object and (tcp_open > 0)
                              and (obj[2] - 0.02 > self.obj_init_pos[2]))
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

    def _get_quat_objects(self):
        geom_xmat = self.data.geom("objGeom").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def update_goal_site(self, new_goal_position):
        """
        Update the position of the 'goal' site in the MuJoCo environment.

        Args:
        new_goal_position (np.ndarray): A 3D numpy array specifying the new position of the goal.
        """
        # Find the ID of the 'goal' site in the MuJoCo model
        goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                         "goal")

        # Update the position of the 'goal' site
        self.model.site_pos[goal_site_id] = new_goal_position

        # Optionally, if immediate visual update in the simulation is required
        mujoco.mj_forward(self.model, self.data)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(
            self.fix_extreme_obj_pos(self.init_config["obj_init_pos"]))
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]

        if self.gripped_start:
            # Load the saved gripped state
            # grasp_pos_qpos = np.load("grasp_pos/pick_place_0.npy")
            # grasp_pos_qvel = np.load("grasp_vel/pick_place_0.npy")

            idx = np.random.randint(10000)
            # make sure the file exists, otherwise sample again
            while True:
                try:
                    grasp_pos_qpos = np.load(f"grasp_pos/pick_place_{idx}.npy")
                    grasp_pos_qvel = np.load(f"grasp_vel/pick_place_{idx}.npy")
                    break
                except:
                    idx = np.random.randint(10000)
            # XXX zeroing out qvel for now
            grasp_pos_qvel = np.zeros_like(grasp_pos_qvel)

            # Set the simulation's state to the saved gripped position
            self.set_state(grasp_pos_qpos, grasp_pos_qvel)

            # Adjust the object's initial position based on the gripped state
            self.obj_init_pos = grasp_pos_qpos[
                9:12]
        else:
            self.obj_init_pos = np.concatenate(
                (goal_pos[:2], [self.obj_init_pos[-1]]))
        while np.linalg.norm(self.obj_init_pos[:2] -
                             self._target_pos[:2]) < 0.15:
            # print(f"Target pos: {self._target_pos[:2]}")
            # print(f"Obj pos: {self.obj_init_pos[:2]}")
            goal_pos = self._get_state_rand_vec()
            self._target_pos = np.concatenate(
                (goal_pos[-3:-1], [self.obj_init_pos[-1]]))
        # change the MuJoCo simulation to reflect the new goal
        self.update_goal_site(self._target_pos)
        self._set_obj_xyz(self.obj_init_pos)

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)
        target_to_obj = np.linalg.norm(obj - self._target_pos)
        target_to_obj_init = np.linalg.norm(self.obj_init_pos -
                                            self._target_pos)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid="long_tail",
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True,
        )
        reward = 2 * object_grasped

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1.0 + reward + 5.0 * in_place
        invalid_success = self.success_requires_touch and tcp_to_obj > 0.05
        if target_to_obj < self.TARGET_RADIUS and not invalid_success:
            reward = 10.0 + self.extra_success_reward
        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped,
                in_place)


class TrainPushv2(SawyerPushEnvV2):
    tasks = None

    def __init__(self):
        SawyerPushEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


def compute_reward(
    action,
    obs,
    tcp_center,
    target_pos,
    obj_init_pos,
    TARGET_RADIUS,
    gripper_caging_reward,
):
    obj = obs[4:7]
    tcp_opened = obs[3]
    tcp_to_obj = np.linalg.norm(obj - tcp_center)
    target_to_obj = np.linalg.norm(obj - target_pos)
    target_to_obj_init = np.linalg.norm(obj_init_pos - target_pos)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, TARGET_RADIUS),
        margin=target_to_obj_init,
        sigmoid="long_tail",
    )

    object_grasped = gripper_caging_reward(
        action,
        obj,
        object_reach_radius=0.01,
        obj_radius=0.015,
        pad_success_thresh=0.05,
        xz_thresh=0.005,
        high_density=True,
    )
    reward = 2 * object_grasped

    if tcp_to_obj < 0.02 and tcp_opened > 0:
        reward += 1.0 + reward + 5.0 * in_place
    if target_to_obj < TARGET_RADIUS:
        reward = 10.0
    return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)


REWARD_DIMS = np.r_[6:9, 9:12, 12:15, 15:18, 24, 25:28]  # XXX INCORRECT


def compute_reward_batch(
    obs, target_pos, action=None, extra_success_reward=0.0, limit_reward_obs=False
):
    # breakpoint()
    # Constants
    _TARGET_RADIUS = 0.05

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
    target = target_pos.unsqueeze(1).expand(-1, obs.shape[1], -1)

    # Calculate distances
    tcp_to_obj = torch.norm(obj - tcp_center, dim=2)
    target_to_obj = torch.norm(obj - target, dim=2)
    target_to_obj_init = torch.norm(obj_init_pos - target, dim=2)

    in_place = reward_utils.tolerance_batch(
        target_to_obj, bounds=(0, _TARGET_RADIUS), margin=target_to_obj_init, sigmoid="long_tail"
    )

    # object_grasped = _gripper_caging_reward_batch(tcp_center, left_pad, right_pad, init_left_pad, init_right_pad, obj_init_pos, obj, action)

    # Compute object grasped reward using a placeholder gripper caging function adapted for batches
    # object_grasped = _gripper_caging_reward_batch(
    #     action,
    #     obj,
    #     object_reach_radius=0.01,
    #     obj_radius=0.015,
    #     pad_success_thresh=0.05,
    #     xz_thresh=0.005,
    #     high_density=True
    # )

    object_grasped = _gripper_caging_reward_batch(
        left_pad,
        right_pad,
        obj_init_pos,
        tcp_center,
        hand_init_pos,
        action,
        obj,
        0.01,
        0.015,
        0.05,
        0.005,
        desired_gripper_effort=1.0,
        high_density=True,
    )

    # Compute reward based on conditions
    reward = 2 * object_grasped
    lift_condition = (tcp_to_obj < 0.02) & (tcp_opened > 0)
    reward[lift_condition] += 1.0 + 5.0 * in_place[lift_condition]

    # Extra success condition
    close_target_condition = target_to_obj < _TARGET_RADIUS
    reward[close_target_condition] = 10.0 + extra_success_reward

    return reward


def _gripper_caging_reward_batch(
    left_pad_pos,
    right_pad_pos,
    obj_init_pos,
    tcp_center,
    init_tcp,
    action,
    obj_pos,
    object_reach_radius,
    obj_radius,
    pad_success_thresh,
    xz_thresh,
    desired_gripper_effort=1.0,
    high_density=False,
    medium_density=False,
):
    """Reward for agent grasping obj.

    Args:
        action(torch.Tensor): (batch_size, 4) tensor representing the action
            delta(x), delta(y), delta(z), gripper_effort
        obj_pos(torch.Tensor): (batch_size, 3) tensor representing the obj x,y,z
        obj_radius(float): radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
        left_pad_pos(torch.Tensor): (batch_size, 3) tensor representing left pad position
        right_pad_pos(torch.Tensor): (batch_size, 3) tensor representing right pad position
        obj_init_pos(torch.Tensor): (3,) tensor representing initial object position
        tcp_center(torch.Tensor): (batch_size, 3) tensor representing TCP center
        init_tcp(torch.Tensor): (3,) tensor representing initial TCP position
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")

    # MARK: Left-right gripper information for caging reward----------------
    pad_y_lr = torch.cat((left_pad_pos[..., 1:2], right_pad_pos[..., 1:2]), dim=-1)
    pad_to_obj_lr = torch.abs(pad_y_lr - obj_pos[..., 1:2])
    pad_to_objinit_lr = torch.abs(pad_y_lr - obj_init_pos[..., 1:2])

    caging_lr_margin = torch.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = reward_utils.tolerance_batch(
        pad_to_obj_lr,
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin,
        sigmoid="long_tail",
    )
    caging_y = reward_utils.hamacher_product_batch(caging_lr[..., 0], caging_lr[..., 1])

    # MARK: X-Z gripper information for caging reward-----------------------
    xz = [0, 2]
    caging_xz_margin = torch.norm(obj_init_pos[..., xz] - init_tcp[..., xz], dim=-1)
    caging_xz_margin -= xz_thresh
    # breakpoint()
    caging_xz = reward_utils.tolerance_batch(
        torch.norm(tcp_center[..., xz] - obj_pos[..., xz], dim=-1),
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,
        sigmoid="long_tail",
    )

    if action is None:
        action_shape = list(obj_pos.shape)
        action_shape[-1] = 4
        action = torch.ones(action_shape)
    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
        torch.clamp(action[..., -1], 0, desired_gripper_effort) / desired_gripper_effort
    )
    # move to device
    gripper_closed = gripper_closed.to(obj_pos.device)

    # MARK: Combine components----------------------------------------------
    caging = reward_utils.hamacher_product_batch(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, gripper_closed, torch.zeros_like(gripper_closed))
    caging_and_gripping = reward_utils.hamacher_product_batch(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp_to_obj = torch.norm(obj_pos - tcp_center, dim=1)
        tcp_to_obj_init = torch.norm(obj_init_pos - init_tcp)
        reach_margin = torch.abs(tcp_to_obj_init - object_reach_radius)
        reach = reward_utils.tolerance_batch(
            tcp_to_obj, bounds=(0, object_reach_radius), margin=reach_margin, sigmoid="long_tail"
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping


class TestPushv2(SawyerPushEnvV2):
    tasks = None

    def __init__(self):
        SawyerPushEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
