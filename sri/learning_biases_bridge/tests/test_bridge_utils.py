import unittest
from argparse import Namespace

import numpy as np

from learning_biases.bridge_export_dataset import _validate_policy_tensor
from sri.learning_biases_bridge.common import (
    build_policy_demonstrations,
    build_rollout_demonstrations,
    flatten_rewards,
    unflatten_reward_vecs,
)
from sri.learning_biases_bridge.train_sri_policy import (
    _augment_training_split,
    _resolve_regularization,
    _subsample_rollout_demonstrations,
)


class TestBridgeUtils(unittest.TestCase):
    def test_policy_validation_passes_for_walls_and_non_walls(self):
        walls = np.array(
            [
                [[1, 0], [0, 1]],
            ],
            dtype=np.float32,
        )
        policies = np.zeros((1, 2, 2, 5), dtype=np.float32)

        # wall cells: STAY=1
        policies[0, 0, 0, 4] = 1.0
        policies[0, 1, 1, 4] = 1.0

        # non-wall cells: valid distributions
        policies[0, 0, 1] = np.array([0.1, 0.2, 0.3, 0.1, 0.3], dtype=np.float32)
        policies[0, 1, 0] = np.array([0.0, 0.4, 0.2, 0.3, 0.1], dtype=np.float32)

        out = _validate_policy_tensor(walls, policies)
        self.assertLessEqual(out["max_non_wall_row_sum_error"], 1e-4)

    def test_flatten_unflatten_rewards_roundtrip(self):
        rewards = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        flat = flatten_rewards(rewards)
        restored = unflatten_reward_vecs(flat, 3, 4)
        np.testing.assert_allclose(restored, rewards)

    def test_policy_feature_shape(self):
        n, h, w, a = 7, 4, 5, 5
        walls = np.zeros((n, h, w), dtype=np.float32)
        policies = np.full((n, h, w, a), 1.0 / a, dtype=np.float32)

        demos = build_policy_demonstrations(walls, policies)
        self.assertEqual(demos.shape, (n, 1, h * w, a + 3))

    def test_rollout_feature_shape_and_seed_reproducibility(self):
        n, h, w, a = 3, 4, 5, 5
        walls = np.zeros((n, h, w), dtype=np.float32)
        policies = np.full((n, h, w, a), 1.0 / a, dtype=np.float32)
        starts = np.array([[1, 1], [2, 2], [3, 1]], dtype=np.int32)

        d1 = build_rollout_demonstrations(
            walls,
            policies,
            starts,
            num_rollouts_per_policy=4,
            rollout_horizon=6,
            seed=7,
            random_starts=False,
        )
        d2 = build_rollout_demonstrations(
            walls,
            policies,
            starts,
            num_rollouts_per_policy=4,
            rollout_horizon=6,
            seed=7,
            random_starts=False,
        )

        self.assertEqual(d1.shape, (n, 4, 6, a + 3))
        np.testing.assert_allclose(d1, d2)

        action_mass = d1[..., :a].sum(axis=-1)
        np.testing.assert_allclose(action_mass, np.ones_like(action_mass))

        coord_and_time = d1[..., a:]
        self.assertTrue(np.all(coord_and_time >= 0.0))
        self.assertTrue(np.all(coord_and_time <= 1.0))

    def test_rollout_features_start_from_dataset_start(self):
        walls = np.zeros((1, 3, 3), dtype=np.float32)
        policies = np.zeros((1, 3, 3, 5), dtype=np.float32)
        policies[..., 4] = 1.0  # always STAY
        starts = np.array([[2, 1]], dtype=np.int32)

        demos = build_rollout_demonstrations(
            walls,
            policies,
            starts,
            num_rollouts_per_policy=2,
            rollout_horizon=3,
            seed=0,
            random_starts=False,
        )

        # First token for each rollout should reflect the provided start state.
        x_norm = demos[0, :, 0, 5]
        y_norm = demos[0, :, 0, 6]
        np.testing.assert_allclose(x_norm, np.ones_like(x_norm))
        np.testing.assert_allclose(y_norm, np.full_like(y_norm, 0.5))

    def test_rollout_subsample_is_deterministic_and_noop_when_large(self):
        demos = np.arange(2 * 6 * 4 * 3, dtype=np.float32).reshape(2, 6, 4, 3)

        s1 = _subsample_rollout_demonstrations(demos, num_rollouts=3, seed=42)
        s2 = _subsample_rollout_demonstrations(demos, num_rollouts=3, seed=42)
        self.assertEqual(s1.shape, (2, 3, 4, 3))
        np.testing.assert_allclose(s1, s2)

        no_change = _subsample_rollout_demonstrations(demos, num_rollouts=8, seed=42)
        self.assertEqual(no_change.shape, demos.shape)
        np.testing.assert_allclose(no_change, demos)

    def test_regularization_resolution_with_legacy_weight_decay(self):
        args = Namespace(
            regularization_type="none",
            regularization_lambda=1e-5,
            weight_decay=2e-4,
        )
        reg_type, reg_lambda = _resolve_regularization(args)
        self.assertEqual(reg_type, "l2")
        self.assertAlmostEqual(reg_lambda, 2e-4)

    def test_regularization_resolution_explicit_l1(self):
        args = Namespace(
            regularization_type="l1",
            regularization_lambda=3e-5,
            weight_decay=0.0,
        )
        reg_type, reg_lambda = _resolve_regularization(args)
        self.assertEqual(reg_type, "l1")
        self.assertAlmostEqual(reg_lambda, 3e-5)

    def test_d4_augmentation_preserves_semantics(self):
        walls = np.zeros((1, 3, 3), dtype=np.float32)
        rewards = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        starts = np.array([[2, 0]], dtype=np.int32)
        policies = np.zeros((1, 3, 3, 5), dtype=np.float32)
        policies[..., 2] = 1.0  # always EAST

        split = {
            "walls": walls,
            "rewards": rewards,
            "start_states": starts,
            "policies": policies,
        }
        aug = _augment_training_split(split, augmentation="d4")

        self.assertEqual(aug["walls"].shape[0], 8)
        self.assertEqual(aug["rewards"].shape[0], 8)
        self.assertEqual(aug["start_states"].shape[0], 8)
        self.assertEqual(aug["policies"].shape[0], 8)

        # Transform index 1 is rot90 (CCW): (2,0) -> (0,0), EAST -> NORTH.
        np.testing.assert_array_equal(aug["start_states"][1], np.array([0, 0], dtype=np.int32))
        self.assertAlmostEqual(float(aug["policies"][1, 0, 0, 0]), 1.0)  # NORTH

        # Transform index 4 is flip_lr: (2,0) -> (0,0), EAST -> WEST.
        np.testing.assert_array_equal(aug["start_states"][4], np.array([0, 0], dtype=np.int32))
        self.assertAlmostEqual(float(aug["policies"][4, 0, 0, 3]), 1.0)  # WEST

    def test_policy_cnn_and_unet_output_shapes(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not available")

        from sri.learning_biases_bridge.cnn_models import (
            PolicyCNNRegressor,
            PolicyUNetRegressor,
        )

        batch, h, w, f = 3, 16, 16, 8
        demos = torch.randn(batch, 1, h * w, f)

        cnn = PolicyCNNRegressor(height=h, width=w, in_channels=f, base_channels=16)
        out_cnn = cnn(demos)
        self.assertEqual(tuple(out_cnn.shape), (batch, h * w))

        unet = PolicyUNetRegressor(height=h, width=w, in_channels=f, base_channels=16)
        out_unet = unet(demos)
        self.assertEqual(tuple(out_unet.shape), (batch, h * w))


if __name__ == "__main__":
    unittest.main()
