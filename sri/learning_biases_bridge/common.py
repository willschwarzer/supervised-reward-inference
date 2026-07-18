import importlib
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np


def load_manifest(dataset_dir: str) -> Dict[str, object]:
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(dataset_dir: str, split_name: str) -> Dict[str, np.ndarray]:
    path = os.path.join(dataset_dir, f"{split_name}.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def flatten_rewards(rewards: np.ndarray) -> np.ndarray:
    return rewards.reshape(rewards.shape[0], -1).astype(np.float32)


def unflatten_reward_vecs(reward_vecs: np.ndarray, height: int, width: int) -> np.ndarray:
    if reward_vecs.shape[-1] != height * width:
        raise ValueError(
            f"Expected final dim {height*width}, got {reward_vecs.shape[-1]}"
        )
    return reward_vecs.reshape(reward_vecs.shape[0], height, width).astype(np.float32)


def build_policy_demonstrations(walls: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """Create SRI-policy demonstrations with shape (N, 1, H*W, A+3).

    Features are [policy_probs(5), wall_indicator(1), x_norm(1), y_norm(1)].
    """
    if walls.shape[:3] != policies.shape[:3]:
        raise ValueError("walls and policies must agree on (N,H,W)")

    n, h, w = walls.shape
    a = policies.shape[-1]

    policy_flat = policies.reshape(n, h * w, a).astype(np.float32)
    wall_flat = walls.reshape(n, h * w, 1).astype(np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    x_denom = float(max(w - 1, 1))
    y_denom = float(max(h - 1, 1))
    x_norm = (x_coords / x_denom).reshape(1, h * w, 1)
    y_norm = (y_coords / y_denom).reshape(1, h * w, 1)
    x_norm = np.repeat(x_norm, n, axis=0)
    y_norm = np.repeat(y_norm, n, axis=0)

    features = np.concatenate([policy_flat, wall_flat, x_norm, y_norm], axis=-1)
    return features[:, None, :, :].astype(np.float32)


_ACTION_DELTAS = np.asarray(
    [
        [0, -1],  # north
        [0, 1],   # south
        [1, 0],   # east
        [-1, 0],  # west
        [0, 0],   # stay
    ],
    dtype=np.int32,
)


def _build_policy_cdf(policies: np.ndarray) -> np.ndarray:
    """Return per-state action CDFs with robust handling of invalid rows."""
    policy = np.asarray(policies, dtype=np.float32)
    policy = np.clip(policy, 0.0, None)
    row_sums = np.sum(policy, axis=-1, keepdims=True)
    num_actions = policy.shape[-1]

    fallback = np.full_like(policy, fill_value=1.0 / float(num_actions), dtype=np.float32)
    normalized = np.where(row_sums > 0.0, policy / row_sums, fallback)
    cdf = np.cumsum(normalized, axis=-1, dtype=np.float32)
    cdf[..., -1] = 1.0
    return cdf


def _sample_action(rng: np.random.Generator, probs: np.ndarray) -> int:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return int(rng.integers(0, probs.shape[0]))
    probs = probs / total
    return int(rng.choice(probs.shape[0], p=probs))


def _step_in_grid(x: int, y: int, action_idx: int, wall_map: np.ndarray) -> Tuple[int, int]:
    h, w = wall_map.shape
    if action_idx < 0 or action_idx >= _ACTION_DELTAS.shape[0]:
        return x, y
    dx, dy = _ACTION_DELTAS[action_idx]
    nx = x + int(dx)
    ny = y + int(dy)
    if nx < 0 or nx >= w or ny < 0 or ny >= h or wall_map[ny, nx]:
        return x, y
    return nx, ny


def build_rollout_demonstrations(
    walls: np.ndarray,
    policies: np.ndarray,
    start_states: np.ndarray,
    num_rollouts_per_policy: int = 100,
    rollout_horizon: int = 20,
    seed: int = 0,
    random_starts: bool = False,
) -> np.ndarray:
    """Sample trajectory demonstrations from policy tensors.

    Returns shape (N, R, T, A+3), where token features are:
    [sampled_action_one_hot(A), x_norm, y_norm, t_norm].
    """
    if walls.shape[:3] != policies.shape[:3]:
        raise ValueError("walls and policies must agree on (N,H,W)")
    if start_states.shape[0] != walls.shape[0]:
        raise ValueError("start_states must have one row per task")
    if num_rollouts_per_policy < 1:
        raise ValueError("num_rollouts_per_policy must be >= 1")
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be >= 1")

    n, h, w = walls.shape
    a = policies.shape[-1]
    if a != _ACTION_DELTAS.shape[0]:
        raise ValueError(f"Expected { _ACTION_DELTAS.shape[0] } actions, got {a}")

    rng = np.random.default_rng(seed)
    demos = np.zeros((n, num_rollouts_per_policy, rollout_horizon, a + 3), dtype=np.float32)
    policy_cdf = _build_policy_cdf(policies)
    x_denom = float(max(w - 1, 1))
    y_denom = float(max(h - 1, 1))
    t_denom = float(max(rollout_horizon - 1, 1))
    rollout_idx = np.arange(num_rollouts_per_policy, dtype=np.int64)
    action_dx = _ACTION_DELTAS[:, 0]
    action_dy = _ACTION_DELTAS[:, 1]

    for task_idx in range(n):
        wall_map = walls[task_idx].astype(bool)
        non_wall_yx = np.argwhere(~wall_map)
        if non_wall_yx.size == 0:
            raise ValueError(f"Task {task_idx} has no traversable states")

        start_x = int(start_states[task_idx][0])
        start_y = int(start_states[task_idx][1])
        fixed_start_valid = (
            0 <= start_x < w and 0 <= start_y < h and not wall_map[start_y, start_x]
        )
        task_demos = demos[task_idx]
        task_policy_cdf = policy_cdf[task_idx]

        if random_starts or not fixed_start_valid:
            start_sel = rng.integers(0, non_wall_yx.shape[0], size=num_rollouts_per_policy)
            starts = non_wall_yx[start_sel]
            y = starts[:, 0].astype(np.int32, copy=False)
            x = starts[:, 1].astype(np.int32, copy=False)
        else:
            x = np.full(num_rollouts_per_policy, start_x, dtype=np.int32)
            y = np.full(num_rollouts_per_policy, start_y, dtype=np.int32)

        for t in range(rollout_horizon):
            state_cdf = task_policy_cdf[y, x]  # (R, A)
            u = rng.random(num_rollouts_per_policy, dtype=np.float32)
            action_idx = np.sum(u[:, None] > state_cdf, axis=1, dtype=np.int32)

            task_demos[rollout_idx, t, action_idx] = 1.0
            task_demos[:, t, a] = x.astype(np.float32, copy=False) / x_denom
            task_demos[:, t, a + 1] = y.astype(np.float32, copy=False) / y_denom
            task_demos[:, t, a + 2] = float(t) / t_denom

            nx = x + action_dx[action_idx]
            ny = y + action_dy[action_idx]
            in_bounds = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            valid = in_bounds.copy()
            if np.any(in_bounds):
                valid[in_bounds] &= ~wall_map[ny[in_bounds], nx[in_bounds]]
            x = np.where(valid, nx, x)
            y = np.where(valid, ny, y)

    return demos


def evaluate_reward_predictions(
    walls: np.ndarray,
    start_states: np.ndarray,
    predicted_rewards: np.ndarray,
    true_rewards: np.ndarray,
    gamma: float = 0.9,
    episode_length: int = 20,
) -> Dict[str, object]:
    _ensure_legacy_import_paths()
    from learning_biases.agent_runner import evaluate_proxy

    per_task = []
    for wall, start_state, pred, label in zip(
        walls, start_states, predicted_rewards, true_rewards
    ):
        per_task.append(
            float(
                evaluate_proxy(
                    wall,
                    tuple(np.asarray(start_state).tolist()),
                    pred,
                    label,
                    gamma=gamma,
                    episode_length=episode_length,
                )
            )
        )
    avg = float(np.mean(per_task)) if per_task else 0.0
    return {
        "average_percent_reward": avg,
        "average_regret": 1.0 - avg,
        "percent_rewards": per_task,
        "num_tasks": int(len(per_task)),
        "gamma": float(gamma),
        "episode_length": int(episode_length),
    }


def maybe_load_json(path: str):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def grid_shape_from_split(split: Dict[str, np.ndarray]) -> Tuple[int, int]:
    _, h, w = split["walls"].shape
    return int(h), int(w)


def _ensure_legacy_import_paths() -> None:
    """Allow legacy absolute imports used by learning_biases modules."""
    pkg_root = os.path.dirname(os.path.dirname(__file__))
    repo_root = os.path.dirname(pkg_root)
    lb_dir = os.path.join(repo_root, "learning_biases")
    for p in (repo_root, lb_dir):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, lb_dir)
    sys.path.insert(0, repo_root)

    if "gridworld" in sys.modules and not hasattr(sys.modules["gridworld"], "__path__"):
        del sys.modules["gridworld"]
    importlib.import_module("gridworld")
