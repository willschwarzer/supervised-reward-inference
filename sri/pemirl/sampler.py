from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass
class TaskTrajectoryBatch:
    obs: torch.Tensor
    acts: torch.Tensor
    next_obs: torch.Tensor
    flat_traj: torch.Tensor



def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)



def trajectories_to_batch(trajectories: Sequence, device: torch.device | str = "cpu") -> TaskTrajectoryBatch:
    parsed = []
    for traj in trajectories:
        obs = _to_numpy(traj.obs)
        acts = _to_numpy(traj.acts)
        t = min(len(acts), len(obs) - 1)
        if t <= 0:
            continue
        parsed.append((obs, acts, t))

    if not parsed:
        raise ValueError("No valid trajectories found for task")

    # Keep a consistent horizon within each task batch.
    min_t = min(t for _, _, t in parsed)
    obs_list = []
    act_list = []
    next_obs_list = []
    flat_list = []
    for obs, acts, _ in parsed:
        obs_t = obs[:min_t]
        obs_tp1 = obs[1 : min_t + 1]
        act_t = acts[:min_t]
        obs_list.append(obs_t)
        act_list.append(act_t)
        next_obs_list.append(obs_tp1)
        flat_list.append(np.concatenate([obs_t, act_t], axis=-1).reshape(-1))

    obs_b = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
    act_b = torch.tensor(np.stack(act_list), dtype=torch.float32, device=device)
    next_obs_b = torch.tensor(np.stack(next_obs_list), dtype=torch.float32, device=device)
    flat_b = torch.tensor(np.stack(flat_list), dtype=torch.float32, device=device)
    return TaskTrajectoryBatch(obs=obs_b, acts=act_b, next_obs=next_obs_b, flat_traj=flat_b)



def build_meta_dataset(rollouts_by_task: Sequence[Sequence], device: torch.device | str = "cpu") -> list[TaskTrajectoryBatch]:
    return [trajectories_to_batch(task_rollouts, device=device) for task_rollouts in rollouts_by_task]



def sample_task_indices(num_tasks: int, meta_batch_size: int, warm_up: bool, warm_up_idx: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if warm_up:
        idx = np.arange(warm_up_idx, warm_up_idx + meta_batch_size) % num_tasks
        next_idx = int((warm_up_idx + meta_batch_size) % num_tasks)
        return idx, next_idx
    idx = rng.integers(0, num_tasks, size=meta_batch_size)
    return idx, warm_up_idx



def sample_traj_indices(num_trajs: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if num_trajs <= 0:
        raise ValueError("num_trajs must be > 0")
    return rng.integers(0, num_trajs, size=batch_size)
