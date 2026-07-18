from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sri.pemirl.fusion import KeyedFusionBuffer
from sri.pemirl.networks import DiagGaussianMLPPolicy
from sri.pemirl.objectives import discriminator_terms, imitation_likelihood_loss, info_losses
from sri.pemirl.sampler import build_meta_dataset, sample_task_indices
from sri.pemirl.trpo import trpo_step


def _make_traj(T: int, obs_dim: int, act_dim: int) -> SimpleNamespace:
    obs = np.random.randn(T + 1, obs_dim).astype(np.float32)
    acts = np.random.randn(T, act_dim).astype(np.float32)
    return SimpleNamespace(obs=obs, acts=acts)


def test_meta_batch_shapes() -> None:
    tasks = [
        [_make_traj(7, 5, 2), _make_traj(6, 5, 2), _make_traj(8, 5, 2)],
        [_make_traj(5, 5, 2), _make_traj(5, 5, 2), _make_traj(5, 5, 2)],
    ]
    dataset = build_meta_dataset(tasks, device="cpu")
    assert len(dataset) == 2
    # Task 0 gets truncated to min horizon 6.
    assert dataset[0].obs.shape == (3, 6, 5)
    assert dataset[0].acts.shape == (3, 6, 2)
    assert dataset[0].next_obs.shape == (3, 6, 5)


def test_loss_decomposition_shapes() -> None:
    log_p = torch.randn(4, 3, 5, 1)
    log_q = torch.randn(4, 3, 5, 1)
    labels = torch.randint(0, 2, (4, 3, 5, 1)).float()
    cent, _, _ = discriminator_terms(log_p, log_q, labels)
    assert cent.ndim == 0

    log_qm = torch.randn(4, 3, 1)
    traj_ret = torch.randn(4, 3, 1)
    labels_step = torch.randint(0, 2, (4, 3, 1, 1)).float()
    info, info_surr = info_losses(log_qm, traj_ret, labels_step)
    assert info.ndim == 0
    assert info_surr.ndim == 0

    imit = imitation_likelihood_loss(torch.randn(32, 1))
    assert imit.ndim == 0


def test_fusion_buffer_and_warmup_sampling() -> None:
    buf = KeyedFusionBuffer(buffer_size=10, subsample_ratio=0.5, seed=0)
    paths = {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}
    buf.add_paths(paths, subsample=True)
    assert len(buf) > 0
    sampled = buf.sample_paths([0, 1], n=2)
    assert sampled is not None
    assert set(sampled.keys()) == {0, 1}

    idxs, next_idx = sample_task_indices(
        num_tasks=10,
        meta_batch_size=4,
        warm_up=True,
        warm_up_idx=8,
        rng=np.random.default_rng(0),
    )
    assert next_idx == 2
    assert idxs.tolist() == [8, 9, 0, 1]


def test_trpo_step_respects_kl_when_accepted() -> None:
    torch.manual_seed(0)
    policy = DiagGaussianMLPPolicy(input_dim=6, output_dim=2, hidden_sizes=(16, 16))
    old = policy.clone()
    obs = torch.randn(64, 6)
    acts = torch.randn(64, 2)
    adv = torch.randn(64, 1)

    stats = trpo_step(
        policy=policy,
        old_policy=old,
        obs=obs,
        actions=acts,
        advantages=adv,
        max_kl=0.01,
        cg_iters=8,
    )
    if stats.accepted_step:
        assert stats.mean_kl_after <= 0.011
