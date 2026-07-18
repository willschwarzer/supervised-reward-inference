from __future__ import annotations

import torch


def airl_log_p_tau(
    reward_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """log p_tau(a|s) term from AIRL-style shaping."""
    return reward_t + gamma * value_tp1 - value_t


def discriminator_terms(
    log_p_tau: torch.Tensor,
    log_q_tau: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (cross_entropy_loss, discrim_output, log_pq).
    Shapes are broadcast-compatible and can include meta-batch/time dimensions.
    """
    labels = labels.to(log_p_tau.dtype)
    log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau], dim=0), dim=0)
    cent = -torch.mean(labels * (log_p_tau - log_pq) + (1.0 - labels) * (log_q_tau - log_pq))
    discrim_output = torch.exp(log_p_tau - log_pq)
    return cent, discrim_output, log_pq


def info_losses(
    log_q_m_tau: torch.Tensor,
    sampled_traj_return: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Matches the MetaIRL objective split:
    - info_loss: gradient target for context encoder
    - info_surr_loss: surrogate for reward parameters
    """
    labels = labels.to(log_q_m_tau.dtype)
    neg_mask = 1.0 - labels
    neg_mean = torch.mean(neg_mask) + eps

    info_loss = -torch.mean(log_q_m_tau * neg_mask) / neg_mean

    baseline = torch.mean(sampled_traj_return * neg_mask, dim=1, keepdim=True) / neg_mean
    centered_return = sampled_traj_return - baseline
    info_surr = -torch.mean(neg_mask * log_q_m_tau * centered_return) / neg_mean
    return info_loss, info_surr


def imitation_likelihood_loss(
    policy_log_prob: torch.Tensor,
) -> torch.Tensor:
    return -policy_log_prob.mean()


def tile_latent_for_time(latents: torch.Tensor, time_steps: int) -> torch.Tensor:
    """[B, Dz] -> [B*T, Dz]"""
    return latents.unsqueeze(1).expand(-1, time_steps, -1).reshape(-1, latents.shape[-1])
