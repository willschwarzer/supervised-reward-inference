from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sri.pemirl.networks import ContextEncoder, DiagGaussianMLPPolicy


@dataclass
class PretrainStats:
    policy_likelihood: float
    kl: float
    total: float


class MetaILPretrainer:
    """Meta-IL initialization for q_psi and pi_omega."""

    def __init__(
        self,
        policy: DiagGaussianMLPPolicy,
        context_encoder: ContextEncoder,
        latent_dim: int,
        max_path_length: int,
        kl_weight: float = 0.1,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim
        self.max_path_length = max_path_length
        self.kl_weight = kl_weight
        self.device = torch.device(device)

        params = list(policy.parameters()) + list(context_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _log_normal_pdf(sample: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        log2pi = torch.log(torch.tensor(2.0 * np.pi, device=sample.device, dtype=sample.dtype))
        return torch.sum(-0.5 * (((sample - mean) ** 2.0) * torch.exp(-logvar) + logvar + log2pi), dim=1)

    def _batch_loss(
        self,
        flat_traj: torch.Tensor,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, info = self.context_encoder.encode(flat_traj, deterministic=False)
        z_t = z.unsqueeze(1).expand(-1, self.max_path_length, -1).reshape(-1, self.latent_dim)

        policy_input = torch.cat([obs, z_t], dim=-1)
        policy_log_prob = self.policy.log_prob(policy_input, act)
        policy_likelihood_loss = -policy_log_prob.mean()

        log_pz = self._log_normal_pdf(z, torch.zeros_like(info.mean), torch.zeros_like(info.log_std * 2.0))
        log_qz = self._log_normal_pdf(z, info.mean, info.log_std * 2.0)
        latent_kl = (log_qz - log_pz).mean()

        total = policy_likelihood_loss + self.kl_weight * latent_kl
        return policy_likelihood_loss, latent_kl, total

    def run(
        self,
        flat_trajs: torch.Tensor,
        obs: torch.Tensor,
        acts: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 400,
    ) -> PretrainStats:
        flat_trajs = flat_trajs.to(self.device)
        obs = obs.to(self.device)
        acts = acts.to(self.device)

        n = flat_trajs.shape[0]
        final_stats = PretrainStats(policy_likelihood=0.0, kl=0.0, total=0.0)
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            p_losses = []
            k_losses = []
            t_losses = []
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                b_flat = flat_trajs[idx]
                b_obs = obs[idx].reshape(-1, obs.shape[-1])
                b_act = acts[idx].reshape(-1, acts.shape[-1])

                self.optimizer.zero_grad(set_to_none=True)
                p_loss, k_loss, t_loss = self._batch_loss(b_flat, b_obs, b_act)
                t_loss.backward()
                self.optimizer.step()

                p_losses.append(float(p_loss.detach().cpu()))
                k_losses.append(float(k_loss.detach().cpu()))
                t_losses.append(float(t_loss.detach().cpu()))
            final_stats = PretrainStats(
                policy_likelihood=float(np.mean(p_losses)) if p_losses else 0.0,
                kl=float(np.mean(k_losses)) if k_losses else 0.0,
                total=float(np.mean(t_losses)) if t_losses else 0.0,
            )
        return final_stats
