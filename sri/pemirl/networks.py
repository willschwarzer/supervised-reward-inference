from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class GaussianDistInfo:
    mean: torch.Tensor
    log_std: torch.Tensor



def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class DiagGaussianMLPPolicy(nn.Module):
    """Gaussian policy pi(a|s,m) and context encoder q(m|tau) backbone."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        log_std_init: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.state_dependent_std = state_dependent_std

        self.mean_net = _build_mlp(input_dim, hidden_sizes, output_dim)
        if state_dependent_std:
            self.log_std_net = _build_mlp(input_dim, hidden_sizes, output_dim)
            self.log_std_param = None
        else:
            self.log_std_net = None
            self.log_std_param = nn.Parameter(torch.ones(output_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> GaussianDistInfo:
        mean = self.mean_net(obs)
        if self.state_dependent_std:
            assert self.log_std_net is not None
            log_std = self.log_std_net(obs)
        else:
            assert self.log_std_param is not None
            log_std = self.log_std_param.expand_as(mean)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return GaussianDistInfo(mean=mean, log_std=log_std)

    def distribution(self, obs: torch.Tensor) -> Normal:
        info = self.forward(obs)
        return Normal(info.mean, info.log_std.exp())

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.entropy().sum(dim=-1, keepdim=True)

    def kl_divergence(self, obs: torch.Tensor, other: "DiagGaussianMLPPolicy") -> torch.Tensor:
        info_1 = self.forward(obs)
        info_0 = other.forward(obs)
        var_1 = (2.0 * info_1.log_std).exp()
        var_0 = (2.0 * info_0.log_std).exp()
        mean_diff_sq = (info_0.mean - info_1.mean).pow(2)
        kl = info_1.log_std - info_0.log_std + (var_0 + mean_diff_sq) / (2.0 * var_1) - 0.5
        return kl.sum(dim=-1, keepdim=True)

    def clone(self) -> "DiagGaussianMLPPolicy":
        return copy.deepcopy(self)


class ContextEncoder(DiagGaussianMLPPolicy):
    """q_psi(m|tau): outputs Gaussian over latent context."""

    def encode(self, flat_traj: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, GaussianDistInfo]:
        info = self.forward(flat_traj)
        dist = Normal(info.mean, info.log_std.exp())
        if deterministic:
            z = dist.mean
        else:
            z = dist.rsample()
        return z, info


class RewardNet(nn.Module):
    """r_theta(s,m)"""

    def __init__(self, state_dim: int, latent_dim: int, hidden_size: int = 32, layers: int = 2) -> None:
        super().__init__()
        hidden_sizes = tuple(hidden_size for _ in range(layers))
        self.net = _build_mlp(state_dim + latent_dim, hidden_sizes, 1)

    def forward(self, states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, latents], dim=-1)
        return self.net(x)


class PotentialNet(nn.Module):
    """h_phi(s,m)"""

    def __init__(self, state_dim: int, latent_dim: int, hidden_size: int = 32, layers: int = 2) -> None:
        super().__init__()
        hidden_sizes = tuple(hidden_size for _ in range(layers))
        self.net = _build_mlp(state_dim + latent_dim, hidden_sizes, 1)

    def forward(self, states: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, latents], dim=-1)
        return self.net(x)
