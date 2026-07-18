from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sri.pemirl.sampler import trajectories_to_batch


@dataclass
class AdaptStats:
    final_loss: float
    context_norm: float


class AdaptedPEMIRLPolicy:
    """SB3-compatible policy wrapper exposing predict()."""

    def __init__(self, policy: torch.nn.Module, context: torch.Tensor, device: torch.device | str = "cpu") -> None:
        self.policy = policy
        self.context = context.detach()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            z = self.context.unsqueeze(0).expand(obs_t.shape[0], -1)
            policy_input = torch.cat([obs_t, z], dim=-1)
            action, _ = self.policy.sample(policy_input, deterministic=deterministic)
            action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy(), None



def infer_context(model: Any, task_rollouts: list, deterministic: bool = True) -> torch.Tensor:
    batch = trajectories_to_batch(task_rollouts, device=model.device)
    flat = batch.flat_traj
    with torch.no_grad():
        z, info = model.context_encoder.encode(flat, deterministic=deterministic)
        if deterministic:
            ctx = info.mean.mean(dim=0)
        else:
            ctx = z.mean(dim=0)
    return ctx



def adapt_policy(
    model: Any,
    task_rollouts: list,
    adapt_iters: int = 200,
    lr: float = 1e-3,
    deterministic_context: bool = True,
) -> tuple[AdaptedPEMIRLPolicy, AdaptStats]:
    batch = trajectories_to_batch(task_rollouts, device=model.device)
    context = infer_context(model, task_rollouts, deterministic=deterministic_context)

    policy = model.policy.clone().to(model.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs = batch.obs.reshape(-1, batch.obs.shape[-1])
    acts = batch.acts.reshape(-1, batch.acts.shape[-1])
    z = context.unsqueeze(0).expand(obs.shape[0], -1)
    inp = torch.cat([obs, z], dim=-1)

    final_loss = 0.0
    for _ in range(max(1, adapt_iters)):
        optimizer.zero_grad(set_to_none=True)
        log_prob = policy.log_prob(inp, acts)
        loss = -log_prob.mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    adapted = AdaptedPEMIRLPolicy(policy=policy, context=context, device=model.device)
    stats = AdaptStats(final_loss=final_loss, context_norm=float(context.norm().detach().cpu()))
    return adapted, stats



def save_adapted_policy(path: str, adapted_policy: AdaptedPEMIRLPolicy) -> None:
    payload = {
        "policy_state_dict": adapted_policy.policy.state_dict(),
        "context": adapted_policy.context.detach().cpu(),
    }
    torch.save(payload, path)
