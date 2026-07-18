from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from sri.pemirl.networks import DiagGaussianMLPPolicy


@dataclass
class TRPOStats:
    loss_before: float
    loss_after: float
    mean_kl_before: float
    mean_kl_after: float
    accepted_step: bool
    nonfinite_candidates: int = 0
    exception_candidates: int = 0
    diagnostic: str = ""


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach()
    finite = torch.isfinite(t)
    finite_count = int(finite.sum().item())
    total = int(t.numel())
    if finite_count > 0:
        t_fin = t[finite]
        t_min = float(t_fin.min().cpu())
        t_max = float(t_fin.max().cpu())
        t_mean = float(t_fin.mean().cpu())
        t_std = float(t_fin.std().cpu()) if t_fin.numel() > 1 else 0.0
    else:
        t_min = float("nan")
        t_max = float("nan")
        t_mean = float("nan")
        t_std = float("nan")
    return (
        f"{name}: shape={tuple(t.shape)} finite={finite_count}/{total} "
        f"min={t_min:.6g} max={t_max:.6g} mean={t_mean:.6g} std={t_std:.6g}"
    )


def _params_are_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if not torch.isfinite(p.detach()).all():
            return False
    return True



def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])



def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n



def _flat_grad(grads: list[torch.Tensor], model: torch.nn.Module) -> torch.Tensor:
    views = []
    for g, p in zip(grads, model.parameters()):
        if g is None:
            views.append(torch.zeros_like(p).reshape(-1))
        else:
            views.append(g.reshape(-1))
    return torch.cat(views)



def conjugate_gradient(Avp, b: torch.Tensor, cg_iters: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = Avp(p)
        v = rdotr / (torch.dot(p, z) + 1e-8)
        x = x + v * p
        r = r - v * z
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        mu = new_rdotr / (rdotr + 1e-8)
        p = r + mu * p
        rdotr = new_rdotr
    return x



def trpo_step(
    policy: DiagGaussianMLPPolicy,
    old_policy: DiagGaussianMLPPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    max_kl: float = 0.01,
    cg_iters: int = 10,
    cg_damping: float = 1e-5,
    backtrack_ratio: float = 0.8,
    max_backtracks: int = 15,
    ent_weight: float = 0.0,
    debug_nonfinite: bool = False,
    reject_nonfinite_steps: bool = True,
    debug_prefix: str = "",
) -> TRPOStats:
    prefix = f"[TRPO {debug_prefix}] " if debug_prefix else "[TRPO] "
    diagnostics: list[str] = []

    def _diag(msg: str) -> None:
        diagnostics.append(msg)
        if debug_nonfinite:
            print(f"{prefix}{msg}")

    if not _params_are_finite(policy.parameters()):
        _diag("policy parameters are non-finite before TRPO step")
        return TRPOStats(
            loss_before=float("nan"),
            loss_after=float("nan"),
            mean_kl_before=float("nan"),
            mean_kl_after=float("nan"),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )

    with torch.no_grad():
        old_info = old_policy.forward(obs)
    old_log_prob = old_policy.log_prob(obs, actions).detach()
    if debug_nonfinite:
        if not torch.isfinite(old_info.mean).all():
            _diag(_tensor_summary("old_info.mean", old_info.mean))
        if not torch.isfinite(old_info.log_std).all():
            _diag(_tensor_summary("old_info.log_std", old_info.log_std))
        if not torch.isfinite(old_log_prob).all():
            _diag(_tensor_summary("old_log_prob", old_log_prob))

    def surrogate_loss() -> torch.Tensor:
        log_prob = policy.log_prob(obs, actions)
        ratio = torch.exp(log_prob - old_log_prob)
        ent_bonus = ent_weight * policy.entropy(obs)
        return -(ratio * (advantages + ent_bonus)).mean()

    def mean_kl() -> torch.Tensor:
        info = policy.forward(obs)
        old_var = (2.0 * old_info.log_std).exp()
        new_var = (2.0 * info.log_std).exp()
        kl = info.log_std - old_info.log_std + (old_var + (old_info.mean - info.mean).pow(2)) / (2.0 * new_var) - 0.5
        return kl.sum(dim=-1, keepdim=True).mean()

    loss_before_t = surrogate_loss()
    kl_before_t = mean_kl()
    if not torch.isfinite(loss_before_t):
        _diag(_tensor_summary("loss_before_t", loss_before_t))
        return TRPOStats(
            loss_before=float("nan"),
            loss_after=float("nan"),
            mean_kl_before=float("nan"),
            mean_kl_after=float("nan"),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )
    if not torch.isfinite(kl_before_t):
        _diag(_tensor_summary("kl_before_t", kl_before_t))
        return TRPOStats(
            loss_before=float(loss_before_t.detach().cpu()),
            loss_after=float(loss_before_t.detach().cpu()),
            mean_kl_before=float("nan"),
            mean_kl_after=float("nan"),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )

    grads = torch.autograd.grad(loss_before_t, policy.parameters(), create_graph=True)
    loss_grad = _flat_grad(list(grads), policy).detach()
    if not torch.isfinite(loss_grad).all():
        _diag(_tensor_summary("loss_grad", loss_grad))
        return TRPOStats(
            loss_before=float(loss_before_t.detach().cpu()),
            loss_after=float(loss_before_t.detach().cpu()),
            mean_kl_before=float(kl_before_t.detach().cpu()),
            mean_kl_after=float(kl_before_t.detach().cpu()),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )

    def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
        kl = mean_kl()
        grads_kl = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_grad_kl = _flat_grad(list(grads_kl), policy)
        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, policy.parameters(), retain_graph=True)
        flat_grad2 = _flat_grad(list(grads2), policy).detach()
        return flat_grad2 + cg_damping * v

    step_dir = conjugate_gradient(fisher_vector_product, -loss_grad, cg_iters=cg_iters)
    if not torch.isfinite(step_dir).all():
        _diag(_tensor_summary("step_dir", step_dir))
        return TRPOStats(
            loss_before=float(loss_before_t.detach().cpu()),
            loss_after=float(loss_before_t.detach().cpu()),
            mean_kl_before=float(kl_before_t.detach().cpu()),
            mean_kl_after=float(kl_before_t.detach().cpu()),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )
    shs = 0.5 * torch.dot(step_dir, fisher_vector_product(step_dir))
    lm = torch.sqrt(torch.clamp(shs / max_kl, min=1e-8))
    full_step = step_dir / (lm + 1e-8)
    if not torch.isfinite(shs):
        _diag(_tensor_summary("shs", shs))
    if not torch.isfinite(full_step).all():
        _diag(_tensor_summary("full_step", full_step))
        return TRPOStats(
            loss_before=float(loss_before_t.detach().cpu()),
            loss_after=float(loss_before_t.detach().cpu()),
            mean_kl_before=float(kl_before_t.detach().cpu()),
            mean_kl_after=float(kl_before_t.detach().cpu()),
            accepted_step=False,
            diagnostic="; ".join(diagnostics),
        )

    old_params = _flat_params(policy)

    accepted = False
    nonfinite_candidates = 0
    exception_candidates = 0
    loss_before = float(loss_before_t.detach().cpu())
    kl_before = float(kl_before_t.detach().cpu())
    loss_after = loss_before
    kl_after = kl_before

    for backtrack_idx in range(max_backtracks + 1):
        frac = backtrack_ratio ** backtrack_idx
        new_params = old_params + frac * full_step
        if reject_nonfinite_steps and not torch.isfinite(new_params).all():
            nonfinite_candidates += 1
            _diag(f"non-finite candidate parameters at backtrack={backtrack_idx}, frac={frac:.6g}")
            continue
        _set_flat_params(policy, new_params)

        try:
            loss_new_t = surrogate_loss().detach()
            kl_new_t = mean_kl().detach()
        except Exception as exc:  # noqa: BLE001
            exception_candidates += 1
            _diag(
                f"exception while evaluating candidate at backtrack={backtrack_idx}, "
                f"frac={frac:.6g}: {type(exc).__name__}: {exc}"
            )
            continue
        if reject_nonfinite_steps and (
            not torch.isfinite(loss_new_t) or not torch.isfinite(kl_new_t)
        ):
            nonfinite_candidates += 1
            _diag(
                f"non-finite loss/kl at backtrack={backtrack_idx}, frac={frac:.6g}; "
                f"{_tensor_summary('loss_new_t', loss_new_t)}; {_tensor_summary('kl_new_t', kl_new_t)}"
            )
            continue
        loss_new = float(loss_new_t.cpu())
        kl_new = float(kl_new_t.cpu())

        improve = loss_new < loss_before
        if improve and kl_new <= max_kl:
            accepted = True
            loss_after = loss_new
            kl_after = kl_new
            break

    if not accepted:
        _set_flat_params(policy, old_params)

    return TRPOStats(
        loss_before=loss_before,
        loss_after=loss_after,
        mean_kl_before=kl_before,
        mean_kl_after=kl_after,
        accepted_step=accepted,
        nonfinite_candidates=nonfinite_candidates,
        exception_candidates=exception_candidates,
        diagnostic="; ".join(diagnostics),
    )
