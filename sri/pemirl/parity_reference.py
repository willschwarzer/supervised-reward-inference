from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gym
import numpy as np
import torch
import torch.nn as nn


def _set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class _PolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, int] = (32, 32),
        use_tanh: bool = True,
    ) -> None:
        super().__init__()
        act_cls = nn.Tanh if use_tanh else nn.ReLU
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hid in hidden_sizes:
            lin = nn.Linear(in_dim, hid)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(act_cls())
            in_dim = hid
        out = nn.Linear(in_dim, output_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianMLPPolicyTorch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, int] = (32, 32),
        min_std: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.mean_net = _PolicyMLP(input_dim, action_dim, hidden_sizes=hidden_sizes, use_tanh=True)
        # TF GaussianMLPPolicy defaults to init_std=1.0 with exp parameterization.
        self.log_std_param = nn.Parameter(torch.zeros(action_dim))
        self.min_log_std = float(np.log(min_std))

    def dist_info(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_net(obs)
        log_std = self.log_std_param.unsqueeze(0).expand_as(mean)
        log_std = torch.clamp(log_std, min=self.min_log_std)
        return mean, log_std

    @staticmethod
    def _gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        log_norm = 0.5 * np.log(2.0 * np.pi)
        return (-0.5 * ((actions - mean) ** 2) / var - log_std - log_norm).sum(dim=-1, keepdim=True)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.dist_info(obs)
        return self._gaussian_log_prob(actions, mean, log_std)

    @staticmethod
    def entropy_from_log_std(log_std: torch.Tensor) -> torch.Tensor:
        c = 0.5 * np.log(2.0 * np.pi * np.e)
        return (log_std + c).sum(dim=-1, keepdim=True)

    def get_actions(self, observations: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            mean_t, log_std_t = self.dist_info(obs_t)
        mean = mean_t.cpu().numpy()
        log_std = log_std_t.cpu().numpy()
        rnd = np.random.normal(size=mean.shape)
        actions = rnd * np.exp(log_std) + mean
        return actions, {"mean": mean, "log_std": log_std}


class ContextEncoderTorch(GaussianMLPPolicyTorch):
    def __init__(self, input_dim: int, latent_dim: int, hidden_sizes: tuple[int, int] = (128, 128)) -> None:
        super().__init__(input_dim=input_dim, action_dim=latent_dim, hidden_sizes=hidden_sizes, min_std=1e-6)


class RewardMLPTorch(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32, layers: int = 2) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(layers):
            lin = nn.Linear(in_dim, hidden)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods.extend([lin, nn.ReLU()])
            in_dim = hidden
        out = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        mods.append(out)
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearFeatureBaselineTorch:
    def __init__(self, reg_coeff: float = 1e-5) -> None:
        self._coeffs: np.ndarray | None = None
        self._reg_coeff = float(reg_coeff)

    @staticmethod
    def _features(path: dict[str, Any]) -> np.ndarray:
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l, 1))], axis=1)

    def fit(self, paths: list[dict[str, Any]]) -> None:
        featmat = np.concatenate([self._features(path) for path in paths], axis=0)
        returns = np.concatenate([path["returns"] for path in paths], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns),
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, path: dict[str, Any]) -> np.ndarray:
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)


class RamFusionDistrCustomTorch:
    def __init__(self, buf_size: int, subsample_ratio: float = 0.5) -> None:
        self.buf_size = int(buf_size)
        self.subsample_ratio = float(subsample_ratio)
        self.buffer: dict[str, list[dict[str, Any]]] = {}

    def add_paths(self, paths: dict[int, list[dict[str, Any]]], expert_traj_batch: np.ndarray, subsample: bool = True) -> None:
        for key in paths.keys():
            expert_traj_key = str(expert_traj_batch[key])
            if expert_traj_key not in self.buffer:
                self.buffer[expert_traj_key] = []
            if subsample:
                keep_n = int(len(paths[key]) * self.subsample_ratio)
                subsample_paths = paths[key][:keep_n]
            else:
                subsample_paths = [d.copy() for d in paths[key]]
            self.buffer[expert_traj_key].extend(subsample_paths)
            overflow = len(self.buffer[expert_traj_key]) - self.buf_size
            while overflow > 0:
                n = len(self.buffer[expert_traj_key])
                probs = np.arange(n) + 1
                probs = probs / float(np.sum(probs))
                pidx = np.random.choice(np.arange(n), p=probs)
                self.buffer[expert_traj_key].pop(int(pidx))
                overflow -= 1

    def sample_paths(self, expert_traj_batch: np.ndarray, n: int) -> dict[int, list[dict[str, Any]]] | None:
        out: dict[int, list[dict[str, Any]]] = {}
        for ret_key, expert_traj in enumerate(expert_traj_batch):
            expert_traj_key = str(expert_traj)
            if expert_traj_key not in self.buffer or len(self.buffer[expert_traj_key]) == 0:
                return None
            idxs = np.random.randint(0, len(self.buffer[expert_traj_key]), size=(n,))
            out[ret_key] = [self.buffer[expert_traj_key][int(i)] for i in idxs]
        return out


def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    running = 0.0
    for i in range(len(x) - 1, -1, -1):
        running = float(x[i]) + discount * running
        y[i] = running
    return y.astype(np.float32)


def _center_advantages(adv: np.ndarray) -> np.ndarray:
    return (adv - np.mean(adv)) / (np.std(adv) + 1e-8)


def _extract_paths(paths: dict[int, list[dict[str, Any]]] | list[dict[str, Any]], keys: tuple[str, ...], T: int) -> list[np.ndarray]:
    if isinstance(paths, dict):
        ret: list[np.ndarray] = []
        for key in keys:
            ret.append(
                np.stack(
                    [
                        np.stack([traj[key].reshape(T, -1) for traj in paths[idx]], axis=0).astype(np.float32)
                        for idx in paths.keys()
                    ],
                    axis=0,
                )
            )
        return ret
    ret = []
    for key in keys:
        ret.append(np.stack([traj[key].reshape(T, -1) for traj in paths], axis=0).astype(np.float32))
    return ret


def _insert_next_state(paths: dict[int, list[dict[str, Any]]] | list[dict[str, Any]], pad_val: float = 0.0):
    if isinstance(paths, dict):
        for key in paths.keys():
            _insert_next_state(paths[key], pad_val=pad_val)
        return paths
    for path in paths:
        if "observations_next" in path:
            continue
        nobs = path["observations"][1:]
        nact = path["actions"][1:]
        nobs = np.r_[nobs, pad_val * np.expand_dims(np.ones_like(nobs[0]), axis=0)]
        nact = np.r_[nact, pad_val * np.expand_dims(np.ones_like(nact[0]), axis=0)]
        path["observations_next"] = nobs
        path["actions_next"] = nact
    return paths


def _compute_path_probs(paths: list[dict[str, Any]], insert: bool = True, insert_key: str = "a_logprobs") -> np.ndarray:
    if insert_key in paths[0]:
        return np.array([path[insert_key] for path in paths], dtype=np.float32)

    out = []
    for path in paths:
        mean = path["agent_infos"]["mean"]
        log_std = path["agent_infos"]["log_std"]
        actions = path["actions"]
        var = np.exp(2.0 * log_std)
        log_norm = 0.5 * np.log(2.0 * np.pi)
        lp = (-0.5 * ((actions - mean) ** 2) / var - log_std - log_norm).sum(axis=-1)
        out.append(lp.astype(np.float32))
    if insert:
        for path, lp in zip(paths, out):
            path[insert_key] = lp
    return np.array(out, dtype=np.float32)


def _compute_path_probs_dict(paths: dict[int, list[dict[str, Any]]], insert: bool = True, insert_key: str = "a_logprobs") -> np.ndarray:
    probs = []
    for key in paths.keys():
        probs.append(_compute_path_probs(paths[key], insert=insert, insert_key=insert_key))
    return np.array(probs, dtype=np.float32)


def _sample_batch(*args: np.ndarray, batch_size: int = 32, warm_up: bool = False, warm_up_idx: int = 0) -> list[np.ndarray]:
    if len(args[0].shape) > 3:
        n = args[0].shape[1]
    else:
        n = args[0].shape[0]
    if not warm_up:
        batch_idxs = np.random.randint(0, n, size=batch_size)
    else:
        batch_idxs = np.arange(warm_up_idx, warm_up_idx + batch_size)
    if len(args[0].shape) > 3:
        return [data[:, batch_idxs, ...] for data in args]
    return [data[batch_idxs] for data in args]


def _eval_expert_probs(
    expert_paths: list[dict[str, Any]],
    policy: GaussianMLPPolicyTorch,
    context: np.ndarray | None = None,
    insert: bool = True,
) -> np.ndarray:
    for path in expert_paths:
        if "agent_infos" in path:
            del path["agent_infos"]
        if "a_logprobs" in path:
            del path["a_logprobs"]

    for i, path in enumerate(expert_paths):
        obs = path["observations"]
        if context is not None:
            obs = np.concatenate((obs, np.tile(np.expand_dims(context[i], axis=0), [obs.shape[0], 1])), axis=-1)
        _, agent_infos = policy.get_actions(obs)
        path["agent_infos"] = agent_infos

    return _compute_path_probs(expert_paths, insert=insert)


def _unpack(data: np.ndarray, paths: dict[int, list[dict[str, Any]]] | list[dict[str, Any]]):
    if isinstance(paths, dict):
        unpacked: dict[int, list[np.ndarray]] = {}
        for i, key in enumerate(paths.keys()):
            unpacked[key] = list(data[i])
        return unpacked
    lengths = [path["observations"].shape[0] for path in paths]
    unpacked = []
    idx = 0
    for l in lengths:
        unpacked.append(data[idx : idx + l])
        idx += l
    return unpacked


@dataclass
class TorchParityConfig:
    latent_dim: int = 3
    meta_batch_size: int = 50
    batch_size: int = 16
    max_itrs: int = 20
    pretrain_epochs: int = 1000
    n_itr: int = 3000
    entropy_weight: float = 1.0
    discount: float = 0.99
    imitation_coeff: float = 0.01
    info_coeff: float = 0.1
    max_path_length: int = 100
    fusion: bool = False
    seed: int = 0
    step_size: float = 0.01
    # Reproduction-first default: mirror TF apply_gradients ordering and Adam slots.
    # Alternative "clean_sum" keeps the single aggregated Adam update behavior.
    discrim_update_mode: str = "tf_legacy"


@dataclass
class TorchParityMetrics:
    discrim_loss: list[float]
    avg_return: list[float]
    trpo_accepted: list[float]
    info_loss: list[float]
    imitation_loss: list[float]


class _TFStyleAdam:
    """Tiny TF1 Adam reimplementation for ordered (possibly duplicate) grad application."""

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.beta1_power = float(beta1)
        self.beta2_power = float(beta2)
        # Slot state is per-parameter, like TF Adam slots.
        self.state: dict[nn.Parameter, tuple[torch.Tensor, torch.Tensor]] = {}
        for p in params:
            if p not in self.state:
                m = torch.zeros_like(p, memory_format=torch.preserve_format)
                v = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p] = (m, v)

    def step(self, ordered_grads: list[tuple[nn.Parameter, torch.Tensor | None]]) -> None:
        # Matches TF Adam's bias-corrected scalar at current beta powers, then
        # applies all var updates before advancing beta powers once.
        lr_t = self.lr * np.sqrt(1.0 - self.beta2_power) / (1.0 - self.beta1_power)
        with torch.no_grad():
            for p, g in ordered_grads:
                if g is None:
                    continue
                m, v = self.state[p]
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                denom = torch.sqrt(v).add(self.epsilon)
                p.addcdiv_(m, denom, value=-lr_t)
        self.beta1_power *= self.beta1
        self.beta2_power *= self.beta2


class TorchPEMIRLParityTrainer:
    def __init__(self, experts: list[dict[str, Any]], config: TorchParityConfig, device: str = "cpu") -> None:
        if not experts:
            raise ValueError("Need at least one expert trajectory")
        self.cfg = config
        self.experts = experts
        self.device = torch.device(device)

        _set_global_seeds(config.seed)
        gym.logger.set_level(40)

        self.dO = int(experts[0]["observations"].shape[-1])
        self.dU = int(experts[0]["actions"].shape[-1])
        self.T = int(config.max_path_length)

        self.policy = GaussianMLPPolicyTorch(
            input_dim=self.dO + config.latent_dim,
            action_dim=self.dU,
            hidden_sizes=(32, 32),
        ).to(self.device)
        self.context_encoder = ContextEncoderTorch(
            input_dim=(self.dO + self.dU) * self.T,
            latent_dim=config.latent_dim,
            hidden_sizes=(128, 128),
        ).to(self.device)
        self.reward_net = RewardMLPTorch(input_dim=self.dO + config.latent_dim, hidden=32, layers=2).to(self.device)
        self.value_net = RewardMLPTorch(input_dim=self.dO + config.latent_dim, hidden=32, layers=2).to(self.device)

        self.pretrain_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.context_encoder.parameters()), lr=1e-3
        )
        self.discrim_update_mode = str(config.discrim_update_mode).strip().lower()
        if self.discrim_update_mode not in {"tf_legacy", "clean_sum"}:
            raise ValueError(
                f"Unsupported discrim_update_mode='{config.discrim_update_mode}'. "
                "Use one of: tf_legacy, clean_sum."
            )

        discrim_params = (
            list(self.policy.parameters())
            + list(self.context_encoder.parameters())
            + list(self.reward_net.parameters())
            + list(self.value_net.parameters())
        )
        self.discrim_optimizer: torch.optim.Optimizer | None = None
        self.discrim_tf_optimizer: _TFStyleAdam | None = None
        if self.discrim_update_mode == "clean_sum":
            self.discrim_optimizer = torch.optim.Adam(discrim_params, lr=1e-3)
        else:
            self.discrim_tf_optimizer = _TFStyleAdam(discrim_params, lr=1e-3)

        self.fusion = RamFusionDistrCustomTorch(100, subsample_ratio=0.5) if config.fusion else None

        # Keep an env per task in the meta-batch to mirror rllab's sampler structure.
        from inverse_rl.envs import register_custom_envs

        register_custom_envs()
        self.envs = []
        for i in range(config.meta_batch_size):
            env = gym.make("PointMazeLeft-v0")
            try:
                env.seed(config.seed + i)
            except Exception:
                pass
            self.envs.append(env)

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    @staticmethod
    def _log_normal_pdf(sample: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        log2pi = torch.log(torch.tensor(2.0 * np.pi, device=sample.device, dtype=sample.dtype))
        return torch.sum(-0.5 * (((sample - mean) ** 2.0) * torch.exp(-logvar) + logvar + log2pi), dim=1)

    def _pretrain(self, randomize_policy: bool = True) -> None:
        if self.cfg.pretrain_epochs <= 0:
            return
        flat_experts = []
        expert_obses = []
        expert_actions = []
        for traj in self.experts:
            obs = traj["observations"][: self.T]
            act = traj["actions"][: self.T]
            traj_cat = np.concatenate([obs, act], axis=1)
            flat_experts.append(np.reshape(traj_cat, [-1]))
            expert_obses.append(obs)
            expert_actions.append(act)

        flat_experts = np.asarray(flat_experts, dtype=np.float32)
        expert_obses = np.asarray(expert_obses, dtype=np.float32)
        expert_actions = np.asarray(expert_actions, dtype=np.float32)

        batch_size = 400
        num_batch = int(np.ceil(len(flat_experts) / batch_size))

        saved_policy = None
        if randomize_policy:
            saved_policy = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}

        for _ in range(self.cfg.pretrain_epochs):
            for i in range(num_batch):
                flat_traj_batch = flat_experts[i * batch_size : (i + 1) * batch_size]
                if len(flat_traj_batch) == 0:
                    continue
                obses_batch = np.array(expert_obses[i * batch_size : (i + 1) * batch_size]).reshape([-1, self.dO])
                actions_batch = np.array(expert_actions[i * batch_size : (i + 1) * batch_size]).reshape([-1, self.dU])

                flat_t = torch.as_tensor(flat_traj_batch, dtype=torch.float32, device=self.device)
                obs_t = torch.as_tensor(obses_batch, dtype=torch.float32, device=self.device)
                act_t = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)

                c_mean, c_log_std = self.context_encoder.dist_info(flat_t)
                eps = torch.randn_like(c_mean)
                z = eps * torch.exp(c_log_std) + c_mean
                z_tile = z.unsqueeze(1).repeat(1, self.T, 1).reshape(-1, self.cfg.latent_dim)

                pol_in = torch.cat([obs_t, z_tile], dim=1)
                p_mean, p_log_std = self.policy.dist_info(pol_in)
                policy_likelihood_loss = -GaussianMLPPolicyTorch._gaussian_log_prob(act_t, p_mean, p_log_std).mean()

                log_pz = self._log_normal_pdf(z, torch.zeros_like(c_mean), torch.zeros_like(c_log_std * 2.0))
                log_qz = self._log_normal_pdf(z, c_mean, c_log_std * 2.0)
                latent_loss = (log_qz - log_pz).mean()
                total = policy_likelihood_loss + 0.1 * latent_loss

                self.pretrain_optimizer.zero_grad(set_to_none=True)
                total.backward()
                self.pretrain_optimizer.step()

        if saved_policy is not None:
            self.policy.load_state_dict(saved_policy)

    def _sample_meta_expert_batch(
        self,
        expert_trajs: np.ndarray,
        expert_contexts: np.ndarray,
        warm_up: bool,
        warm_up_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            _sample_batch(
                expert_trajs,
                expert_contexts,
                batch_size=self.cfg.meta_batch_size,
                warm_up=warm_up,
                warm_up_idx=warm_up_idx,
            )
        )

    def _rollout_meta_batch(self, learner_env_goals: np.ndarray, policy_contexts: np.ndarray) -> dict[int, list[dict[str, Any]]]:
        paths: dict[int, list[dict[str, Any]]] = {}
        for i in range(self.cfg.meta_batch_size):
            env = self.envs[i]
            goal = np.asarray(learner_env_goals[i], dtype=np.float32)
            context = np.asarray(policy_contexts[i], dtype=np.float32)
            task_paths: list[dict[str, Any]] = []
            for _ in range(self.cfg.batch_size):
                obs = env.reset(reset_args=goal, policy_contexts=context)
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs = np.asarray(obs, dtype=np.float32)

                obs_buf = []
                act_buf = []
                rew_buf = []
                mean_buf = []
                log_std_buf = []
                rew_dist_buf = []
                rew_ctrl_buf = []

                for _t in range(self.T):
                    action, infos = self.policy.get_actions(np.expand_dims(obs, axis=0))
                    action = action[0]
                    next_obs, reward, done, info = env.step(action)
                    next_obs = np.asarray(next_obs, dtype=np.float32)

                    obs_buf.append(obs)
                    act_buf.append(action)
                    rew_buf.append(float(reward))
                    mean_buf.append(infos["mean"][0].astype(np.float32))
                    log_std_buf.append(infos["log_std"][0].astype(np.float32))
                    rew_dist_buf.append(float(info.get("reward_dist", 0.0)))
                    rew_ctrl_buf.append(float(info.get("reward_ctrl", 0.0)))

                    obs = next_obs
                    if done:
                        break

                # PointMaze episodes should be fixed horizon, but pad defensively.
                if len(obs_buf) < self.T:
                    pad_n = self.T - len(obs_buf)
                    obs_buf.extend([obs_buf[-1].copy() for _ in range(pad_n)])
                    act_buf.extend([np.zeros_like(act_buf[-1]) for _ in range(pad_n)])
                    rew_buf.extend([0.0 for _ in range(pad_n)])
                    mean_buf.extend([mean_buf[-1].copy() for _ in range(pad_n)])
                    log_std_buf.extend([log_std_buf[-1].copy() for _ in range(pad_n)])
                    rew_dist_buf.extend([0.0 for _ in range(pad_n)])
                    rew_ctrl_buf.extend([0.0 for _ in range(pad_n)])

                task_paths.append(
                    {
                        "observations": np.asarray(obs_buf, dtype=np.float32),
                        "actions": np.asarray(act_buf, dtype=np.float32),
                        "rewards": np.asarray(rew_buf, dtype=np.float32),
                        "agent_infos": {
                            "mean": np.asarray(mean_buf, dtype=np.float32),
                            "log_std": np.asarray(log_std_buf, dtype=np.float32),
                        },
                        "env_infos": {
                            "reward_dist": np.asarray(rew_dist_buf, dtype=np.float32),
                            "reward_ctrl": np.asarray(rew_ctrl_buf, dtype=np.float32),
                        },
                    }
                )
            paths[i] = task_paths
        return paths

    def _discrim_forward(
        self,
        expert_traj_batch_input: np.ndarray,
        obs_batch: np.ndarray,
        nobs_batch: np.ndarray,
        act_batch: np.ndarray,
        lprobs_batch: np.ndarray,
        labels: np.ndarray,
        imitation_expert_obses_input: np.ndarray,
        imitation_expert_acts_input: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mbs = self.cfg.meta_batch_size
        two_b = obs_batch.shape[1]

        expert_t = torch.as_tensor(expert_traj_batch_input, dtype=torch.float32, device=self.device)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        nobs_t = torch.as_tensor(nobs_batch, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_batch, dtype=torch.float32, device=self.device)
        lprobs_t = torch.as_tensor(lprobs_batch, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        imit_obs_t = torch.as_tensor(imitation_expert_obses_input, dtype=torch.float32, device=self.device)
        imit_act_t = torch.as_tensor(imitation_expert_acts_input, dtype=torch.float32, device=self.device)

        flat_expert = expert_t.reshape(-1, self.T * (self.dO + self.dU))
        c_mean, c_log_std = self.context_encoder.dist_info(flat_expert)
        eps = torch.randn_like(c_mean)
        z = eps * torch.exp(c_log_std) + c_mean
        z_tile = z.unsqueeze(1).repeat(1, self.T, 1)

        # One-shot imitation term uses the first trajectory per meta-task.
        imit_z_tile = z_tile.reshape(mbs, -1, self.T, self.cfg.latent_dim)[:, 0, :, :].reshape(-1, self.cfg.latent_dim)
        imit_obs = imit_obs_t.reshape(-1, self.dO)
        imit_act = imit_act_t.reshape(-1, self.dU)
        imit_in = torch.cat([imit_obs, imit_z_tile], dim=1)
        p_mean_i, p_log_std_i = self.policy.dist_info(imit_in)
        policy_likelihood_loss = -GaussianMLPPolicyTorch._gaussian_log_prob(imit_act, p_mean_i, p_log_std_i).mean()

        rew_in = torch.cat([obs_t.reshape(-1, self.dO), z_tile.reshape(-1, self.cfg.latent_dim)], dim=1)
        nrew_in = torch.cat([nobs_t.reshape(-1, self.dO), z_tile.reshape(-1, self.cfg.latent_dim)], dim=1)

        reward = self.reward_net(rew_in)
        sampled_traj_return = reward.reshape(mbs, two_b, self.T).sum(dim=-1, keepdim=True)
        v_n = self.value_net(nrew_in)
        v = self.value_net(rew_in)

        log_p_tau = reward + self.cfg.discount * v_n - v
        log_p_tau = log_p_tau.reshape(mbs, two_b, self.T, 1)
        log_q_tau = lprobs_t

        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau], dim=0), dim=0)
        cent_loss = -torch.mean(labels_t * (log_p_tau - log_pq) + (1.0 - labels_t) * (log_q_tau - log_pq))

        log_q_m_tau = GaussianMLPPolicyTorch._gaussian_log_prob(z, c_mean, c_log_std).reshape(mbs, two_b, 1)
        labels_sq = torch.squeeze(labels_t, dim=-1)
        neg_mask = 1.0 - labels_sq
        neg_mean = torch.mean(1.0 - labels_t)

        info_loss = -torch.mean(log_q_m_tau * neg_mask) / neg_mean
        baseline = torch.mean(sampled_traj_return * neg_mask, dim=1, keepdim=True) / neg_mean
        info_surr_loss = -torch.mean(neg_mask * log_q_m_tau * sampled_traj_return - neg_mask * log_q_m_tau * baseline) / neg_mean

        loss = cent_loss + self.cfg.info_coeff * info_loss
        return loss, cent_loss, info_loss, info_surr_loss, policy_likelihood_loss

    def _fit_discriminator(self, paths: dict[int, list[dict[str, Any]]], expert_traj_batch: np.ndarray) -> tuple[float, float, float]:
        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(expert_traj_batch, n=len(paths[0]))
            self.fusion.add_paths(paths, expert_traj_batch, subsample=True)
            if old_paths is not None:
                for key in paths.keys():
                    paths[key] += old_paths[key]

        _compute_path_probs_dict(paths, insert=True)
        _insert_next_state(paths)
        _insert_next_state(self.experts)

        obs, obs_next, acts, acts_next, path_probs = _extract_paths(
            paths,
            keys=("observations", "observations_next", "actions", "actions_next", "a_logprobs"),
            T=self.T,
        )
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, _expert_contexts = _extract_paths(
            self.experts,
            keys=("observations", "observations_next", "actions", "actions_next", "contexts"),
            T=self.T,
        )

        expert_trajs = np.concatenate([expert_obs, expert_acts], axis=-1)
        m_hat_expert = self.context_encoder.get_actions(expert_trajs.reshape(-1, self.T * (self.dO + self.dU)))[0]
        _eval_expert_probs(self.experts, self.policy, insert=True, context=m_hat_expert)
        expert_probs = _extract_paths(self.experts, keys=("a_logprobs",), T=self.T)[0]

        expert_traj_batch_tile = np.tile(expert_traj_batch.reshape(self.cfg.meta_batch_size, 1, self.T, -1), [1, self.cfg.batch_size, 1, 1])

        mean_loss = 0.0
        last_info = 0.0
        last_imitation = 0.0
        for _ in range(self.cfg.max_itrs):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = _sample_batch(
                obs_next,
                obs,
                acts_next,
                acts,
                path_probs,
                batch_size=self.cfg.batch_size,
            )

            if obs_batch.shape[-1] == self.dO + self.cfg.latent_dim:
                nobs_batch = nobs_batch[..., :-self.cfg.latent_dim]
                obs_batch = obs_batch[..., :-self.cfg.latent_dim]

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = _sample_batch(
                expert_obs_next,
                expert_obs,
                expert_acts_next,
                expert_acts,
                expert_probs,
                batch_size=self.cfg.meta_batch_size * self.cfg.batch_size,
            )
            if expert_obs_batch.shape[-1] == self.dO + self.cfg.latent_dim:
                nexpert_obs_batch = nexpert_obs_batch[..., :-self.cfg.latent_dim]
                expert_obs_batch = expert_obs_batch[..., :-self.cfg.latent_dim]

            labels = np.zeros((self.cfg.meta_batch_size, self.cfg.batch_size * 2, 1, 1), dtype=np.float32)
            labels[:, self.cfg.batch_size :, ...] = 1.0

            imitation_expert_obses_input = expert_traj_batch.reshape(self.cfg.meta_batch_size, 1, self.T, -1)[:, :, :, : self.dO]
            imitation_expert_acts_input = expert_traj_batch.reshape(self.cfg.meta_batch_size, 1, self.T, -1)[:, :, :, self.dO :]

            expert_traj_batch_input = np.concatenate(
                [
                    expert_traj_batch_tile,
                    np.concatenate([expert_obs_batch, expert_act_batch], axis=-1).reshape(
                        self.cfg.meta_batch_size,
                        self.cfg.batch_size,
                        self.T,
                        -1,
                    ),
                ],
                axis=1,
            )

            obs_batch = np.concatenate(
                [obs_batch, expert_obs_batch.reshape(self.cfg.meta_batch_size, self.cfg.batch_size, self.T, -1)],
                axis=1,
            )
            nobs_batch = np.concatenate(
                [nobs_batch, nexpert_obs_batch.reshape(self.cfg.meta_batch_size, self.cfg.batch_size, self.T, -1)],
                axis=1,
            )
            act_batch = np.concatenate(
                [act_batch, expert_act_batch.reshape(self.cfg.meta_batch_size, self.cfg.batch_size, self.T, -1)],
                axis=1,
            )
            nact_batch = np.concatenate(
                [nact_batch, nexpert_act_batch.reshape(self.cfg.meta_batch_size, self.cfg.batch_size, self.T, -1)],
                axis=1,
            )
            lprobs_batch = np.concatenate(
                [
                    lprobs_batch,
                    expert_lprobs_batch.reshape(self.cfg.meta_batch_size, self.cfg.batch_size, self.T, -1),
                ],
                axis=1,
            ).astype(np.float32)

            loss, cent_loss, info_loss, info_surr_loss, policy_likelihood_loss = self._discrim_forward(
                expert_traj_batch_input=expert_traj_batch_input,
                obs_batch=obs_batch,
                nobs_batch=nobs_batch,
                act_batch=act_batch,
                lprobs_batch=lprobs_batch,
                labels=labels,
                imitation_expert_obses_input=imitation_expert_obses_input,
                imitation_expert_acts_input=imitation_expert_acts_input,
            )

            reward_params = list(self.reward_net.parameters())
            value_params = list(self.value_net.parameters())
            context_params = list(self.context_encoder.parameters())
            policy_params = list(self.policy.parameters())

            cent_params = reward_params + value_params + context_params
            info_params = context_params
            info_surr_params = reward_params
            policy_imitation_params = policy_params + context_params

            cent_grads = torch.autograd.grad(cent_loss, cent_params, retain_graph=True, allow_unused=True)
            info_grads = torch.autograd.grad(
                self.cfg.info_coeff * info_loss,
                info_params,
                retain_graph=True,
                allow_unused=True,
            )
            info_surr_grads = torch.autograd.grad(
                self.cfg.info_coeff * info_surr_loss,
                info_surr_params,
                retain_graph=True,
                allow_unused=True,
            )
            policy_imitation_grads = torch.autograd.grad(
                self.cfg.imitation_coeff * policy_likelihood_loss,
                policy_imitation_params,
                retain_graph=False,
                allow_unused=True,
            )

            if self.discrim_update_mode == "clean_sum":
                assert self.discrim_optimizer is not None
                self.discrim_optimizer.zero_grad(set_to_none=True)
                grads_accum: dict[nn.Parameter, torch.Tensor] = {}
                for params, grads in (
                    (cent_params, cent_grads),
                    (info_params, info_grads),
                    (info_surr_params, info_surr_grads),
                    (policy_imitation_params, policy_imitation_grads),
                ):
                    for p, g in zip(params, grads):
                        if g is None:
                            continue
                        if p in grads_accum:
                            grads_accum[p] = grads_accum[p] + g
                        else:
                            grads_accum[p] = g
                for p in self.discrim_optimizer.param_groups[0]["params"]:
                    p.grad = grads_accum.get(p, None)
                self.discrim_optimizer.step()
            else:
                # TF-fidelity path: apply gradient groups in TF order, including
                # repeated params across groups, with shared Adam slot state.
                assert self.discrim_tf_optimizer is not None
                ordered_grads: list[tuple[nn.Parameter, torch.Tensor | None]] = []
                ordered_grads.extend(list(zip(cent_params, cent_grads)))
                ordered_grads.extend(list(zip(info_params, info_grads)))
                ordered_grads.extend(list(zip(info_surr_params, info_surr_grads)))
                ordered_grads.extend(list(zip(policy_imitation_params, policy_imitation_grads)))
                self.discrim_tf_optimizer.step(ordered_grads)

            mean_loss = float(loss.detach().cpu().item())
            last_info = float(info_loss.detach().cpu().item())
            last_imitation = float(policy_likelihood_loss.detach().cpu().item())

        return mean_loss, last_info, last_imitation

    def _eval_rewards(self, paths: dict[int, list[dict[str, Any]]], expert_traj_batch: np.ndarray) -> dict[int, list[np.ndarray]]:
        obs, acts = _extract_paths(paths, keys=("observations", "actions"), T=self.T)
        expert_tile = np.tile(expert_traj_batch.reshape(self.cfg.meta_batch_size, 1, self.T, -1), [1, acts.shape[1], 1, 1])

        with torch.no_grad():
            expert_t = torch.as_tensor(expert_tile, dtype=torch.float32, device=self.device)
            obs_t = torch.as_tensor(obs[..., :-self.cfg.latent_dim], dtype=torch.float32, device=self.device)

            c_mean, c_log_std = self.context_encoder.dist_info(expert_t.reshape(-1, self.T * (self.dO + self.dU)))
            eps = torch.randn_like(c_mean)
            z = eps * torch.exp(c_log_std) + c_mean
            z_tile = z.unsqueeze(1).repeat(1, self.T, 1).reshape(-1, self.cfg.latent_dim)

            rew_in = torch.cat([obs_t.reshape(-1, self.dO), z_tile], dim=1)
            reward = self.reward_net(rew_in).cpu().numpy()

        score = reward[:, 0].reshape(self.cfg.meta_batch_size, -1, self.T)
        unpacked = _unpack(score, paths)
        assert isinstance(unpacked, dict)
        return unpacked

    def _compute_irl(self, paths: dict[int, list[dict[str, Any]]], expert_traj_batch: np.ndarray, warm_up: bool) -> tuple[dict[int, list[dict[str, Any]]], float, float]:
        original_ret = []
        discrim_ret = []

        tot_rew = 0.0
        for key in paths.keys():
            for path in paths[key]:
                original_ret.extend(path["rewards"])
                tot_rew += float(np.sum(path["rewards"]))
                path["rewards"] *= 0.0
        original_task_avg_return = tot_rew / float(len(paths) * len(next(iter(paths.values()))))

        discrim_loss, info_loss, imitation_loss = self._fit_discriminator(paths, expert_traj_batch=expert_traj_batch)
        probs = self._eval_rewards(paths, expert_traj_batch=expert_traj_batch)

        for key in probs.keys():
            for i, path_scores in enumerate(probs[key]):
                if warm_up:
                    discrim_ret.extend(path_scores)
                else:
                    if i < self.cfg.batch_size:
                        discrim_ret.extend(path_scores)

        for key in paths.keys():
            for i, path in enumerate(paths[key]):
                path["rewards"] = path["rewards"] + probs[key][i]

        return paths, discrim_loss, original_task_avg_return, info_loss, imitation_loss

    def _process_samples(self, paths: list[dict[str, Any]]) -> dict[str, Any]:
        baseline = LinearFeatureBaselineTorch()
        for path in paths:
            path["returns"] = _discount_cumsum(path["rewards"], self.cfg.discount)

        baseline.fit(paths)

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(baseline.predict(path), 0)
            deltas = path["rewards"] + self.cfg.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = _discount_cumsum(deltas, self.cfg.discount * 1.0)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.concatenate([path["observations"] for path in paths], axis=0)
        actions = np.concatenate([path["actions"] for path in paths], axis=0)
        rewards = np.concatenate([path["rewards"] for path in paths], axis=0)
        ret = np.concatenate([path["returns"] for path in paths], axis=0)
        advantages = np.concatenate([path["advantages"] for path in paths], axis=0)
        advantages = _center_advantages(advantages)

        agent_infos = {
            "mean": np.concatenate([path["agent_infos"]["mean"] for path in paths], axis=0),
            "log_std": np.concatenate([path["agent_infos"]["log_std"] for path in paths], axis=0),
        }

        return {
            "observations": observations.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "returns": ret.astype(np.float32),
            "advantages": advantages.astype(np.float32),
            "agent_infos": {k: v.astype(np.float32) for k, v in agent_infos.items()},
            "paths": paths,
        }

    @staticmethod
    def _flat_params(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    @staticmethod
    def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n

    @staticmethod
    def _flat_grad(grads: list[torch.Tensor], model: nn.Module) -> torch.Tensor:
        out = []
        for g, p in zip(grads, model.parameters()):
            if g is None:
                out.append(torch.zeros_like(p).reshape(-1))
            else:
                out.append(g.reshape(-1))
        return torch.cat(out)

    @staticmethod
    def _conjugate_gradient(Avp, b: torch.Tensor, cg_iters: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
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

    def _optimize_policy(self, samples_data: dict[int, dict[str, Any]], expert_traj_batch: np.ndarray) -> bool:
        obs_list, action_list, adv_list = [], [], []
        old_mean_list, old_log_std_list = [], []
        for i in range(self.cfg.meta_batch_size):
            s = samples_data[i]
            obs_list.append(s["observations"][:, :-self.cfg.latent_dim])
            action_list.append(s["actions"])
            adv_list.append(s["advantages"])
            old_mean_list.append(s["agent_infos"]["mean"])
            old_log_std_list.append(s["agent_infos"]["log_std"])

        clean_obs = np.concatenate(obs_list, axis=0).astype(np.float32)
        actions = np.concatenate(action_list, axis=0).astype(np.float32)
        advantages = np.concatenate(adv_list, axis=0).astype(np.float32)
        old_mean = np.concatenate(old_mean_list, axis=0).astype(np.float32)
        old_log_std = np.concatenate(old_log_std_list, axis=0).astype(np.float32)

        n_paths = samples_data[0]["observations"].shape[0] // self.T
        expert_tiled = np.tile(np.expand_dims(expert_traj_batch, axis=1), [1, n_paths, 1, 1]).astype(np.float32)

        obs_t = torch.as_tensor(clean_obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
        old_mean_t = torch.as_tensor(old_mean, dtype=torch.float32, device=self.device)
        old_log_std_t = torch.as_tensor(old_log_std, dtype=torch.float32, device=self.device)

        ex_t = torch.as_tensor(expert_tiled, dtype=torch.float32, device=self.device)

        def _sample_policy_input() -> torch.Tensor:
            # Mirrors TF graph behavior where latent reparameterization is evaluated
            # on each optimizer function call.
            with torch.no_grad():
                c_mean, c_log_std = self.context_encoder.dist_info(ex_t.reshape(-1, self.T * (self.dO + self.dU)))
                eps = torch.randn_like(c_mean)
                z = eps * torch.exp(c_log_std) + c_mean
                z_tile = z.unsqueeze(1).repeat(1, self.T, 1).reshape(-1, self.cfg.latent_dim)
                return torch.cat([obs_t, z_tile], dim=1)

        def _surrogate_from_input(pol_in: torch.Tensor) -> torch.Tensor:
            mean, log_std = self.policy.dist_info(pol_in)
            log_prob = GaussianMLPPolicyTorch._gaussian_log_prob(act_t, mean, log_std)
            old_lp = GaussianMLPPolicyTorch._gaussian_log_prob(act_t, old_mean_t, old_log_std_t)
            ratio = torch.exp(log_prob - old_lp)
            ent = GaussianMLPPolicyTorch.entropy_from_log_std(log_std).detach()
            adv_ent = adv_t + self.cfg.entropy_weight * ent
            return -(ratio * adv_ent).mean()

        def _mean_kl_from_input(pol_in: torch.Tensor) -> torch.Tensor:
            mean, log_std = self.policy.dist_info(pol_in)
            old_var = torch.exp(2.0 * old_log_std_t)
            new_var = torch.exp(2.0 * log_std)
            kl = log_std - old_log_std_t + (old_var + (old_mean_t - mean).pow(2)) / (2.0 * new_var) - 0.5
            return kl.sum(dim=-1, keepdim=True).mean()

        def _loss_only() -> torch.Tensor:
            return _surrogate_from_input(_sample_policy_input())

        def _kl_only() -> torch.Tensor:
            return _mean_kl_from_input(_sample_policy_input())

        def _loss_and_kl() -> tuple[torch.Tensor, torch.Tensor]:
            pol_in = _sample_policy_input()
            return _surrogate_from_input(pol_in), _mean_kl_from_input(pol_in)

        loss_before = _loss_only()
        grads = torch.autograd.grad(loss_before, self.policy.parameters(), create_graph=True)
        loss_grad = self._flat_grad(list(grads), self.policy).detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            kl = _kl_only()
            g1 = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_g1 = self._flat_grad(list(g1), self.policy)
            g1v = (flat_g1 * v).sum()
            g2 = torch.autograd.grad(g1v, self.policy.parameters(), retain_graph=True)
            flat_g2 = self._flat_grad(list(g2), self.policy).detach()
            return flat_g2 + 1e-5 * v

        step_dir = self._conjugate_gradient(fvp, loss_grad, cg_iters=10)
        denom = torch.dot(step_dir, fvp(step_dir)) + 1e-8
        initial_step_size = torch.sqrt(torch.tensor(2.0 * self.cfg.step_size, device=self.device) / denom)
        full_step = initial_step_size * step_dir

        old_params = self._flat_params(self.policy)
        loss_before_v = float(loss_before.detach().cpu())

        accepted = False
        for ratio in 0.8 ** np.arange(15):
            cur_step = float(ratio) * full_step
            cur_param = old_params - cur_step
            self._set_flat_params(self.policy, cur_param)
            loss_new, kl_new = _loss_and_kl()
            if float(loss_new.detach().cpu()) < loss_before_v and float(kl_new.detach().cpu()) <= self.cfg.step_size:
                accepted = True
                break

        if not accepted:
            self._set_flat_params(self.policy, old_params)
        return accepted

    def train(self) -> TorchParityMetrics:
        self._pretrain(randomize_policy=True)

        expert_obs, expert_acts, expert_contexts = _extract_paths(
            self.experts,
            keys=("observations", "actions", "contexts"),
            T=self.T,
        )
        expert_trajs = np.concatenate((expert_obs, expert_acts), axis=-1)

        warm_up = True
        warm_up_step = int(len(self.experts) / self.cfg.meta_batch_size)
        warm_up_idx = 0

        discrim_loss_hist: list[float] = []
        avg_return_hist: list[float] = []
        trpo_accepted_hist: list[float] = []
        info_loss_hist: list[float] = []
        imitation_loss_hist: list[float] = []

        for itr in range(self.cfg.n_itr):
            expert_traj_batch, m_batch = self._sample_meta_expert_batch(
                expert_trajs=expert_trajs,
                expert_contexts=expert_contexts,
                warm_up=warm_up,
                warm_up_idx=warm_up_idx,
            )
            warm_up_idx = (self.cfg.meta_batch_size + warm_up_idx) % len(self.experts)
            if itr >= warm_up_step:
                warm_up = False

            flat_expert_batch = expert_traj_batch.reshape(-1, self.T * (self.dO + self.dU))
            m_hat_batch = self.context_encoder.get_actions(flat_expert_batch)[0]

            learner_env_goals = np.asarray(m_batch[:, 0, :], dtype=np.float32)
            paths = self._rollout_meta_batch(learner_env_goals, m_hat_batch)

            paths, discrim_loss, avg_ret, info_loss, imitation_loss = self._compute_irl(
                paths, expert_traj_batch, warm_up=warm_up
            )
            discrim_loss_hist.append(float(discrim_loss))
            avg_return_hist.append(float(avg_ret))
            info_loss_hist.append(float(info_loss))
            imitation_loss_hist.append(float(imitation_loss))

            samples_data: dict[int, dict[str, Any]] = {}
            for key in paths.keys():
                samples_data[key] = self._process_samples(paths[key])

            accepted = self._optimize_policy(samples_data, expert_traj_batch)
            trpo_accepted_hist.append(float(accepted))

        return TorchParityMetrics(
            discrim_loss=discrim_loss_hist,
            avg_return=avg_return_hist,
            trpo_accepted=trpo_accepted_hist,
            info_loss=info_loss_hist,
            imitation_loss=imitation_loss_hist,
        )
