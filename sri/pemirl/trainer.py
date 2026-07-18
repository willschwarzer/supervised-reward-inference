from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from sri.pemirl.adaptation import AdaptStats, AdaptedPEMIRLPolicy
from sri.pemirl.fusion import KeyedFusionBuffer
from sri.pemirl.networks import ContextEncoder, DiagGaussianMLPPolicy, PotentialNet, RewardNet
from sri.pemirl.objectives import (
    airl_log_p_tau,
    discriminator_terms,
    imitation_likelihood_loss,
    info_losses,
)
from sri.pemirl.pretrain import MetaILPretrainer
from sri.pemirl.sampler import (
    build_meta_dataset,
    sample_task_indices,
    sample_traj_indices,
    trajectories_to_batch,
)
from sri.pemirl.trpo import trpo_step


class _LinearFeatureBaseline:
    def __init__(self, reg_coeff: float = 1e-5) -> None:
        self._coeffs: np.ndarray | None = None
        self._reg_coeff = float(reg_coeff)

    @staticmethod
    def _features(observations: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        obs = np.clip(observations, -10.0, 10.0)
        t = np.arange(len(rewards), dtype=np.float32).reshape(-1, 1) / 100.0
        return np.concatenate([obs, obs**2, t, t**2, t**3, np.ones((len(rewards), 1), dtype=np.float32)], axis=1)

    def fit(self, paths: list[dict[str, np.ndarray]]) -> None:
        featmat = np.concatenate(
            [self._features(path["observations"], path["rewards"]) for path in paths], axis=0
        )
        returns = np.concatenate([path["returns"] for path in paths], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns),
                rcond=None,
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10.0

    def predict(self, observations: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        if self._coeffs is None:
            return np.zeros(len(rewards), dtype=np.float32)
        return self._features(observations, rewards).dot(self._coeffs).astype(np.float32)


def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    running = 0.0
    for i in range(len(x) - 1, -1, -1):
        running = float(x[i]) + discount * running
        y[i] = running
    return y.astype(np.float32)


def _center_advantages(adv: np.ndarray) -> np.ndarray:
    return ((adv - np.mean(adv)) / (np.std(adv) + 1e-8)).astype(np.float32)


@dataclass
class PEMIRLConfig:
    latent_dim: int = 3
    meta_batch_size: int = 50
    task_batch_size: int = 16
    discrim_updates: int = 20
    pretrain_epochs: int = 1000
    info_coeff: float = 0.1
    imitation_coeff: float = 0.01
    entropy_weight: float = 1.0
    discount: float = 0.99
    trpo_step_size: float = 0.01
    fusion_buffer_size: int = 100
    fusion_subsample_ratio: float = 0.5
    policy_hidden_sizes: tuple[int, int] = (64, 64)
    context_hidden_sizes: tuple[int, int] = (128, 128)
    reward_hidden_size: int = 32
    value_hidden_size: int = 32
    meta_train_iters: int = 200
    adapt_iters: int = 200
    seed: int = 0
    lr: float = 1e-3
    kl_weight: float = 0.1
    use_fusion: bool = True
    trpo_debug_nonfinite: bool = False
    trpo_reject_nonfinite_steps: bool = True
    meta_train_rollout_budget: int = -1
    on_policy_training: bool = True
    on_policy_adaptation: bool = True
    rollout_horizon: int = -1
    adapt_rollouts_per_iter: int = 16
    adapt_deterministic_policy: bool = False


@dataclass
class PEMIRLTrainMetrics:
    cent_loss: list[float] = field(default_factory=list)
    info_loss: list[float] = field(default_factory=list)
    info_surr_loss: list[float] = field(default_factory=list)
    imitation_loss: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    trpo_kl: list[float] = field(default_factory=list)
    trpo_accepted: list[float] = field(default_factory=list)
    avg_return: list[float] = field(default_factory=list)


class PEMIRLModel:
    def __init__(self, obs_dim: int, act_dim: int, horizon: int, config: PEMIRLConfig, device: str | torch.device = "cpu") -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.horizon = int(horizon)
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(config.seed)

        self.policy = DiagGaussianMLPPolicy(
            input_dim=self.obs_dim + config.latent_dim,
            output_dim=self.act_dim,
            hidden_sizes=config.policy_hidden_sizes,
            state_dependent_std=False,
        ).to(self.device)
        self.context_encoder = ContextEncoder(
            input_dim=(self.obs_dim + self.act_dim) * self.horizon,
            output_dim=config.latent_dim,
            hidden_sizes=config.context_hidden_sizes,
            state_dependent_std=False,
        ).to(self.device)
        self.reward_net = RewardNet(
            state_dim=self.obs_dim,
            latent_dim=config.latent_dim,
            hidden_size=config.reward_hidden_size,
            layers=2,
        ).to(self.device)
        self.value_net = PotentialNet(
            state_dim=self.obs_dim,
            latent_dim=config.latent_dim,
            hidden_size=config.value_hidden_size,
            layers=2,
        ).to(self.device)
        self._policy_params = list(self.policy.parameters())
        self._context_params = list(self.context_encoder.parameters())
        self._reward_params = list(self.reward_net.parameters())
        self._value_params = list(self.value_net.parameters())

        self.policy_optimizer = torch.optim.Adam(self._policy_params, lr=config.lr)
        self.context_optimizer = torch.optim.Adam(self._context_params, lr=config.lr)
        self.reward_optimizer = torch.optim.Adam(self._reward_params, lr=config.lr)
        self.value_optimizer = torch.optim.Adam(self._value_params, lr=config.lr)

        self.fusion = None
        if config.use_fusion:
            self.fusion = KeyedFusionBuffer(
                buffer_size=config.fusion_buffer_size,
                subsample_ratio=config.fusion_subsample_ratio,
                seed=config.seed,
            )

    @classmethod
    def from_rollouts(cls, rollouts_by_task: list[list], config: PEMIRLConfig, device: str | torch.device = "cpu") -> "PEMIRLModel":
        first = None
        for task in rollouts_by_task:
            if task:
                first = task[0]
                break
        if first is None:
            raise ValueError("No trajectories available to initialize PEMIRLModel")
        obs = np.asarray(first.obs)
        acts = np.asarray(first.acts)
        t = min(len(acts), len(obs) - 1)
        if t <= 0:
            raise ValueError("Invalid trajectory lengths for PEMIRL")
        return cls(obs_dim=obs.shape[-1], act_dim=acts.shape[-1], horizon=t, config=config, device=device)

    def _prepare_flat_pretrain_data(self, dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = torch.cat([d.flat_traj for d in dataset], dim=0)
        obs = torch.cat([d.obs for d in dataset], dim=0)
        acts = torch.cat([d.acts for d in dataset], dim=0)
        return flat, obs, acts

    def _forward_discriminator(self, obs_t, obs_tp1, acts_t, log_q_tau, latents):
        reward_t = self.reward_net(obs_t, latents)
        value_t = self.value_net(obs_t, latents)
        value_tp1 = self.value_net(obs_tp1, latents)
        log_p_tau = airl_log_p_tau(reward_t, value_t, value_tp1, self.config.discount)
        return log_p_tau, reward_t

    @staticmethod
    def _set_param_grads(params, grads) -> None:
        for p, g in zip(params, grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach()

    def train_meta(
        self,
        rollouts_by_task: list[list],
        task_goals: np.ndarray | None = None,
        rollout_collector: Any | None = None,
    ) -> PEMIRLTrainMetrics:
        if self.config.on_policy_training:
            if rollout_collector is None or task_goals is None:
                raise ValueError(
                    "On-policy PEMIRL training requires both rollout_collector and task_goals."
                )
            return self._train_meta_on_policy(
                rollouts_by_task=rollouts_by_task,
                task_goals=np.asarray(task_goals, dtype=np.float32),
                rollout_collector=rollout_collector,
            )
        return self._train_meta_offline(rollouts_by_task)

    def _paths_to_batch_tensors(self, paths: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        if len(paths) == 0:
            raise ValueError("No paths were collected for on-policy PEMIRL.")
        obs_list, next_obs_list, act_list, logp_list, rew_list = [], [], [], [], []
        horizon = self.horizon if self.config.rollout_horizon <= 0 else int(self.config.rollout_horizon)
        for path in paths:
            obs = np.asarray(path["observations"], dtype=np.float32)
            next_obs = np.asarray(path["next_observations"], dtype=np.float32)
            acts = np.asarray(path["actions"], dtype=np.float32)
            log_probs = np.asarray(path["log_probs"], dtype=np.float32).reshape(-1, 1)
            env_rewards = np.asarray(path.get("env_rewards", np.zeros(len(acts), dtype=np.float32)), dtype=np.float32)

            t = min(len(obs), len(next_obs), len(acts), len(log_probs), horizon)
            if t <= 0:
                continue
            obs = obs[:t]
            next_obs = next_obs[:t]
            acts = acts[:t]
            log_probs = log_probs[:t]
            env_rewards = env_rewards[:t]

            if t < horizon:
                pad_n = horizon - t
                obs = np.concatenate([obs, np.repeat(obs[-1:], pad_n, axis=0)], axis=0)
                next_obs = np.concatenate(
                    [next_obs, np.repeat(next_obs[-1:], pad_n, axis=0)], axis=0
                )
                acts = np.concatenate([acts, np.zeros((pad_n, self.act_dim), dtype=np.float32)], axis=0)
                log_probs = np.concatenate([log_probs, np.zeros((pad_n, 1), dtype=np.float32)], axis=0)
                env_rewards = np.concatenate([env_rewards, np.zeros((pad_n,), dtype=np.float32)], axis=0)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(acts)
            logp_list.append(log_probs)
            rew_list.append(env_rewards)

        if len(obs_list) == 0:
            raise ValueError("Collected paths were empty after horizon truncation.")

        return {
            "obs": torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(
                np.stack(next_obs_list), dtype=torch.float32, device=self.device
            ),
            "acts": torch.as_tensor(np.stack(act_list), dtype=torch.float32, device=self.device),
            "log_probs": torch.as_tensor(np.stack(logp_list), dtype=torch.float32, device=self.device),
            "env_rewards": torch.as_tensor(np.stack(rew_list), dtype=torch.float32, device=self.device),
        }

    def _prepare_trpo_batch(
        self, trpo_paths: list[dict[str, np.ndarray]], trpo_latents: list[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(trpo_paths) == 0:
            raise ValueError("No trajectories available for TRPO update.")
        assert len(trpo_paths) == len(trpo_latents)

        baseline = _LinearFeatureBaseline()
        for path in trpo_paths:
            path["returns"] = _discount_cumsum(path["rewards"], self.config.discount)
        baseline.fit(trpo_paths)

        obs_ctx_list, act_list, adv_list = [], [], []
        for path, z_np in zip(trpo_paths, trpo_latents):
            path_baseline = baseline.predict(path["observations"], path["rewards"])
            path_baseline = np.append(path_baseline, 0.0).astype(np.float32)
            deltas = (
                path["rewards"]
                + self.config.discount * path_baseline[1:]
                - path_baseline[:-1]
            )
            advantages = _discount_cumsum(deltas, self.config.discount)
            z_tile = np.repeat(z_np.reshape(1, -1), path["observations"].shape[0], axis=0)
            obs_ctx = np.concatenate([path["observations"], z_tile], axis=-1)
            obs_ctx_list.append(obs_ctx.astype(np.float32))
            act_list.append(path["actions"].astype(np.float32))
            adv_list.append(advantages.astype(np.float32))

        advantages = _center_advantages(np.concatenate(adv_list, axis=0))
        obs_ctx = np.concatenate(obs_ctx_list, axis=0)
        acts = np.concatenate(act_list, axis=0)
        return (
            torch.as_tensor(obs_ctx, dtype=torch.float32, device=self.device),
            torch.as_tensor(acts, dtype=torch.float32, device=self.device),
            torch.as_tensor(advantages.reshape(-1, 1), dtype=torch.float32, device=self.device),
        )

    def _train_meta_on_policy(
        self,
        rollouts_by_task: list[list],
        task_goals: np.ndarray,
        rollout_collector: Any,
    ) -> PEMIRLTrainMetrics:
        dataset = build_meta_dataset(rollouts_by_task, device=self.device)
        pretrainer = MetaILPretrainer(
            policy=self.policy,
            context_encoder=self.context_encoder,
            latent_dim=self.config.latent_dim,
            max_path_length=self.horizon,
            kl_weight=self.config.kl_weight,
            lr=self.config.lr,
            device=self.device,
        )
        flat, obs, acts = self._prepare_flat_pretrain_data(dataset)
        if self.config.pretrain_epochs > 0:
            pretrainer.run(flat, obs, acts, epochs=self.config.pretrain_epochs, batch_size=400)

        metrics = PEMIRLTrainMetrics()
        warm_up = True
        warm_up_step = max(1, len(dataset) // max(1, self.config.meta_batch_size))
        warm_up_idx = 0

        for itr in range(self.config.meta_train_iters):
            task_idxs, warm_up_idx = sample_task_indices(
                num_tasks=len(dataset),
                meta_batch_size=min(self.config.meta_batch_size, len(dataset)),
                warm_up=warm_up,
                warm_up_idx=warm_up_idx,
                rng=self.rng,
            )
            if itr >= warm_up_step:
                warm_up = False

            task_payloads: list[dict[str, Any]] = []
            for task_idx in task_idxs.tolist():
                task = dataset[int(task_idx)]
                idxs = sample_traj_indices(task.obs.shape[0], self.config.task_batch_size, self.rng)

                exp_obs = task.obs[idxs]
                exp_next_obs = task.next_obs[idxs]
                exp_acts = task.acts[idxs]

                one_shot_flat = task.flat_traj[idxs[0]].unsqueeze(0)
                z, z_info = self.context_encoder.encode(one_shot_flat, deterministic=False)
                z = z.squeeze(0)
                z_np = z.detach().cpu().numpy().astype(np.float32)

                current_paths = rollout_collector(
                    policy=self.policy,
                    context=z_np,
                    goal=np.asarray(task_goals[int(task_idx)], dtype=np.float32),
                    num_rollouts=int(self.config.task_batch_size),
                    horizon=int(self.horizon if self.config.rollout_horizon <= 0 else self.config.rollout_horizon),
                )
                replay_paths = list(current_paths)

                if self.fusion is not None:
                    old_paths = self.fusion.sample_paths([int(task_idx)], n=len(current_paths))
                    self.fusion.add_paths({int(task_idx): current_paths}, subsample=True)
                    if old_paths is not None:
                        replay_paths.extend(old_paths[int(task_idx)])

                neg = self._paths_to_batch_tensors(replay_paths)
                task_payloads.append(
                    {
                        "task_idx": int(task_idx),
                        "z": z,
                        "z_info": z_info,
                        "z_np": z_np,
                        "exp_obs": exp_obs,
                        "exp_next_obs": exp_next_obs,
                        "exp_acts": exp_acts,
                        "one_obs": task.obs[idxs[0]],
                        "one_acts": task.acts[idxs[0]],
                        "neg_obs": neg["obs"],
                        "neg_next_obs": neg["next_obs"],
                        "neg_acts": neg["acts"],
                        "neg_log_probs": neg["log_probs"],
                        "current_paths": current_paths,
                    }
                )

            cent_losses, info_losses_hist, info_surr_hist = [], [], []
            imit_hist, total_hist = [], []

            for _ in range(self.config.discrim_updates):
                per_task_log_p, per_task_lq = [], []
                per_task_labels, per_task_log_qm = [], []
                per_task_traj_return = []
                imitation_log_probs = []

                for payload in task_payloads:
                    exp_obs = payload["exp_obs"].reshape(-1, self.obs_dim)
                    exp_next_obs = payload["exp_next_obs"].reshape(-1, self.obs_dim)
                    exp_acts = payload["exp_acts"].reshape(-1, self.act_dim)
                    neg_obs = payload["neg_obs"].reshape(-1, self.obs_dim)
                    neg_next_obs = payload["neg_next_obs"].reshape(-1, self.obs_dim)
                    neg_acts = payload["neg_acts"].reshape(-1, self.act_dim)
                    neg_log_q = payload["neg_log_probs"].reshape(-1, 1).detach()

                    z = payload["z"]
                    z_info = payload["z_info"]
                    z_exp = z.unsqueeze(0).expand(exp_obs.shape[0], -1)
                    z_neg = z.unsqueeze(0).expand(neg_obs.shape[0], -1)

                    exp_in = torch.cat([exp_obs, z_exp], dim=-1)
                    exp_log_q = self.policy.log_prob(exp_in, exp_acts).detach()

                    neg_log_p, neg_rew = self._forward_discriminator(
                        neg_obs, neg_next_obs, neg_acts, neg_log_q, z_neg
                    )
                    exp_log_p, exp_rew = self._forward_discriminator(
                        exp_obs, exp_next_obs, exp_acts, exp_log_q, z_exp
                    )

                    all_log_p = torch.cat([neg_log_p, exp_log_p], dim=0)
                    all_log_q = torch.cat([neg_log_q, exp_log_q], dim=0)
                    all_labels = torch.cat(
                        [
                            torch.zeros((neg_obs.shape[0], 1), device=self.device),
                            torch.ones((exp_obs.shape[0], 1), device=self.device),
                        ],
                        dim=0,
                    )
                    q_dist = torch.distributions.Normal(
                        z_info.mean, z_info.log_std.exp()
                    )
                    log_qm = q_dist.log_prob(z.unsqueeze(0)).sum(dim=-1, keepdim=True)
                    log_qm = log_qm.expand(all_log_p.shape[0], 1)
                    traj_return = torch.cat([neg_rew, exp_rew], dim=0)

                    one_obs = payload["one_obs"].reshape(-1, self.obs_dim)
                    one_acts = payload["one_acts"].reshape(-1, self.act_dim)
                    one_z = z.unsqueeze(0).expand(one_obs.shape[0], -1)
                    one_in = torch.cat([one_obs, one_z], dim=-1)
                    imitation_log_probs.append(self.policy.log_prob(one_in, one_acts))

                    per_task_log_p.append(all_log_p)
                    per_task_lq.append(all_log_q)
                    per_task_labels.append(all_labels)
                    per_task_log_qm.append(log_qm)
                    per_task_traj_return.append(traj_return)

                log_p_batch = torch.stack(per_task_log_p, dim=0)
                log_q_batch = torch.stack(per_task_lq, dim=0)
                labels = torch.stack(per_task_labels, dim=0)
                log_qm = torch.stack(per_task_log_qm, dim=0)
                sampled_ret = torch.stack(per_task_traj_return, dim=0)

                cent_loss, _, _ = discriminator_terms(log_p_batch, log_q_batch, labels)
                info_loss, info_surr = info_losses(log_qm, sampled_ret, labels)
                imitation_loss = imitation_likelihood_loss(torch.cat(imitation_log_probs, dim=0))

                context_loss = (
                    cent_loss
                    + self.config.info_coeff * info_loss
                    + self.config.imitation_coeff * imitation_loss
                )
                reward_loss = cent_loss + self.config.info_coeff * info_surr
                value_loss = cent_loss
                policy_loss = self.config.imitation_coeff * imitation_loss
                total_loss = (
                    cent_loss
                    + self.config.info_coeff * info_loss
                    + self.config.info_coeff * info_surr
                    + self.config.imitation_coeff * imitation_loss
                )

                self.policy_optimizer.zero_grad(set_to_none=True)
                self.context_optimizer.zero_grad(set_to_none=True)
                self.reward_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)

                context_grads = torch.autograd.grad(
                    context_loss, self._context_params, retain_graph=True, allow_unused=True
                )
                reward_grads = torch.autograd.grad(
                    reward_loss, self._reward_params, retain_graph=True, allow_unused=True
                )
                value_grads = torch.autograd.grad(
                    value_loss, self._value_params, retain_graph=True, allow_unused=True
                )
                policy_grads = torch.autograd.grad(
                    policy_loss, self._policy_params, retain_graph=False, allow_unused=True
                )

                self._set_param_grads(self._context_params, context_grads)
                self._set_param_grads(self._reward_params, reward_grads)
                self._set_param_grads(self._value_params, value_grads)
                self._set_param_grads(self._policy_params, policy_grads)
                self.context_optimizer.step()
                self.reward_optimizer.step()
                self.value_optimizer.step()
                self.policy_optimizer.step()

                cent_losses.append(float(cent_loss.detach().cpu()))
                info_losses_hist.append(float(info_loss.detach().cpu()))
                info_surr_hist.append(float(info_surr.detach().cpu()))
                imit_hist.append(float(imitation_loss.detach().cpu()))
                total_hist.append(float(total_loss.detach().cpu()))

            trpo_paths: list[dict[str, np.ndarray]] = []
            trpo_latents: list[np.ndarray] = []
            env_returns: list[float] = []
            with torch.no_grad():
                for payload in task_payloads:
                    z = payload["z"]
                    for path in payload["current_paths"]:
                        obs_np = np.asarray(path["observations"], dtype=np.float32)
                        next_obs_np = np.asarray(path["next_observations"], dtype=np.float32)
                        act_np = np.asarray(path["actions"], dtype=np.float32)
                        env_rew_np = np.asarray(path.get("env_rewards", np.zeros(self.horizon)), dtype=np.float32)
                        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
                        next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=self.device)
                        act_t = torch.as_tensor(act_np, dtype=torch.float32, device=self.device)
                        z_tile = z.unsqueeze(0).expand(obs_t.shape[0], -1)
                        log_q_dummy = torch.zeros((obs_t.shape[0], 1), dtype=torch.float32, device=self.device)
                        shaped, _ = self._forward_discriminator(obs_t, next_obs_t, act_t, log_q_dummy, z_tile)
                        shaped_np = shaped.squeeze(-1).detach().cpu().numpy().astype(np.float32)
                        trpo_paths.append(
                            {
                                "observations": obs_np,
                                "actions": act_np,
                                "rewards": shaped_np,
                            }
                        )
                        trpo_latents.append(payload["z_np"])
                        env_returns.append(float(np.sum(env_rew_np)))

            obs_ctx, acts_t, adv_t = self._prepare_trpo_batch(trpo_paths, trpo_latents)
            old_policy = self.policy.clone().to(self.device)
            trpo_stats = trpo_step(
                self.policy,
                old_policy,
                obs_ctx,
                acts_t,
                adv_t,
                max_kl=self.config.trpo_step_size,
                ent_weight=self.config.entropy_weight,
                debug_nonfinite=self.config.trpo_debug_nonfinite,
                reject_nonfinite_steps=self.config.trpo_reject_nonfinite_steps,
                debug_prefix=f"itr={itr}",
            )
            if self.config.trpo_debug_nonfinite and (
                trpo_stats.nonfinite_candidates > 0
                or trpo_stats.exception_candidates > 0
                or trpo_stats.diagnostic
            ):
                print(
                    "[PEMIRL] TRPO diagnostics "
                    f"(itr={itr}): nonfinite_candidates={trpo_stats.nonfinite_candidates}, "
                    f"exception_candidates={trpo_stats.exception_candidates}, "
                    f"accepted={trpo_stats.accepted_step}"
                )
                if trpo_stats.diagnostic:
                    print(f"[PEMIRL] TRPO diagnostic detail: {trpo_stats.diagnostic}")

            metrics.cent_loss.append(float(np.mean(cent_losses)))
            metrics.info_loss.append(float(np.mean(info_losses_hist)))
            metrics.info_surr_loss.append(float(np.mean(info_surr_hist)))
            metrics.imitation_loss.append(float(np.mean(imit_hist)))
            metrics.total_loss.append(float(np.mean(total_hist)))
            metrics.trpo_kl.append(trpo_stats.mean_kl_after)
            metrics.trpo_accepted.append(float(trpo_stats.accepted_step))
            metrics.avg_return.append(float(np.mean(env_returns)) if env_returns else 0.0)
        return metrics

    def _train_meta_offline(self, rollouts_by_task: list[list]) -> PEMIRLTrainMetrics:
        dataset = build_meta_dataset(rollouts_by_task, device=self.device)

        pretrainer = MetaILPretrainer(
            policy=self.policy,
            context_encoder=self.context_encoder,
            latent_dim=self.config.latent_dim,
            max_path_length=self.horizon,
            kl_weight=self.config.kl_weight,
            lr=self.config.lr,
            device=self.device,
        )
        flat, obs, acts = self._prepare_flat_pretrain_data(dataset)
        if self.config.pretrain_epochs > 0:
            pretrainer.run(flat, obs, acts, epochs=self.config.pretrain_epochs, batch_size=400)

        metrics = PEMIRLTrainMetrics()
        warm_up = True
        warm_up_step = max(1, len(dataset) // max(1, self.config.meta_batch_size))
        warm_up_idx = 0

        for itr in range(self.config.meta_train_iters):
            task_idxs, warm_up_idx = sample_task_indices(
                num_tasks=len(dataset),
                meta_batch_size=min(self.config.meta_batch_size, len(dataset)),
                warm_up=warm_up,
                warm_up_idx=warm_up_idx,
                rng=self.rng,
            )
            if itr >= warm_up_step:
                warm_up = False

            cent_losses = []
            info_losses_hist = []
            info_surr_hist = []
            imit_hist = []
            total_hist = []

            for _ in range(self.config.discrim_updates):
                per_task_log_p = []
                per_task_obs = []
                per_task_next_obs = []
                per_task_acts = []
                per_task_lq = []
                per_task_labels = []
                per_task_log_qm = []
                per_task_traj_return = []

                imitation_log_probs = []

                for local_task_id, task_idx in enumerate(task_idxs):
                    task = dataset[int(task_idx)]
                    idxs = sample_traj_indices(task.obs.shape[0], self.config.task_batch_size, self.rng)

                    exp_obs = task.obs[idxs].reshape(-1, self.obs_dim)
                    exp_next_obs = task.next_obs[idxs].reshape(-1, self.obs_dim)
                    exp_acts = task.acts[idxs].reshape(-1, self.act_dim)

                    # "Policy" samples: shuffle actions to simulate negatives (offline approximation).
                    neg_obs = exp_obs
                    neg_next_obs = exp_next_obs
                    neg_acts = exp_acts[torch.randperm(exp_acts.shape[0], device=self.device)]

                    # One-shot context inferred from one expert trajectory.
                    one_shot_flat = task.flat_traj[idxs[0]].unsqueeze(0)
                    z, z_info = self.context_encoder.encode(one_shot_flat, deterministic=False)
                    z = z.squeeze(0)

                    z_exp = z.unsqueeze(0).expand(exp_obs.shape[0], -1)
                    z_neg = z.unsqueeze(0).expand(neg_obs.shape[0], -1)

                    # q_tau = current policy probability
                    exp_in = torch.cat([exp_obs, z_exp], dim=-1)
                    neg_in = torch.cat([neg_obs, z_neg], dim=-1)
                    # Match TF code: log q_tau is treated as a fixed input for discriminator updates.
                    exp_log_q = self.policy.log_prob(exp_in, exp_acts).detach()
                    neg_log_q = self.policy.log_prob(neg_in, neg_acts).detach()

                    neg_log_p, neg_rew = self._forward_discriminator(neg_obs, neg_next_obs, neg_acts, neg_log_q, z_neg)
                    exp_log_p, exp_rew = self._forward_discriminator(exp_obs, exp_next_obs, exp_acts, exp_log_q, z_exp)

                    # concatenate negatives then experts to match labels convention
                    all_log_p = torch.cat([neg_log_p, exp_log_p], dim=0)
                    all_log_q = torch.cat([neg_log_q, exp_log_q], dim=0)
                    all_obs = torch.cat([neg_obs, exp_obs], dim=0)
                    all_next_obs = torch.cat([neg_next_obs, exp_next_obs], dim=0)
                    all_acts = torch.cat([neg_acts, exp_acts], dim=0)
                    all_labels = torch.cat(
                        [
                            torch.zeros((neg_obs.shape[0], 1), device=self.device),
                            torch.ones((exp_obs.shape[0], 1), device=self.device),
                        ],
                        dim=0,
                    )

                    # log q(m|tau)
                    q_dist = torch.distributions.Normal(z_info.mean, z_info.log_std.exp())
                    log_qm = q_dist.log_prob(z.unsqueeze(0)).sum(dim=-1, keepdim=True)
                    log_qm = log_qm.expand(all_log_p.shape[0], 1)

                    traj_return = torch.cat([neg_rew, exp_rew], dim=0)

                    # imitation term from one-shot expert trajectory
                    one_obs = task.obs[idxs[0]].reshape(-1, self.obs_dim)
                    one_acts = task.acts[idxs[0]].reshape(-1, self.act_dim)
                    one_z = z.unsqueeze(0).expand(one_obs.shape[0], -1)
                    one_in = torch.cat([one_obs, one_z], dim=-1)
                    imitation_log_probs.append(self.policy.log_prob(one_in, one_acts))

                    per_task_obs.append(all_obs)
                    per_task_next_obs.append(all_next_obs)
                    per_task_acts.append(all_acts)
                    per_task_log_p.append(all_log_p)
                    per_task_lq.append(all_log_q)
                    per_task_labels.append(all_labels)
                    per_task_log_qm.append(log_qm)
                    per_task_traj_return.append(traj_return)

                log_p_batch = torch.stack(per_task_log_p, dim=0)
                log_q_batch = torch.stack(per_task_lq, dim=0)
                labels = torch.stack(per_task_labels, dim=0)
                log_qm = torch.stack(per_task_log_qm, dim=0)
                sampled_ret = torch.stack(per_task_traj_return, dim=0)

                cent_loss, _, _ = discriminator_terms(log_p_batch, log_q_batch, labels)
                info_loss, info_surr = info_losses(log_qm, sampled_ret, labels)
                imitation_loss = imitation_likelihood_loss(torch.cat(imitation_log_probs, dim=0))
                context_loss = (
                    cent_loss
                    + self.config.info_coeff * info_loss
                    + self.config.imitation_coeff * imitation_loss
                )
                reward_loss = cent_loss + self.config.info_coeff * info_surr
                value_loss = cent_loss
                policy_loss = self.config.imitation_coeff * imitation_loss

                total_loss = (
                    cent_loss
                    + self.config.info_coeff * info_loss
                    + self.config.info_coeff * info_surr
                    + self.config.imitation_coeff * imitation_loss
                )
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.context_optimizer.zero_grad(set_to_none=True)
                self.reward_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)

                context_grads = torch.autograd.grad(
                    context_loss,
                    self._context_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                reward_grads = torch.autograd.grad(
                    reward_loss,
                    self._reward_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                value_grads = torch.autograd.grad(
                    value_loss,
                    self._value_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                policy_grads = torch.autograd.grad(
                    policy_loss,
                    self._policy_params,
                    retain_graph=False,
                    allow_unused=True,
                )

                self._set_param_grads(self._context_params, context_grads)
                self._set_param_grads(self._reward_params, reward_grads)
                self._set_param_grads(self._value_params, value_grads)
                self._set_param_grads(self._policy_params, policy_grads)

                self.context_optimizer.step()
                self.reward_optimizer.step()
                self.value_optimizer.step()
                self.policy_optimizer.step()

                cent_losses.append(float(cent_loss.detach().cpu()))
                info_losses_hist.append(float(info_loss.detach().cpu()))
                info_surr_hist.append(float(info_surr.detach().cpu()))
                imit_hist.append(float(imitation_loss.detach().cpu()))
                total_hist.append(float(total_loss.detach().cpu()))

            # Optional TRPO policy step with offline surrogate advantages.
            old_policy = self.policy.clone().to(self.device)
            obs_b = obs.reshape(-1, self.obs_dim)
            acts_b = acts.reshape(-1, self.act_dim)
            with torch.no_grad():
                z0, _ = self.context_encoder.encode(flat[:1], deterministic=True)
                z0 = z0.squeeze(0).expand(obs_b.shape[0], -1)
                inp = torch.cat([obs_b, z0], dim=-1)
                adv = self.policy.log_prob(inp, acts_b)
            trpo_stats = trpo_step(
                self.policy,
                old_policy,
                inp,
                acts_b,
                adv,
                max_kl=self.config.trpo_step_size,
                ent_weight=self.config.entropy_weight,
                debug_nonfinite=self.config.trpo_debug_nonfinite,
                reject_nonfinite_steps=self.config.trpo_reject_nonfinite_steps,
                debug_prefix=f"itr={itr}",
            )
            if self.config.trpo_debug_nonfinite and (
                trpo_stats.nonfinite_candidates > 0
                or trpo_stats.exception_candidates > 0
                or trpo_stats.diagnostic
            ):
                print(
                    "[PEMIRL] TRPO diagnostics "
                    f"(itr={itr}): nonfinite_candidates={trpo_stats.nonfinite_candidates}, "
                    f"exception_candidates={trpo_stats.exception_candidates}, "
                    f"accepted={trpo_stats.accepted_step}"
                )
                if trpo_stats.diagnostic:
                    print(f"[PEMIRL] TRPO diagnostic detail: {trpo_stats.diagnostic}")

            metrics.cent_loss.append(float(np.mean(cent_losses)))
            metrics.info_loss.append(float(np.mean(info_losses_hist)))
            metrics.info_surr_loss.append(float(np.mean(info_surr_hist)))
            metrics.imitation_loss.append(float(np.mean(imit_hist)))
            metrics.total_loss.append(float(np.mean(total_hist)))
            metrics.trpo_kl.append(trpo_stats.mean_kl_after)
            metrics.trpo_accepted.append(float(trpo_stats.accepted_step))
            metrics.avg_return.append(float("nan"))

            if self.fusion is not None:
                self.fusion.add_paths({int(i): rollouts_by_task[int(i)] for i in task_idxs.tolist()}, subsample=True)

        return metrics

    def adapt(
        self,
        task_rollouts: list,
        adapt_iters: int | None = None,
        goal: np.ndarray | None = None,
        rollout_collector: Any | None = None,
    ) -> tuple[AdaptedPEMIRLPolicy, AdaptStats]:
        iters = self.config.adapt_iters if adapt_iters is None else int(adapt_iters)
        batch = trajectories_to_batch(task_rollouts, device=self.device)
        with torch.no_grad():
            z, info = self.context_encoder.encode(batch.flat_traj, deterministic=True)
            context = info.mean.mean(dim=0)
        context_np = context.detach().cpu().numpy().astype(np.float32)

        policy = self.policy.clone().to(self.device)

        if self.config.on_policy_adaptation:
            if rollout_collector is None or goal is None:
                raise ValueError(
                    "On-policy PEMIRL adaptation requires rollout_collector and goal."
                )
            trpo_loss = 0.0
            for _ in range(max(1, iters)):
                paths = rollout_collector(
                    policy=policy,
                    context=context_np,
                    goal=np.asarray(goal, dtype=np.float32),
                    num_rollouts=max(1, int(self.config.adapt_rollouts_per_iter)),
                    horizon=int(self.horizon if self.config.rollout_horizon <= 0 else self.config.rollout_horizon),
                    deterministic=bool(self.config.adapt_deterministic_policy),
                )
                parsed = self._paths_to_batch_tensors(paths)
                trpo_paths: list[dict[str, np.ndarray]] = []
                trpo_latents: list[np.ndarray] = []
                with torch.no_grad():
                    for path_idx in range(parsed["obs"].shape[0]):
                        obs_t = parsed["obs"][path_idx]
                        next_obs_t = parsed["next_obs"][path_idx]
                        act_t = parsed["acts"][path_idx]
                        z_tile = context.unsqueeze(0).expand(obs_t.shape[0], -1)
                        log_q_dummy = torch.zeros(
                            (obs_t.shape[0], 1), dtype=torch.float32, device=self.device
                        )
                        shaped, _ = self._forward_discriminator(
                            obs_t, next_obs_t, act_t, log_q_dummy, z_tile
                        )
                        trpo_paths.append(
                            {
                                "observations": obs_t.detach().cpu().numpy().astype(np.float32),
                                "actions": act_t.detach().cpu().numpy().astype(np.float32),
                                "rewards": shaped.squeeze(-1).detach().cpu().numpy().astype(np.float32),
                            }
                        )
                        trpo_latents.append(context_np)
                obs_ctx, acts_t, adv_t = self._prepare_trpo_batch(trpo_paths, trpo_latents)
                old_policy = policy.clone().to(self.device)
                trpo_stats = trpo_step(
                    policy,
                    old_policy,
                    obs_ctx,
                    acts_t,
                    adv_t,
                    max_kl=self.config.trpo_step_size,
                    ent_weight=self.config.entropy_weight,
                    debug_nonfinite=self.config.trpo_debug_nonfinite,
                    reject_nonfinite_steps=self.config.trpo_reject_nonfinite_steps,
                    debug_prefix="adapt",
                )
                trpo_loss = float(trpo_stats.loss_after)

            adapted = AdaptedPEMIRLPolicy(policy=policy, context=context, device=self.device)
            return adapted, AdaptStats(
                final_loss=trpo_loss,
                context_norm=float(context.norm().detach().cpu()),
            )

        # Offline fallback (legacy approximation): supervised BC-style adaptation.
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr)
        obs = batch.obs.reshape(-1, batch.obs.shape[-1])
        acts = batch.acts.reshape(-1, batch.acts.shape[-1])
        z_tile = context.unsqueeze(0).expand(obs.shape[0], -1)
        inp = torch.cat([obs, z_tile], dim=-1)
        final_loss = 0.0
        for _ in range(max(1, iters)):
            optimizer.zero_grad(set_to_none=True)
            log_prob = policy.log_prob(inp, acts)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        adapted = AdaptedPEMIRLPolicy(policy=policy, context=context, device=self.device)
        return adapted, AdaptStats(
            final_loss=final_loss,
            context_norm=float(context.norm().detach().cpu()),
        )

    def save_checkpoint(self, path: str, metrics: PEMIRLTrainMetrics | None = None) -> None:
        payload = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "horizon": self.horizon,
            "config": asdict(self.config),
            "policy": self.policy.state_dict(),
            "context_encoder": self.context_encoder.state_dict(),
            "reward_net": self.reward_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "metrics": asdict(metrics) if metrics is not None else None,
        }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "PEMIRLModel":
        ckpt = torch.load(path, map_location=map_location)
        config = PEMIRLConfig(**ckpt["config"])
        model = cls(
            obs_dim=int(ckpt["obs_dim"]),
            act_dim=int(ckpt["act_dim"]),
            horizon=int(ckpt["horizon"]),
            config=config,
            device=map_location,
        )
        model.policy.load_state_dict(ckpt["policy"])
        model.context_encoder.load_state_dict(ckpt["context_encoder"])
        model.reward_net.load_state_dict(ckpt["reward_net"])
        model.value_net.load_state_dict(ckpt["value_net"])
        return model
