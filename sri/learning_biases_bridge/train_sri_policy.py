import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from sri.learning_biases_bridge.common import (
    build_rollout_demonstrations,
    build_policy_demonstrations,
    flatten_rewards,
    grid_shape_from_split,
    load_manifest,
    load_split,
    save_json,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model_kwargs(
    height: int,
    width: int,
    feature_dim: int,
    demonstration_horizon: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    rep_dim = height * width
    return {
        "demonstration_rep_dim": rep_dim,
        "state_rep_dim": rep_dim,
        "internal_tst_dim": args.internal_tst_dim,
        "state_hidden_size": args.state_hidden_size,
        "final_hidden_size": args.final_hidden_size,
        "demonstration_hidden_size": args.demonstration_hidden_size,
        "obs_size": 1,
        "dem_obs_size": feature_dim,
        "horizon": demonstration_horizon,
        "num_demonstration_layers": args.num_demonstration_layers,
        "num_state_layers": args.num_state_layers,
        "dem_encoder_type": args.dem_encoder_type,
        "mlp": False,
        "output_type": "goal",
        "transformer_nhead": args.transformer_nhead,
    }


def _make_flat_mlp(input_dim: int, output_dim: int, hidden_size: int, num_hidden_layers: int):
    import torch.nn as nn

    layers = [nn.Flatten(start_dim=1)]
    in_dim = input_dim
    for _ in range(max(num_hidden_layers, 0)):
        layers.extend([nn.Linear(in_dim, hidden_size), nn.LeakyReLU()])
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def _make_policy_grid_model(
    model_type: str,
    height: int,
    width: int,
    in_channels: int,
    base_channels: int,
):
    from sri.learning_biases_bridge.cnn_models import (
        PolicyCNNRegressor,
        PolicyUNetRegressor,
    )

    model_type = str(model_type).strip().lower()
    if model_type == "cnn":
        return PolicyCNNRegressor(
            height=height,
            width=width,
            in_channels=in_channels,
            base_channels=base_channels,
        )
    if model_type == "unet":
        return PolicyUNetRegressor(
            height=height,
            width=width,
            in_channels=in_channels,
            base_channels=base_channels,
        )
    raise ValueError(f"Unsupported grid model type: {model_type}")


def _predict(model, device, demos: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(demos), batch_size):
            batch = torch.from_numpy(demos[start : start + batch_size]).to(device)
            pred = model(batch)
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def _get_resolved_transformer_nhead(model, dem_encoder_type: str):
    if dem_encoder_type == "transformer":
        return getattr(getattr(model, "demonstration_encoder", None), "nhead", None)
    if dem_encoder_type == "tst":
        dem = getattr(model, "demonstration_encoder", None)
        tr = getattr(dem, "demonstration_transformer", None)
        return getattr(tr, "nhead", None)
    return None


def _parse_wandb_tags(tags: str) -> List[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


def _resolve_regularization(args: argparse.Namespace) -> Tuple[str, float]:
    reg_type = str(args.regularization_type).strip().lower()
    reg_lambda = float(args.regularization_lambda)

    if reg_type not in {"none", "l1", "l2"}:
        raise ValueError(f"Unsupported regularization type: {reg_type}")
    if reg_lambda < 0.0:
        raise ValueError("regularization-lambda must be >= 0")

    # Backward compatibility for existing runs that pass --weight-decay.
    if reg_type == "none" and float(args.weight_decay) > 0.0:
        reg_type = "l2"
        reg_lambda = float(args.weight_decay)

    if reg_type != "none" and reg_lambda <= 0.0:
        raise ValueError(
            "regularization-lambda must be > 0 when regularization-type is l1/l2"
        )

    if reg_type == "none":
        reg_lambda = 0.0

    return reg_type, reg_lambda


def _regularization_penalty(model, reg_type: str):
    if reg_type == "none":
        return None

    penalty = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if reg_type == "l1":
            term = param.abs().sum()
        else:
            term = param.pow(2).sum()
        penalty = term if penalty is None else penalty + term

    if penalty is None:
        raise ValueError("Model has no trainable parameters for regularization")
    return penalty


_POLICY_ACTION_DELTAS = np.asarray(
    [
        [0, -1],  # north
        [0, 1],   # south
        [1, 0],   # east
        [-1, 0],  # west
        [0, 0],   # stay
    ],
    dtype=np.int32,
)
_ACTION_TO_INDEX = {tuple(delta.tolist()): idx for idx, delta in enumerate(_POLICY_ACTION_DELTAS)}


def _transform_delta(dx: int, dy: int, rot_k: int, flip_lr: bool) -> Tuple[int, int]:
    k = int(rot_k) % 4
    if k == 1:
        dx, dy = dy, -dx
    elif k == 2:
        dx, dy = -dx, -dy
    elif k == 3:
        dx, dy = -dy, dx
    if flip_lr:
        dx = -dx
    return int(dx), int(dy)


def _policy_action_permutation(rot_k: int, flip_lr: bool) -> List[int]:
    perm = []
    for dx, dy in _POLICY_ACTION_DELTAS:
        new_delta = _transform_delta(int(dx), int(dy), rot_k, flip_lr)
        if new_delta not in _ACTION_TO_INDEX:
            raise ValueError(f"Invalid transformed action delta: {new_delta}")
        perm.append(int(_ACTION_TO_INDEX[new_delta]))
    return perm


def _transform_start_states(starts: np.ndarray, height: int, width: int, rot_k: int, flip_lr: bool) -> np.ndarray:
    x = starts[:, 0].astype(np.int32, copy=False)
    y = starts[:, 1].astype(np.int32, copy=False)
    k = int(rot_k) % 4

    if k == 0:
        tx, ty = x, y
        out_h, out_w = height, width
    elif k == 1:
        tx, ty = y, (width - 1) - x
        out_h, out_w = width, height
    elif k == 2:
        tx, ty = (width - 1) - x, (height - 1) - y
        out_h, out_w = height, width
    else:
        tx, ty = (height - 1) - y, x
        out_h, out_w = width, height

    if flip_lr:
        tx = (out_w - 1) - tx

    out = np.stack([tx, ty], axis=1).astype(np.int32, copy=False)
    if np.any(out[:, 0] < 0) or np.any(out[:, 0] >= out_w) or np.any(out[:, 1] < 0) or np.any(out[:, 1] >= out_h):
        raise ValueError("Transformed start states left grid bounds")
    return out


def _transform_spatial(arr: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(arr, k=int(rot_k) % 4, axes=(1, 2))
    if flip_lr:
        out = np.flip(out, axis=2)
    return out


def _augment_training_split(split: Dict[str, np.ndarray], augmentation: str) -> Dict[str, np.ndarray]:
    mode = str(augmentation).strip().lower()
    if mode == "none":
        return split
    if mode != "d4":
        raise ValueError(f"Unsupported train augmentation: {mode}")

    walls = split["walls"]
    rewards = split["rewards"]
    starts = split["start_states"]
    policies = split["policies"]

    _, h, w = walls.shape
    if h != w:
        raise ValueError(f"D4 augmentation requires square grids, got H={h}, W={w}")

    transforms = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]

    walls_aug = []
    rewards_aug = []
    starts_aug = []
    policies_aug = []

    for rot_k, flip_lr in transforms:
        tw = _transform_spatial(walls, rot_k, flip_lr)
        tr = _transform_spatial(rewards, rot_k, flip_lr)
        ts = _transform_start_states(starts, h, w, rot_k, flip_lr)

        tp = _transform_spatial(policies, rot_k, flip_lr)
        perm_old_to_new = _policy_action_permutation(rot_k, flip_lr)
        tp_remap = np.empty_like(tp)
        for old_idx, new_idx in enumerate(perm_old_to_new):
            tp_remap[..., new_idx] = tp[..., old_idx]

        walls_aug.append(tw.astype(np.float32, copy=False))
        rewards_aug.append(tr.astype(np.float32, copy=False))
        starts_aug.append(ts.astype(np.int32, copy=False))
        policies_aug.append(tp_remap.astype(np.float32, copy=False))

    augmented = dict(split)
    augmented["walls"] = np.concatenate(walls_aug, axis=0)
    augmented["rewards"] = np.concatenate(rewards_aug, axis=0)
    augmented["start_states"] = np.concatenate(starts_aug, axis=0)
    augmented["policies"] = np.concatenate(policies_aug, axis=0)
    return augmented


def _resolve_rollout_sampling(
    args: argparse.Namespace,
) -> Tuple[Optional[int], Optional[int]]:
    if args.demonstration_source != "rollout":
        return None, None

    rollout_bank_size = (
        int(args.rollout_bank_size)
        if args.rollout_bank_size is not None
        else int(args.num_rollouts_per_policy)
    )
    rollout_subsample_per_policy = (
        int(args.rollout_subsample_per_policy)
        if args.rollout_subsample_per_policy is not None
        else int(args.num_rollouts_per_policy)
    )

    if rollout_bank_size < 1:
        raise ValueError("rollout-bank-size must be >= 1")
    if rollout_subsample_per_policy < 1:
        raise ValueError("rollout-subsample-per-policy must be >= 1")
    if rollout_subsample_per_policy > rollout_bank_size:
        raise ValueError(
            "rollout-subsample-per-policy cannot exceed rollout-bank-size "
            f"({rollout_subsample_per_policy} > {rollout_bank_size})"
        )

    return rollout_bank_size, rollout_subsample_per_policy


def _subsample_rollout_demonstrations(
    demos: np.ndarray,
    num_rollouts: Optional[int],
    seed: int,
) -> np.ndarray:
    if num_rollouts is None:
        return demos
    if demos.ndim != 4:
        raise ValueError(f"Expected rollout demos shape (N,R,T,D), got {demos.shape}")

    total_rollouts = int(demos.shape[1])
    if num_rollouts >= total_rollouts:
        return demos

    rng = np.random.default_rng(seed)
    out = np.empty(
        (demos.shape[0], num_rollouts, demos.shape[2], demos.shape[3]),
        dtype=demos.dtype,
    )
    for task_idx in range(demos.shape[0]):
        idx = rng.choice(total_rollouts, size=num_rollouts, replace=False)
        idx.sort()
        out[task_idx] = demos[task_idx, idx]
    return out


def _make_rollout_train_collate_fn(num_rollouts: int, seed: int):
    import torch

    rng = np.random.default_rng(seed)

    def _collate(batch):
        demos = []
        targets = []
        for demo, target in batch:
            if num_rollouts < int(demo.shape[0]):
                idx = rng.choice(int(demo.shape[0]), size=num_rollouts, replace=False)
                idx.sort()
                demo = demo.index_select(
                    0,
                    torch.from_numpy(idx.astype(np.int64)),
                )
            demos.append(demo)
            targets.append(target)
        return torch.stack(demos, dim=0), torch.stack(targets, dim=0)

    return _collate


def _build_demonstrations(
    split: Dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
    num_rollouts_per_policy: Optional[int] = None,
) -> np.ndarray:
    if args.demonstration_source == "rollout":
        num_rollouts = (
            int(num_rollouts_per_policy)
            if num_rollouts_per_policy is not None
            else int(args.num_rollouts_per_policy)
        )
        return build_rollout_demonstrations(
            split["walls"],
            split["policies"],
            split["start_states"],
            num_rollouts_per_policy=num_rollouts,
            rollout_horizon=args.rollout_horizon,
            seed=seed,
            random_starts=args.rollout_random_starts,
        )
    return build_policy_demonstrations(split["walls"], split["policies"])


def train(args: argparse.Namespace) -> Dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from sri.reward_inference.models import NonLinearNet

    os.makedirs(args.out, exist_ok=True)

    _set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest = load_manifest(args.dataset)
    train_split = load_split(args.dataset, "planner_train")
    val_split = load_split(args.dataset, "planner_val")
    infer_split = load_split(args.dataset, "reward_infer")

    h, w = grid_shape_from_split(train_split)
    rollout_bank_size, rollout_subsample_per_policy = _resolve_rollout_sampling(args)
    regularization_type, regularization_lambda = _resolve_regularization(args)
    train_augmentation = str(args.train_augmentation).strip().lower()
    train_tasks_original = int(train_split["walls"].shape[0])
    train_split = _augment_training_split(train_split, train_augmentation)
    train_tasks_augmented = int(train_split["walls"].shape[0])
    rollout_eval_size = (
        rollout_subsample_per_policy if args.demonstration_source == "rollout" else None
    )

    train_dems = _build_demonstrations(
        train_split,
        args,
        seed=args.seed + 0,
        num_rollouts_per_policy=rollout_bank_size,
    )
    val_dems = _build_demonstrations(
        val_split,
        args,
        seed=args.seed + 1,
        num_rollouts_per_policy=rollout_eval_size,
    )
    infer_dems = _build_demonstrations(
        infer_split,
        args,
        seed=args.seed + 2,
        num_rollouts_per_policy=rollout_eval_size,
    )

    train_targets = flatten_rewards(train_split["rewards"])
    val_targets = flatten_rewards(val_split["rewards"])

    feature_dim = train_dems.shape[-1]
    demonstration_horizon = int(train_dems.shape[-2])
    effective_train_rollouts_per_policy = (
        int(min(train_dems.shape[1], rollout_subsample_per_policy))
        if args.demonstration_source == "rollout" and rollout_subsample_per_policy is not None
        else None
    )
    effective_eval_rollouts_per_policy = (
        int(infer_dems.shape[1]) if args.demonstration_source == "rollout" else None
    )
    if args.dem_encoder_type == "mlp":
        mlp_input_dim = int(np.prod(train_dems.shape[1:]))
        model_kwargs = {
            "model_type": "mlp",
            "input_dim": mlp_input_dim,
            "output_dim": int(h * w),
            "hidden_size": int(args.final_hidden_size),
            "num_hidden_layers": int(args.num_demonstration_layers),
            "demonstration_horizon": demonstration_horizon,
            "demonstration_source": args.demonstration_source,
        }
        model = _make_flat_mlp(
            input_dim=mlp_input_dim,
            output_dim=int(h * w),
            hidden_size=args.final_hidden_size,
            num_hidden_layers=args.num_demonstration_layers,
        ).to(device)
        resolved_nhead = None
    elif args.dem_encoder_type in {"cnn", "unet"}:
        if args.demonstration_source != "policy":
            raise ValueError(
                "dem-encoder-type cnn/unet currently requires demonstration-source=policy"
            )
        if demonstration_horizon != int(h * w):
            raise ValueError(
                f"Expected policy horizon H*W={h*w}, got {demonstration_horizon}"
            )
        model_kwargs = {
            "model_type": args.dem_encoder_type,
            "grid_height": int(h),
            "grid_width": int(w),
            "in_channels": int(feature_dim),
            "base_channels": int(args.grid_model_base_channels),
            "demonstration_horizon": demonstration_horizon,
            "demonstration_source": args.demonstration_source,
        }
        model = _make_policy_grid_model(
            model_type=args.dem_encoder_type,
            height=int(h),
            width=int(w),
            in_channels=int(feature_dim),
            base_channels=int(args.grid_model_base_channels),
        ).to(device)
        resolved_nhead = None
    else:
        model_kwargs = _make_model_kwargs(h, w, feature_dim, demonstration_horizon, args)
        model = NonLinearNet(**model_kwargs).to(device)
        resolved_nhead = _get_resolved_transformer_nhead(model, args.dem_encoder_type)

    wandb = None
    wandb_run = None
    if args.wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=_parse_wandb_tags(args.wandb_tags),
            config={
                "method": "sri_policy",
                "dataset_dir": os.path.abspath(args.dataset),
                "output_dir": os.path.abspath(args.out),
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "regularization_type": regularization_type,
                "regularization_lambda": float(regularization_lambda),
                "train_augmentation": train_augmentation,
                "train_tasks_original": train_tasks_original,
                "train_tasks_augmented": train_tasks_augmented,
                "device": str(device),
                "dem_encoder_type": args.dem_encoder_type,
                "demonstration_source": args.demonstration_source,
                "num_rollouts_per_policy": (
                    int(args.num_rollouts_per_policy) if args.demonstration_source == "rollout" else None
                ),
                "rollout_bank_size": (
                    int(rollout_bank_size) if args.demonstration_source == "rollout" else None
                ),
                "rollout_subsample_per_policy": (
                    int(rollout_subsample_per_policy) if args.demonstration_source == "rollout" else None
                ),
                "effective_train_rollouts_per_policy": effective_train_rollouts_per_policy,
                "effective_eval_rollouts_per_policy": effective_eval_rollouts_per_policy,
                "rollout_horizon": (
                    int(args.rollout_horizon) if args.demonstration_source == "rollout" else None
                ),
                "rollout_random_starts": bool(args.rollout_random_starts),
                "transformer_nhead": args.transformer_nhead,
                "resolved_transformer_nhead": resolved_nhead,
                "feature_dim": int(feature_dim),
                "reward_dim": int(h * w),
                "demonstration_horizon": demonstration_horizon,
                "grid_model_base_channels": (
                    int(args.grid_model_base_channels)
                    if args.dem_encoder_type in {"cnn", "unet"}
                    else None
                ),
                "model_kwargs": model_kwargs,
                "agent": manifest.get("agent"),
                "dataset_seed": manifest.get("seed"),
                "grid": manifest.get("grid"),
                "split_sizes": manifest.get("split_sizes"),
            },
            mode=args.wandb_mode if args.wandb_mode else None,
            reinit=True,
        )

    if args.verbose:
        print(
            f"dem_encoder_type={args.dem_encoder_type} demonstration_source={args.demonstration_source} "
            f"feature_dim={feature_dim} horizon={demonstration_horizon} "
            f"transformer_nhead={args.transformer_nhead} resolved_nhead={resolved_nhead}"
        )
        print(
            f"regularization_type={regularization_type} "
            f"regularization_lambda={regularization_lambda} "
            f"legacy_weight_decay_arg={args.weight_decay}"
        )
        print(
            f"train_augmentation={train_augmentation} "
            f"train_tasks_original={train_tasks_original} "
            f"train_tasks_augmented={train_tasks_augmented}"
        )
        if args.demonstration_source == "rollout":
            print(
                f"rollout_bank_size={rollout_bank_size} "
                f"rollout_subsample_per_policy={rollout_subsample_per_policy} "
                f"effective_train_rollouts_per_policy={effective_train_rollouts_per_policy} "
                f"effective_eval_rollouts_per_policy={effective_eval_rollouts_per_policy}"
            )

    train_ds = TensorDataset(
        torch.from_numpy(train_dems),
        torch.from_numpy(train_targets),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_dems),
        torch.from_numpy(val_targets),
    )

    train_collate_fn = None
    if (
        args.demonstration_source == "rollout"
        and rollout_subsample_per_policy is not None
        and rollout_subsample_per_policy < int(train_dems.shape[1])
    ):
        train_collate_fn = _make_rollout_train_collate_fn(
            num_rollouts=rollout_subsample_per_policy,
            seed=args.seed + 303,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    best_val = float("inf")
    best_state = None
    train_losses = []
    train_mse_losses = []
    train_reg_losses = []
    val_losses = []
    last_pred_vecs = None
    try:
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            mse_total = 0.0
            reg_total = 0.0
            count = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                pred = model(x_batch)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                mse_loss = loss_fn(pred, y_batch)
                if regularization_type != "none":
                    reg_penalty = _regularization_penalty(model, regularization_type)
                    reg_loss = regularization_lambda * reg_penalty
                    loss = mse_loss + reg_loss
                else:
                    reg_loss = mse_loss.new_zeros(())
                    loss = mse_loss
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * x_batch.shape[0]
                mse_total += float(mse_loss.item()) * x_batch.shape[0]
                reg_total += float(reg_loss.item()) * x_batch.shape[0]
                count += x_batch.shape[0]
            train_loss = total / max(count, 1)
            train_mse_loss = mse_total / max(count, 1)
            train_reg_loss = reg_total / max(count, 1)
            train_losses.append(train_loss)
            train_mse_losses.append(train_mse_loss)
            train_reg_losses.append(train_reg_loss)

            model.eval()
            v_total = 0.0
            v_count = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(x_batch)
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(0)
                    loss = loss_fn(pred, y_batch)
                    v_total += float(loss.item()) * x_batch.shape[0]
                    v_count += x_batch.shape[0]
            val_loss = v_total / max(v_count, 1)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": float(train_loss),
                        "train/mse_loss": float(train_mse_loss),
                        "train/regularization_loss": float(train_reg_loss),
                        "val/loss": float(val_loss),
                        "val/best_loss": float(best_val),
                        "train/epoch": int(epoch + 1),
                    },
                    step=int(epoch + 1),
                )

            if args.verbose:
                print(
                    f"epoch={epoch+1}/{args.epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )

        if args.save_last_pred:
            # Capture inference predictions from the final training step.
            last_pred_vecs = _predict(model, device, infer_dems, args.eval_batch_size)

        if best_state is not None:
            model.load_state_dict(best_state)

        pred_vecs = _predict(model, device, infer_dems, args.eval_batch_size)
    finally:
        if wandb_run is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
            if train_losses:
                wandb_run.summary["final_train_loss"] = float(train_losses[-1])
            if train_mse_losses:
                wandb_run.summary["final_train_mse_loss"] = float(train_mse_losses[-1])
            if train_reg_losses:
                wandb_run.summary["final_train_regularization_loss"] = float(train_reg_losses[-1])
            if val_losses:
                wandb_run.summary["final_val_loss"] = float(val_losses[-1])
            wandb_run.finish()

    model_path = os.path.join(args.out, "sri_policy_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "grid_shape": [h, w],
            "feature_dim": feature_dim,
            "dataset_manifest": manifest,
            "resolved_transformer_nhead": resolved_nhead,
        },
        model_path,
    )

    pred_path = os.path.join(args.out, "sri_pred_reward_vec.npy")
    np.save(pred_path, pred_vecs.astype(np.float32))
    last_pred_path = None
    if last_pred_vecs is not None:
        last_pred_path = os.path.join(args.out, "sri_pred_reward_vec_last.npy")
        np.save(last_pred_path, last_pred_vecs.astype(np.float32))

    metrics = {
        "method": "sri_policy",
        "dataset_dir": os.path.abspath(args.dataset),
        "output_dir": os.path.abspath(args.out),
        "model_path": os.path.abspath(model_path),
        "pred_path": os.path.abspath(pred_path),
        "last_pred_path": (os.path.abspath(last_pred_path) if last_pred_path else None),
        "save_last_pred": bool(args.save_last_pred),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "regularization_type": regularization_type,
        "regularization_lambda": float(regularization_lambda),
        "train_augmentation": train_augmentation,
        "train_tasks_original": train_tasks_original,
        "train_tasks_augmented": train_tasks_augmented,
        "best_val_loss": float(best_val),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_train_mse_loss": float(train_mse_losses[-1]) if train_mse_losses else None,
        "final_train_regularization_loss": float(train_reg_losses[-1]) if train_reg_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        "train_losses": train_losses,
        "train_mse_losses": train_mse_losses,
        "train_regularization_losses": train_reg_losses,
        "val_losses": val_losses,
        "feature_dim": int(feature_dim),
        "reward_dim": int(h * w),
        "dem_encoder_type": args.dem_encoder_type,
        "grid_model_base_channels": (
            int(args.grid_model_base_channels)
            if args.dem_encoder_type in {"cnn", "unet"}
            else None
        ),
        "demonstration_source": args.demonstration_source,
        "demonstration_horizon": demonstration_horizon,
        "num_rollouts_per_policy": (
            int(args.num_rollouts_per_policy) if args.demonstration_source == "rollout" else None
        ),
        "rollout_bank_size": (
            int(rollout_bank_size) if args.demonstration_source == "rollout" else None
        ),
        "rollout_subsample_per_policy": (
            int(rollout_subsample_per_policy) if args.demonstration_source == "rollout" else None
        ),
        "effective_train_rollouts_per_policy": effective_train_rollouts_per_policy,
        "effective_eval_rollouts_per_policy": effective_eval_rollouts_per_policy,
        "rollout_horizon": (
            int(args.rollout_horizon) if args.demonstration_source == "rollout" else None
        ),
        "rollout_random_starts": bool(args.rollout_random_starts),
        "transformer_nhead": args.transformer_nhead,
        "resolved_transformer_nhead": (
            int(resolved_nhead) if resolved_nhead is not None else None
        ),
    }
    if wandb_run is not None:
        metrics["wandb_run_url"] = wandb_run.url
        metrics["wandb_run_id"] = wandb_run.id
    save_json(os.path.join(args.out, "sri_train_metrics.json"), metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRI-Policy reward-map regressor from policy tensors")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--train-augmentation",
        default="none",
        choices=["none", "d4"],
        help="Apply geometric augmentation to planner_train only.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help=(
            "Legacy alias for L2 regularization lambda. "
            "If regularization-type=none and weight-decay>0, L2 is enabled."
        ),
    )
    parser.add_argument(
        "--regularization-type",
        default="none",
        choices=["none", "l1", "l2"],
        help="Optional parameter regularization added to training loss.",
    )
    parser.add_argument(
        "--regularization-lambda",
        type=float,
        default=1e-5,
        help="Regularization coefficient for l1/l2.",
    )
    parser.add_argument(
        "--save-last-pred",
        action="store_true",
        help="Also save reward predictions from the last training epoch model.",
    )

    parser.add_argument(
        "--dem-encoder-type",
        default="transformer",
        choices=["set_transformer", "transformer", "tst", "lstm", "mlp", "cnn", "unet"],
        help="Encoder family for policy inputs. Includes flat MLP and spatial CNN/U-Net options.",
    )
    parser.add_argument(
        "--grid-model-base-channels",
        type=int,
        default=32,
        help="Base channel count for dem-encoder-type cnn/unet.",
    )
    parser.add_argument(
        "--demonstration-source",
        default="policy",
        choices=["policy", "rollout"],
        help="Use full policy tensors or sampled rollouts as demonstrations.",
    )
    parser.add_argument(
        "--num-rollouts-per-policy",
        type=int,
        default=100,
        help=(
            "If demonstration-source=rollout, default per-task rollout count. "
            "Also used as the train-time subsample size unless overridden."
        ),
    )
    parser.add_argument(
        "--rollout-bank-size",
        type=int,
        default=None,
        help=(
            "If demonstration-source=rollout, generate this many rollouts per task first. "
            "Use with --rollout-subsample-per-policy for large-bank subsampling."
        ),
    )
    parser.add_argument(
        "--rollout-subsample-per-policy",
        type=int,
        default=None,
        help=(
            "If demonstration-source=rollout and smaller than rollout bank size, "
            "randomly subsample this many rollouts per task for each training batch."
        ),
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=20,
        help="If demonstration-source=rollout, number of timesteps per rollout.",
    )
    parser.add_argument(
        "--rollout-random-starts",
        action="store_true",
        help="If set, start each sampled rollout from a random traversable state.",
    )
    parser.add_argument(
        "--transformer-nhead",
        type=int,
        default=None,
        help="Optional attention head count for transformer/tst demonstration encoder.",
    )
    parser.add_argument("--num-demonstration-layers", type=int, default=2)
    parser.add_argument("--num-state-layers", type=int, default=2)
    parser.add_argument("--state-hidden-size", type=int, default=128)
    parser.add_argument("--final-hidden-size", type=int, default=128)
    parser.add_argument("--demonstration-hidden-size", type=int, default=128)
    parser.add_argument("--internal-tst-dim", type=int, default=128)

    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode override.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
