import argparse
import hashlib
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _ensure_legacy_import_paths() -> None:
    """Allow learning_biases legacy absolute imports to resolve."""
    here = os.path.dirname(__file__)
    repo_root = os.path.dirname(here)
    for p in (repo_root, here):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, here)
    sys.path.insert(0, repo_root)

    # Force `gridworld` to resolve to the compatibility package, not
    # learning_biases/gridworld.py.
    if "gridworld" in sys.modules and not hasattr(sys.modules["gridworld"], "__path__"):
        del sys.modules["gridworld"]
    importlib.import_module("gridworld")


def _load_generation_fns():
    _ensure_legacy_import_paths()
    from learning_biases.gridworld_data import (
        create_agent,
        generate_n_examples,
        parse_rewards_into_goals,
    )

    return create_agent, generate_n_examples, parse_rewards_into_goals


NUM_ACTIONS = 5
SCHEMA_VERSION = "1.0"


@dataclass
class ExportConfig:
    agent: str
    seed: int
    imsize: int = 16
    noise: float = 0.2
    num_rewards: int = 7
    planner_train_size: int = 5000
    planner_val_size: int = 2000
    reward_infer_size: int = 1000
    gamma: float = 0.95
    beta: Optional[float] = 1.0
    num_iters: int = 50
    max_delay: int = 10
    hyperbolic_constant: float = 1.0
    calibration_factor: float = 1.0
    simple_mdp: bool = False
    wall_prob: float = 0.05
    reward_prob: float = 0.05
    action_distance_threshold: float = 0.5
    max_generation_retries: int = 20


def _default_agent_params(agent: str) -> Tuple[float, float]:
    if agent in ("optimal", "overconfident"):
        beta = 0.1
    else:
        beta = 1.0

    if agent == "overconfident":
        calibration = 5.0
    elif agent == "underconfident":
        calibration = 0.5
    else:
        calibration = 1.0

    return beta, calibration


def _build_generation_namespace(cfg: ExportConfig):
    return argparse.Namespace(
        simple_mdp=cfg.simple_mdp,
        imsize=cfg.imsize,
        num_actions=NUM_ACTIONS,
        wall_prob=cfg.wall_prob,
        reward_prob=cfg.reward_prob,
        num_rewards=cfg.num_rewards,
        noise=cfg.noise,
        action_distance_threshold=cfg.action_distance_threshold,
        gamma=cfg.gamma,
        beta=cfg.beta,
        num_iters=cfg.num_iters,
        max_delay=cfg.max_delay,
        hyperbolic_constant=cfg.hyperbolic_constant,
        calibration_factor=cfg.calibration_factor,
        other_agent=None,
        other_gamma=cfg.gamma,
        other_beta=cfg.beta,
        other_num_iters=cfg.num_iters,
        other_max_delay=cfg.max_delay,
        other_hyperbolic_constant=cfg.hyperbolic_constant,
        other_calibration_factor=cfg.calibration_factor,
    )


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _validate_policy_tensor(walls: np.ndarray, policies: np.ndarray, tol: float = 1e-4) -> Dict[str, float]:
    row_sums = np.sum(policies, axis=-1)
    walls_bool = walls.astype(bool)
    non_walls = ~walls_bool

    non_wall_row_sums = row_sums[non_walls]
    max_row_sum_err = float(np.max(np.abs(non_wall_row_sums - 1.0))) if non_wall_row_sums.size else 0.0

    wall_action_probs = policies[walls_bool]
    wall_stay_probs = wall_action_probs[:, -1] if wall_action_probs.size else np.array([], dtype=np.float32)
    wall_non_stay = wall_action_probs[:, :-1] if wall_action_probs.size else np.array([], dtype=np.float32)
    max_wall_non_stay = float(np.max(np.abs(wall_non_stay))) if wall_non_stay.size else 0.0
    min_wall_stay = float(np.min(wall_stay_probs)) if wall_stay_probs.size else 1.0

    if max_row_sum_err > tol:
        raise ValueError(f"Policy row sums invalid for non-wall states (max err {max_row_sum_err})")
    if max_wall_non_stay > tol:
        raise ValueError(f"Wall states have non-STAY action mass (max {max_wall_non_stay})")
    if wall_stay_probs.size and float(np.max(np.abs(wall_stay_probs - 1.0))) > tol:
        raise ValueError("Wall states do not have STAY probability of 1.0")

    return {
        "max_non_wall_row_sum_error": max_row_sum_err,
        "max_wall_non_stay_prob": max_wall_non_stay,
        "min_wall_stay_prob": min_wall_stay,
    }


def _save_split(path: str, walls: np.ndarray, rewards: np.ndarray, starts: np.ndarray, policies: np.ndarray, extras: Dict[str, np.ndarray] = None) -> Dict[str, object]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    walls = walls.astype(np.float32)
    rewards = rewards.astype(np.float32)
    starts = starts.astype(np.int32)
    policies = policies.astype(np.float32)

    payload = {
        "walls": walls,
        "rewards": rewards,
        "start_states": starts,
        "policies": policies,
    }
    if extras:
        payload.update(extras)

    np.savez_compressed(path, **payload)

    info = {
        "path": path,
        "num_tasks": int(walls.shape[0]),
        "shapes": {k: list(v.shape) for k, v in payload.items()},
        "dtypes": {k: str(v.dtype) for k, v in payload.items()},
        "hashes": {k: _sha256_array(v) for k, v in payload.items()},
        "policy_validation": _validate_policy_tensor(walls, policies),
    }
    return info


def _generate_split_with_retries(
    n: int,
    generate_fn,
    agent,
    config,
    seed: int,
    folder: str,
    goals=None,
    max_retries: int = 20,
):
    """Generate n examples, retrying on known random-grid construction failures."""
    last_error = None
    for attempt in range(max_retries + 1):
        current_seed = seed + attempt
        try:
            data = generate_fn(
                n,
                agent,
                config,
                seed=current_seed,
                other_agents=[],
                goals=goals,
                folder=folder,
            )
            return data, current_seed, attempt
        except IndexError as e:
            # Known failure mode in gridworld connected generation.
            if "pop from empty list" not in str(e):
                raise
            last_error = e
            print(
                f"[warn] generation failed for seed={current_seed} with '{e}'. "
                f"Retrying with seed={current_seed + 1} ({attempt + 1}/{max_retries})"
            )

    raise RuntimeError(
        f"Failed to generate split after {max_retries + 1} seeds "
        f"(starting at seed={seed})"
    ) from last_error


def export_dataset(cfg: ExportConfig, out_dir: str) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    create_agent, generate_n_examples, parse_rewards_into_goals = _load_generation_fns()
    cache_dir = os.path.join(out_dir, "cache") + os.sep

    gen_cfg = _build_generation_namespace(cfg)
    agent = create_agent(
        cfg.agent,
        cfg.gamma,
        cfg.beta,
        cfg.num_iters,
        cfg.max_delay,
        cfg.hyperbolic_constant,
        cfg.calibration_factor,
    )

    train_seed = cfg.seed
    val_seed = cfg.seed + 1
    reward_seed = cfg.seed + 2
    reward_test_seed = cfg.seed + 3

    (train_w, train_r, train_s, train_pi), train_seed_used, train_retries = _generate_split_with_retries(
        cfg.planner_train_size,
        generate_n_examples,
        agent,
        gen_cfg,
        seed=train_seed,
        folder=cache_dir,
        max_retries=cfg.max_generation_retries,
    )
    (val_w, val_r, val_s, val_pi), val_seed_used, val_retries = _generate_split_with_retries(
        cfg.planner_val_size,
        generate_n_examples,
        agent,
        gen_cfg,
        seed=val_seed,
        folder=cache_dir,
        max_retries=cfg.max_generation_retries,
    )

    (infer_w, infer_r, infer_s, infer_pi), infer_seed_used, infer_retries = _generate_split_with_retries(
        cfg.reward_infer_size,
        generate_n_examples,
        agent,
        gen_cfg,
        seed=reward_seed,
        folder=cache_dir,
        max_retries=cfg.max_generation_retries,
    )

    infer_goals = parse_rewards_into_goals(infer_r)
    (infer_test_w, infer_test_r, infer_test_s, infer_test_pi), reward_test_seed_used, reward_test_retries = _generate_split_with_retries(
        cfg.reward_infer_size,
        generate_n_examples,
        agent,
        gen_cfg,
        seed=reward_test_seed,
        folder=cache_dir,
        goals=infer_goals,
        max_retries=cfg.max_generation_retries,
    )

    if not np.array_equal(infer_r, infer_test_r):
        raise ValueError("Reward-infer test set rewards do not match reward-infer rewards")

    split_infos = {}
    split_infos["planner_train"] = _save_split(
        os.path.join(out_dir, "planner_train.npz"),
        train_w,
        train_r,
        train_s,
        train_pi,
    )
    split_infos["planner_val"] = _save_split(
        os.path.join(out_dir, "planner_val.npz"),
        val_w,
        val_r,
        val_s,
        val_pi,
    )
    split_infos["reward_infer"] = _save_split(
        os.path.join(out_dir, "reward_infer.npz"),
        infer_w,
        infer_r,
        infer_s,
        infer_pi,
        extras={
            "test_walls": infer_test_w.astype(np.float32),
            "test_start_states": infer_test_s.astype(np.int32),
            "test_policies": infer_test_pi.astype(np.float32),
        },
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "agent": cfg.agent,
        "seed": int(cfg.seed),
        "num_actions": NUM_ACTIONS,
        "grid": {
            "height": int(cfg.imsize),
            "width": int(cfg.imsize),
            "includes_borders": True,
        },
        "split_sizes": {
            "planner_train": int(cfg.planner_train_size),
            "planner_val": int(cfg.planner_val_size),
            "reward_infer": int(cfg.reward_infer_size),
        },
        "config": asdict(cfg),
        "effective_seeds": {
            "planner_train": int(train_seed_used),
            "planner_val": int(val_seed_used),
            "reward_infer": int(infer_seed_used),
            "reward_infer_test": int(reward_test_seed_used),
        },
        "generation_retries": {
            "planner_train": int(train_retries),
            "planner_val": int(val_retries),
            "reward_infer": int(infer_retries),
            "reward_infer_test": int(reward_test_retries),
        },
        "splits": split_infos,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical learning_biases bridge datasets")
    parser.add_argument("--agent", required=True, choices=["optimal", "naive", "sophisticated", "myopic", "underconfident", "overconfident"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output dataset directory")

    parser.add_argument("--imsize", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--num-rewards", type=int, default=7)

    parser.add_argument("--planner-train-size", type=int, default=5000)
    parser.add_argument("--planner-val-size", type=int, default=2000)
    parser.add_argument("--reward-infer-size", type=int, default=1000)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument(
        "--beta",
        default="default",
        help=(
            "Agent rationality. Use a float (e.g. 1.0), "
            "'default' for agent-specific default, or 'none' to disable Boltzmann noise."
        ),
    )
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--max-delay", type=int, default=10)
    parser.add_argument("--hyperbolic-constant", type=float, default=1.0)
    parser.add_argument("--calibration-factor", type=float, default=None)
    parser.add_argument("--max-generation-retries", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    beta_default, calibration_default = _default_agent_params(args.agent)
    beta_arg = str(args.beta).strip().lower()
    if beta_arg in ("default", "auto"):
        beta_value = beta_default
    elif beta_arg in ("none", "null"):
        beta_value = None
    else:
        beta_value = float(args.beta)

    cfg = ExportConfig(
        agent=args.agent,
        seed=args.seed,
        imsize=args.imsize,
        noise=args.noise,
        num_rewards=args.num_rewards,
        planner_train_size=args.planner_train_size,
        planner_val_size=args.planner_val_size,
        reward_infer_size=args.reward_infer_size,
        gamma=args.gamma,
        beta=beta_value,
        num_iters=args.num_iters,
        max_delay=args.max_delay,
        hyperbolic_constant=args.hyperbolic_constant,
        calibration_factor=(
            calibration_default if args.calibration_factor is None else args.calibration_factor
        ),
        max_generation_retries=args.max_generation_retries,
    )

    manifest = export_dataset(cfg, args.out)
    print(json.dumps({
        "status": "ok",
        "dataset_dir": os.path.abspath(args.out),
        "manifest": os.path.join(os.path.abspath(args.out), "manifest.json"),
        "agent": manifest["agent"],
        "seed": manifest["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
