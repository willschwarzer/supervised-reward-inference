import argparse
import importlib
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np


def _ensure_legacy_import_paths() -> None:
    """Allow legacy absolute imports in learning_biases/train.py to resolve."""
    here = os.path.dirname(__file__)
    repo_root = os.path.dirname(here)
    for p in (repo_root, here):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, here)
    sys.path.insert(0, repo_root)

    if "gridworld" in sys.modules and not hasattr(sys.modules["gridworld"], "__path__"):
        del sys.modules["gridworld"]
    importlib.import_module("gridworld")


def _load_split(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        data["walls"].astype(np.float32),
        data["rewards"].astype(np.float32),
        data["start_states"].astype(np.int32),
        data["policies"].astype(np.float32),
    )


def _load_manifest(dataset_dir: str) -> Dict[str, object]:
    path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batchsize=args.batchsize,
        imsize=args.imsize,
        num_actions=args.num_actions,
        k=args.k,
        ch_h=args.ch_h,
        ch_p=args.ch_p,
        ch_q=args.ch_q,
        verbosity=args.verbosity,
        model=args.model,
        vin_regularizer_C=args.vin_regularizer_C,
        reward_regularizer_C=args.reward_regularizer_C,
        lr=args.lr,
        reward_lr=args.reward_lr,
        display_step=args.display_step,
        savemodel=False,
        log=False,
        use_gpu=args.use_gpu,
        reward_epochs=args.reward_epochs,
        epochs=args.epochs,
        em_iterations=args.em_iterations,
        plot_rewards=False,
        num_iters=args.num_iters,
        gamma=args.gamma,
        noise=args.noise,
    )


def _beta_from_method_and_manifest(
    method: str, args: argparse.Namespace, manifest: Dict[str, object]
) -> Optional[float]:
    if method == "optimal_planner":
        return None
    if method not in {"boltzmann_planner", "joint_with_init"}:
        return None

    if args.planner_beta is not None:
        return args.planner_beta

    manifest_cfg = manifest.get("config", {})
    if isinstance(manifest_cfg, dict):
        manifest_beta = manifest_cfg.get("beta")
    else:
        manifest_beta = None
    # Match legacy behavior:
    # - boltzmann_planner falls back to beta=1.0 when unspecified
    # - joint_with_init uses config.beta directly (None remains None)
    if method == "boltzmann_planner":
        if manifest_beta is None:
            return 1.0
        return float(manifest_beta)

    if manifest_beta is None:
        return None
    return float(manifest_beta)


def _make_rational_planner_split(
    split: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    gamma: float,
    beta: Optional[float],
    num_iters: int,
    num_actions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from learning_biases.fast_agents import FastOptimalAgent
    from learning_biases.gridworld import Direction, GridworldMdp

    walls, rewards, start_states, _ = split
    n, h, w = walls.shape
    policies = np.zeros((n, h, w, num_actions), dtype=np.float32)

    stay_idx = Direction.get_number_from_direction(Direction.STAY)
    wall_policy = np.zeros((num_actions,), dtype=np.float32)
    wall_policy[stay_idx] = 1.0

    planner = FastOptimalAgent(gamma=gamma, beta=beta, num_iters=num_iters)
    to_index = Direction.get_number_from_direction

    for i in range(n):
        mdp = GridworldMdp.from_numpy_input(
            walls[i],
            rewards[i],
            tuple(start_states[i].tolist()),
        )
        planner.set_mdp(mdp)
        for y in range(h):
            for x in range(w):
                if walls[i, y, x] > 0.5:
                    policies[i, y, x] = wall_policy
                else:
                    dist = planner.get_action_distribution((x, y))
                    policies[i, y, x] = dist.as_numpy_array(to_index, num_actions)

    return walls, rewards, policies


def _select_algorithm(
    method: str,
    args: argparse.Namespace,
    manifest: Dict[str, object],
    planner_train: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    planner_val: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
):
    from learning_biases.train import joint_algorithm, two_phase_algorithm

    if method == "given_rewards":
        train_tuple = (planner_train[0], planner_train[1], planner_train[3])
        val_tuple = (planner_val[0], planner_val[1], planner_val[3])
        return two_phase_algorithm, train_tuple, val_tuple, None

    if method not in {"boltzmann_planner", "optimal_planner", "joint_with_init"}:
        raise ValueError(f"Unsupported Shah method: {method}")

    assumed_beta = _beta_from_method_and_manifest(method, args, manifest)
    train_sim = _make_rational_planner_split(
        planner_train,
        gamma=args.gamma,
        beta=assumed_beta,
        num_iters=args.num_iters,
        num_actions=args.num_actions,
    )
    val_sim = _make_rational_planner_split(
        planner_val,
        gamma=args.gamma,
        beta=assumed_beta,
        num_iters=args.num_iters,
        num_actions=args.num_actions,
    )
    algorithm_fn = joint_algorithm if method == "joint_with_init" else two_phase_algorithm
    return algorithm_fn, train_sim, val_sim, assumed_beta


def run_shah(dataset_dir: str, out_dir: str, args: argparse.Namespace) -> Dict[str, object]:
    _ensure_legacy_import_paths()

    # Import after path adjustment.
    from learning_biases.agent_runner import evaluate_proxy
    from learning_biases.train import PlannerArchitecture
    from learning_biases.utils import set_seeds
    import tensorflow as tf

    os.makedirs(out_dir, exist_ok=True)
    manifest = _load_manifest(dataset_dir)
    method = str(args.method).strip()

    planner_train = _load_split(os.path.join(dataset_dir, "planner_train.npz"))
    planner_val = _load_split(os.path.join(dataset_dir, "planner_val.npz"))

    reward_data = np.load(os.path.join(dataset_dir, "reward_infer.npz"))
    image_irl = reward_data["walls"].astype(np.float32)
    reward_irl = reward_data["rewards"].astype(np.float32)
    start_states_irl = reward_data["start_states"].astype(np.int32)
    y_irl = reward_data["policies"].astype(np.float32)

    if "test_walls" in reward_data:
        image_test = reward_data["test_walls"].astype(np.float32)
        start_states_test = reward_data["test_start_states"].astype(np.int32)
        y_test = reward_data["test_policies"].astype(np.float32)
    else:
        image_test = image_irl
        start_states_test = start_states_irl
        y_test = y_irl

    config = _build_config(args)
    set_seeds(args.seed)

    architecture = PlannerArchitecture(config)

    algorithm_fn, train_tuple, val_tuple, assumed_beta = _select_algorithm(
        method,
        args,
        manifest,
        planner_train,
        planner_val,
    )
    reward_tuple = (image_irl, y_irl)

    gpu_config = None
    if config.use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)

    logs = {
        "train_planner_costs": [],
        "train_planner_train_errs": [],
        "train_planner_validation_errs": [],
        "train_planner_times": [],
        "train_planner_predicted_action_dists": [],
        "train_planner_actual_action_dists": [],
        "train_reward_costs": [],
        "train_reward_errs": [],
        "train_joint_costs": [],
        "train_joint_errs": [],
        "train_joint_times": [],
        "accuracy": [],
    }

    with tf.Session(config=gpu_config) as sess:
        architecture.register_new_session(sess)
        inferred_rewards = algorithm_fn(
            architecture,
            sess,
            train_tuple,
            val_tuple,
            reward_tuple,
            config,
            logs,
        )

        reward_percents = []
        for label, reward, wall, start_state in zip(
            reward_irl, inferred_rewards, image_irl, start_states_irl
        ):
            reward_percents.append(
                float(
                    evaluate_proxy(
                        wall,
                        tuple(start_state.tolist()),
                        reward,
                        label,
                        episode_length=args.episode_length,
                        gamma=args.gamma,
                    )
                )
            )

        average_percent_reward = float(np.mean(reward_percents))

        avg_loss, err = architecture.evaluate_loss_and_err(
            sess,
            image_test,
            inferred_rewards,
            y_test,
            logs,
        )

    inferred_rewards = np.asarray(inferred_rewards, dtype=np.float32)
    np.save(os.path.join(out_dir, "shah_inferred_rewards.npy"), inferred_rewards)

    result = {
        "method": f"shah_{method}",
        "shah_method": method,
        "planner_assumption_beta": assumed_beta,
        "dataset_dir": os.path.abspath(dataset_dir),
        "average_percent_reward": average_percent_reward,
        "average_regret": 1.0 - average_percent_reward,
        "percent_rewards": reward_percents,
        "average_loss_on_test_walls": float(avg_loss),
        "error_on_test_walls": float(err),
        "accuracy_on_test_walls": float(1.0 - err),
        "num_tasks": int(len(reward_percents)),
        "episode_length": int(args.episode_length),
        "gamma": float(args.gamma),
        "batchsize": int(args.batchsize),
    }

    with open(os.path.join(out_dir, "shah_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Shah-style IRL baseline on exported bridge dataset. "
            "Supports given_rewards, boltzmann_planner, optimal_planner, and joint_with_init."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory with planner_train/val/reward_infer npz files")
    parser.add_argument("--out", required=True, help="Output directory for Shah predictions and metrics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        default="given_rewards",
        choices=["given_rewards", "boltzmann_planner", "optimal_planner", "joint_with_init"],
        help="Which Shah baseline variant to run.",
    )
    parser.add_argument(
        "--planner-beta",
        type=lambda x: None if str(x).strip().lower() in {"none", "null"} else float(x),
        default=None,
        help=(
            "Optional override for assumed beta in boltzmann_planner/joint_with_init. "
            "Default behavior: boltzmann_planner uses dataset beta if present, "
            "else 1.0; joint_with_init uses dataset beta directly (including None)."
        ),
    )

    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--imsize", type=int, default=16)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ch-h", type=int, default=150)
    parser.add_argument("--ch-p", type=int, default=5)
    parser.add_argument("--ch-q", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.2)

    parser.add_argument("--model", default="VIN")
    parser.add_argument("--vin-regularizer-C", type=float, default=1e-4)
    parser.add_argument("--reward-regularizer-C", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reward-lr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--reward-epochs", type=int, default=50)
    parser.add_argument("--em-iterations", type=int, default=0)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--episode-length", type=int, default=20)

    parser.add_argument("--display-step", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_shah(args.dataset, args.out, args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
