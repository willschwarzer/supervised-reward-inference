import argparse
import json
import os

import numpy as np

from sri.learning_biases_bridge.common import (
    evaluate_reward_predictions,
    grid_shape_from_split,
    load_manifest,
    load_split,
    maybe_load_json,
    save_json,
    unflatten_reward_vecs,
)


def evaluate(args: argparse.Namespace):
    os.makedirs(args.out, exist_ok=True)

    manifest = load_manifest(args.dataset)
    infer_split = load_split(args.dataset, "reward_infer")
    h, w = grid_shape_from_split(infer_split)

    pred_vec = np.load(args.pred).astype(np.float32)
    if pred_vec.shape[0] != infer_split["rewards"].shape[0]:
        raise ValueError(
            f"Prediction count mismatch: {pred_vec.shape[0]} vs {infer_split['rewards'].shape[0]}"
        )

    pred_rewards = unflatten_reward_vecs(pred_vec, h, w)
    np.save(os.path.join(args.out, "sri_inferred_rewards.npy"), pred_rewards)

    sri_metrics = evaluate_reward_predictions(
        infer_split["walls"],
        infer_split["start_states"],
        pred_rewards,
        infer_split["rewards"],
        gamma=args.gamma,
        episode_length=args.episode_length,
    )
    sri_metrics.update(
        {
            "method": "sri_policy",
            "dataset_dir": os.path.abspath(args.dataset),
            "pred_path": os.path.abspath(args.pred),
            "output_dir": os.path.abspath(args.out),
        }
    )
    save_json(os.path.join(args.out, "sri_metrics.json"), sri_metrics)

    shah_metrics_path = args.shah_metrics
    if shah_metrics_path is None:
        candidate = os.path.join(args.out, "shah_metrics.json")
        if os.path.exists(candidate):
            shah_metrics_path = candidate

    shah_metrics = maybe_load_json(shah_metrics_path)

    comparison = {
        "schema_version": "1.0",
        "agent": manifest.get("agent"),
        "seed": manifest.get("seed"),
        "num_tasks": sri_metrics["num_tasks"],
        "sri_policy": {
            "average_percent_reward": sri_metrics["average_percent_reward"],
            "average_regret": sri_metrics["average_regret"],
        },
    }

    task_metrics = []
    sri_per_task = sri_metrics["percent_rewards"]
    shah_per_task = shah_metrics.get("percent_rewards") if shah_metrics is not None else None

    for idx, sri_val in enumerate(sri_per_task):
        row = {"index": idx, "sri_percent_reward": float(sri_val)}
        if shah_per_task is not None and idx < len(shah_per_task):
            row["shah_percent_reward"] = float(shah_per_task[idx])
            row["delta_sri_minus_shah"] = float(sri_val - shah_per_task[idx])
        task_metrics.append(row)

    comparison["task_metrics"] = task_metrics

    if shah_metrics is not None:
        comparison["shah_given_rewards"] = {
            "average_percent_reward": shah_metrics["average_percent_reward"],
            "average_regret": shah_metrics["average_regret"],
        }
        comparison["delta"] = {
            "sri_minus_shah": float(
                sri_metrics["average_percent_reward"] - shah_metrics["average_percent_reward"]
            )
        }

    save_json(os.path.join(args.out, "comparison.json"), comparison)
    return {"sri_metrics": sri_metrics, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SRI reward vectors with Shah metric")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pred", required=True, help="Path to sri_pred_reward_vec.npy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shah-metrics", default=None)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episode-length", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    print(json.dumps(payload["sri_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
