import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_corr(a: np.ndarray, b: np.ndarray):
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.size == 0 or b_flat.size == 0:
        return None
    if float(np.std(a_flat)) < 1e-12 or float(np.std(b_flat)) < 1e-12:
        return None
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _trimmed_mean(values: np.ndarray, frac: float = 0.05) -> float:
    if values.size == 0:
        return 0.0
    lo = float(np.quantile(values, frac))
    hi = float(np.quantile(values, 1.0 - frac))
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        return float(np.mean(values))
    return float(np.mean(trimmed))


def _reward_reconstruction_metrics(pred_vecs: np.ndarray, true_rewards: np.ndarray) -> dict:
    true_vecs = true_rewards.reshape(true_rewards.shape[0], -1).astype(np.float32)
    if pred_vecs.shape != true_vecs.shape:
        raise ValueError(f"Prediction shape {pred_vecs.shape} does not match true reward shape {true_vecs.shape}")

    diff = pred_vecs - true_vecs
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "corr": _safe_corr(pred_vecs, true_vecs),
        "pred_mean": float(np.mean(pred_vecs)),
        "pred_std": float(np.std(pred_vecs)),
    }


def build_summary(runs_dir: Path, dataset_dir: Path) -> dict:
    aligned_train = _load_json(runs_dir / "aligned" / "sri_train_metrics.json")
    shuffled_train = _load_json(runs_dir / "shuffled" / "sri_train_metrics.json")
    aligned_eval = _load_json(runs_dir / "aligned_eval" / "sri_metrics.json")
    shuffled_eval = _load_json(runs_dir / "shuffled_eval" / "sri_metrics.json")
    with np.load(dataset_dir / "reward_infer.npz") as arr:
        true_rewards = arr["rewards"].astype(np.float32)

    aligned_pred = np.load(runs_dir / "aligned" / "sri_pred_reward_vec.npy").astype(np.float32)
    shuffled_pred = np.load(runs_dir / "shuffled" / "sri_pred_reward_vec.npy").astype(np.float32)

    aligned_recon = _reward_reconstruction_metrics(aligned_pred, true_rewards)
    shuffled_recon = _reward_reconstruction_metrics(shuffled_pred, true_rewards)
    aligned_eval_arr = np.asarray(aligned_eval["percent_rewards"], dtype=np.float32)
    shuffled_eval_arr = np.asarray(shuffled_eval["percent_rewards"], dtype=np.float32)
    true_vecs = true_rewards.reshape(true_rewards.shape[0], -1).astype(np.float32)
    mean_baseline = np.repeat(np.mean(true_vecs, axis=0, keepdims=True), true_vecs.shape[0], axis=0)
    baseline_diff = mean_baseline - true_vecs

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "aligned_best_val_loss": aligned_train["best_val_loss"],
        "shuffled_best_val_loss": shuffled_train["best_val_loss"],
        "aligned_final_train_loss": aligned_train["final_train_loss"],
        "shuffled_final_train_loss": shuffled_train["final_train_loss"],
        "aligned_val_drop": aligned_train["val_losses"][0] - aligned_train["val_losses"][-1],
        "shuffled_val_drop": shuffled_train["val_losses"][0] - shuffled_train["val_losses"][-1],
        "aligned_avg_percent_reward": aligned_eval["average_percent_reward"],
        "shuffled_avg_percent_reward": shuffled_eval["average_percent_reward"],
        "aligned_avg_regret": aligned_eval["average_regret"],
        "shuffled_avg_regret": shuffled_eval["average_regret"],
        "aligned_median_percent_reward": float(np.median(aligned_eval_arr)),
        "shuffled_median_percent_reward": float(np.median(shuffled_eval_arr)),
        "aligned_trimmed_mean_percent_reward_5pct": _trimmed_mean(aligned_eval_arr, frac=0.05),
        "shuffled_trimmed_mean_percent_reward_5pct": _trimmed_mean(shuffled_eval_arr, frac=0.05),
        "aligned_reward_recon_mse": aligned_recon["mse"],
        "shuffled_reward_recon_mse": shuffled_recon["mse"],
        "aligned_reward_recon_mae": aligned_recon["mae"],
        "shuffled_reward_recon_mae": shuffled_recon["mae"],
        "aligned_reward_recon_corr": aligned_recon["corr"],
        "shuffled_reward_recon_corr": shuffled_recon["corr"],
        "aligned_pred_reward_std": aligned_recon["pred_std"],
        "shuffled_pred_reward_std": shuffled_recon["pred_std"],
        "aligned_pred_reward_mean": aligned_recon["pred_mean"],
        "shuffled_pred_reward_mean": shuffled_recon["pred_mean"],
        "true_reward_std": float(np.std(true_vecs)),
        "true_reward_mean": float(np.mean(true_vecs)),
        "aligned_vs_shuffled_pred_corr": _safe_corr(aligned_pred, shuffled_pred),
        "delta_avg_percent_reward_aligned_minus_shuffled": (
            aligned_eval["average_percent_reward"] - shuffled_eval["average_percent_reward"]
        ),
        "delta_reward_recon_mse_shuffled_minus_aligned": (
            shuffled_recon["mse"] - aligned_recon["mse"]
        ),
        "delta_reward_recon_corr_aligned_minus_shuffled": (
            (aligned_recon["corr"] if aligned_recon["corr"] is not None else 0.0)
            - (shuffled_recon["corr"] if shuffled_recon["corr"] is not None else 0.0)
        ),
        "mean_baseline_reward_recon_mse": float(np.mean(baseline_diff ** 2)),
        "mean_baseline_reward_recon_mae": float(np.mean(np.abs(baseline_diff))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write alignment sanity summary json from run artifacts.")
    parser.add_argument("--runs-dir", required=True, help="Path containing aligned/shuffled run outputs.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing reward_infer.npz (defaults to aligned training dataset).",
    )
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if args.dataset_dir is None:
        dataset_dir = Path(_load_json(runs_dir / "aligned" / "sri_train_metrics.json")["dataset_dir"])
    else:
        dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    summary = build_summary(runs_dir, dataset_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
