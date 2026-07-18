import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean_and_sem(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return mean, sem, n


def _parse_path_metadata(run_dir: Path) -> Dict[str, object]:
    beta_label = None
    path_agent = None
    trial_idx = None
    for part in run_dir.parts:
        if part.startswith("trial_"):
            try:
                trial_idx = int(part.split("_", 1)[1])
            except (IndexError, ValueError):
                pass
        elif part.startswith("default_"):
            beta_label = "default"
            path_agent = part.split("_", 1)[1] if "_" in part else None
        elif part.startswith("none_"):
            beta_label = "none"
            path_agent = part.split("_", 1)[1] if "_" in part else None
    return {
        "trial_idx": trial_idx,
        "beta_label_from_path": beta_label,
        "agent_from_path": path_agent,
    }


def _collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(root.rglob("sri_metrics.json")):
        eval_dir = metrics_path.parent
        if eval_dir.name == "last_checkpoint_eval":
            checkpoint = "last"
            run_dir = eval_dir.parent
        else:
            checkpoint = "best"
            run_dir = eval_dir

        sri = _load_json(metrics_path) or {}
        comparison = _load_json(eval_dir / "comparison.json") or {}
        manifest = _load_json(run_dir / "manifest.json") or {}
        train = _load_json(run_dir / "sri_train_metrics.json") or {}

        meta = _parse_path_metadata(run_dir)
        cfg = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
        manifest_beta = cfg.get("beta")
        beta_label_manifest = "none" if manifest_beta is None else "default"

        agent = comparison.get("agent") or manifest.get("agent") or meta["agent_from_path"]
        seed = comparison.get("seed")
        if seed is None:
            seed = manifest.get("seed")
        if seed is None:
            seed = run_dir.name if run_dir.name.isdigit() else None
        try:
            seed = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            pass

        row = {
            "root": str(root),
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "checkpoint": checkpoint,
            "trial_idx": meta["trial_idx"],
            "agent": agent,
            "seed": seed,
            "beta_label": meta["beta_label_from_path"] or beta_label_manifest,
            "beta_value": manifest_beta,
            "model_type": train.get("dem_encoder_type"),
            "demonstration_source": train.get("demonstration_source"),
            "epochs": train.get("epochs"),
            "best_val_loss": train.get("best_val_loss"),
            "final_train_loss": train.get("final_train_loss"),
            "final_val_loss": train.get("final_val_loss"),
            "sri_average_percent_reward": sri.get("average_percent_reward"),
            "sri_average_regret": sri.get("average_regret"),
            "sri_num_tasks": sri.get("num_tasks"),
            "shah_average_percent_reward": comparison.get("shah_given_rewards", {}).get("average_percent_reward"),
            "shah_average_regret": comparison.get("shah_given_rewards", {}).get("average_regret"),
            "delta_sri_minus_shah": comparison.get("delta", {}).get("sri_minus_shah"),
        }
        rows.append(row)
    return rows


def _summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, object, object], List[Dict[str, object]]] = {}
    for row in rows:
        key = (row.get("checkpoint"), row.get("beta_label"), row.get("agent"))
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(groups.keys()):
        checkpoint, beta_label, agent = key
        group = groups[key]

        sri_mean, sri_sem, sri_n = _mean_and_sem(
            _safe_float(r.get("sri_average_percent_reward")) for r in group
        )
        shah_mean, shah_sem, shah_n = _mean_and_sem(
            _safe_float(r.get("shah_average_percent_reward")) for r in group
        )
        delta_mean, delta_sem, delta_n = _mean_and_sem(
            _safe_float(r.get("delta_sri_minus_shah")) for r in group
        )
        val_loss_mean, val_loss_sem, val_loss_n = _mean_and_sem(
            _safe_float(r.get("best_val_loss")) for r in group
        )

        summary_rows.append(
            {
                "checkpoint": checkpoint,
                "beta_label": beta_label,
                "agent": agent,
                "num_runs": len(group),
                "sri_percent_reward_mean": sri_mean,
                "sri_percent_reward_sem": sri_sem,
                "sri_percent_reward_n": sri_n,
                "shah_percent_reward_mean": shah_mean,
                "shah_percent_reward_sem": shah_sem,
                "shah_percent_reward_n": shah_n,
                "delta_sri_minus_shah_mean": delta_mean,
                "delta_sri_minus_shah_sem": delta_sem,
                "delta_sri_minus_shah_n": delta_n,
                "best_val_loss_mean": val_loss_mean,
                "best_val_loss_sem": val_loss_sem,
                "best_val_loss_n": val_loss_n,
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collate learning_biases_bridge trial runs into tidy per-run and "
            "per-condition tables."
        )
    )
    parser.add_argument("--root", required=True, help="Root directory containing trial subdirectories.")
    parser.add_argument("--out", default=None, help="Output directory (defaults to --root).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root
    out.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(root)
    summary_rows = _summarize_rows(rows)

    payload = {
        "root": str(root),
        "num_rows": len(rows),
        "num_summary_rows": len(summary_rows),
        "rows": rows,
        "summary_by_condition": summary_rows,
    }
    with (out / "trial_collation.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    row_fields = [
        "root",
        "run_dir",
        "eval_dir",
        "checkpoint",
        "trial_idx",
        "agent",
        "seed",
        "beta_label",
        "beta_value",
        "model_type",
        "demonstration_source",
        "epochs",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "sri_average_percent_reward",
        "sri_average_regret",
        "sri_num_tasks",
        "shah_average_percent_reward",
        "shah_average_regret",
        "delta_sri_minus_shah",
    ]
    _write_csv(out / "trial_collation_rows.csv", rows, row_fields)

    summary_fields = [
        "checkpoint",
        "beta_label",
        "agent",
        "num_runs",
        "sri_percent_reward_mean",
        "sri_percent_reward_sem",
        "sri_percent_reward_n",
        "shah_percent_reward_mean",
        "shah_percent_reward_sem",
        "shah_percent_reward_n",
        "delta_sri_minus_shah_mean",
        "delta_sri_minus_shah_sem",
        "delta_sri_minus_shah_n",
        "best_val_loss_mean",
        "best_val_loss_sem",
        "best_val_loss_n",
    ]
    _write_csv(out / "trial_collation_by_condition.csv", summary_rows, summary_fields)

    print(
        json.dumps(
            {
                "status": "ok",
                "root": str(root),
                "out": str(out),
                "num_rows": len(rows),
                "num_summary_rows": len(summary_rows),
                "rows_csv": str(out / "trial_collation_rows.csv"),
                "summary_csv": str(out / "trial_collation_by_condition.csv"),
                "json": str(out / "trial_collation.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
