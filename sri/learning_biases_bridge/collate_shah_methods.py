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


def _find_run_dirs(base_dir: Path) -> List[Path]:
    runs: List[Path] = []
    if not base_dir.exists():
        return runs
    for agent_dir in sorted(base_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        for seed_dir in sorted(agent_dir.iterdir()):
            if seed_dir.is_dir():
                runs.append(seed_dir)
    return runs


def _infer_beta_label(run_dir: Path, manifest: Optional[Dict[str, object]]) -> str:
    for part in run_dir.parts:
        if part.startswith("default_"):
            return "default"
        if part.startswith("none_"):
            return "none"
    if manifest is not None:
        cfg = manifest.get("config", {})
        if isinstance(cfg, dict) and cfg.get("beta") is None:
            return "none"
    return "default"


def collate(base_dir: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    for run_dir in _find_run_dirs(base_dir):
        manifest = _load_json(run_dir / "manifest.json")
        shah_methods_dir = run_dir / "shah_methods"
        if not shah_methods_dir.exists():
            continue
        for method_dir in sorted(shah_methods_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            metrics = _load_json(method_dir / "shah_metrics.json")
            if metrics is None:
                continue
            row = {
                "run_dir": str(run_dir),
                "method_dir": str(method_dir),
                "method": metrics.get("shah_method", method_dir.name),
                "agent": (manifest.get("agent") if manifest else run_dir.parent.name),
                "seed": (manifest.get("seed") if manifest else run_dir.name),
                "beta_label": _infer_beta_label(run_dir, manifest),
                "beta_value": (
                    manifest.get("config", {}).get("beta")
                    if manifest and isinstance(manifest.get("config"), dict)
                    else None
                ),
                "average_percent_reward": metrics.get("average_percent_reward"),
                "average_regret": metrics.get("average_regret"),
                "average_loss_on_test_walls": metrics.get("average_loss_on_test_walls"),
                "accuracy_on_test_walls": metrics.get("accuracy_on_test_walls"),
            }
            rows.append(row)

    grouped: Dict[Tuple[object, object, object], List[Dict[str, object]]] = {}
    for row in rows:
        key = (row["method"], row["beta_label"], row["agent"])
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        method, beta_label, agent = key
        group = grouped[key]
        mean, sem, n = _mean_and_sem(r.get("average_percent_reward") for r in group)
        reg_mean, reg_sem, reg_n = _mean_and_sem(r.get("average_regret") for r in group)
        summary_rows.append(
            {
                "method": method,
                "beta_label": beta_label,
                "agent": agent,
                "num_runs": len(group),
                "percent_reward_mean": mean,
                "percent_reward_sem": sem,
                "percent_reward_n": n,
                "regret_mean": reg_mean,
                "regret_sem": reg_sem,
                "regret_n": reg_n,
            }
        )

    return {
        "base_dir": str(base_dir),
        "num_rows": len(rows),
        "num_summary_rows": len(summary_rows),
        "rows": rows,
        "summary_by_method_condition": summary_rows,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate Shah-method metrics from bridge matrix runs."
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    payload = collate(base_dir)
    with (out / "shah_methods_collation.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    row_fields = [
        "run_dir",
        "method_dir",
        "method",
        "agent",
        "seed",
        "beta_label",
        "beta_value",
        "average_percent_reward",
        "average_regret",
        "average_loss_on_test_walls",
        "accuracy_on_test_walls",
    ]
    _write_csv(out / "shah_methods_rows.csv", payload["rows"], row_fields)

    summary_fields = [
        "method",
        "beta_label",
        "agent",
        "num_runs",
        "percent_reward_mean",
        "percent_reward_sem",
        "percent_reward_n",
        "regret_mean",
        "regret_sem",
        "regret_n",
    ]
    _write_csv(
        out / "shah_methods_by_condition.csv",
        payload["summary_by_method_condition"],
        summary_fields,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "base_dir": str(base_dir),
                "out": str(out),
                "num_rows": payload["num_rows"],
                "num_summary_rows": payload["num_summary_rows"],
                "rows_csv": str(out / "shah_methods_rows.csv"),
                "summary_csv": str(out / "shah_methods_by_condition.csv"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
