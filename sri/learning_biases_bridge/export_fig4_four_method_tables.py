import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROW_ORDER = [
    "Average",
    "Optimal",
    "Naive",
    "Sophisticated",
    "Myopic",
    "Overconfident",
    "Underconfident",
    "Boltzmann-Optimal",
    "Boltzmann-Naive",
    "Boltzmann-Sophisticated",
    "Boltzmann-Myopic",
    "Boltzmann-Overconfident",
    "Boltzmann-Underconfident",
]

ALGORITHMS = [
    "boltzmann_planner",
    "optimal_planner",
    "given_rewards",
    "joint_with_init",
    "sri_policy",
]

TRAINFREE_METHODS = {"boltzmann_planner", "optimal_planner", "joint_with_init"}


def _mean_sem(values: Iterable[float]) -> Tuple[Optional[float], Optional[float], int]:
    xs = list(values)
    n = len(xs)
    if n == 0:
        return None, None, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return mean, sem, n


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _agent_title(agent: str) -> str:
    return agent[:1].upper() + agent[1:]


def _row_name(agent: str, beta_label: str) -> Optional[str]:
    if not agent:
        return None
    base = _agent_title(agent.strip())
    if beta_label == "none":
        return base
    if beta_label == "default":
        return f"Boltzmann-{base}"
    return None


def _parse_beta_agent(name: str) -> Tuple[Optional[str], Optional[str]]:
    if name.startswith("default_"):
        return "default", name.split("_", 1)[1]
    if name.startswith("none_"):
        return "none", name.split("_", 1)[1]
    return None, None


def _init_values() -> Dict[str, Dict[str, List[float]]]:
    return {row: {alg: [] for alg in ALGORITHMS} for row in ROW_ORDER if row != "Average"}


def _collect_from_sri_runs(
    sri_rows_csvs: List[Path], checkpoint: str, values: Dict[str, Dict[str, List[float]]]
) -> None:
    for rows_csv in sri_rows_csvs:
        with rows_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("checkpoint") != checkpoint:
                    continue
                name = _row_name(str(row.get("agent", "")), str(row.get("beta_label", "")))
                if name is None or name not in values:
                    continue
                val = _safe_float(row.get("shah_average_percent_reward"))
                if val is None:
                    continue
                values[name]["given_rewards"].append(100.0 * val)
                sri_val = _safe_float(row.get("sri_average_percent_reward"))
                if sri_val is not None:
                    values[name]["sri_policy"].append(100.0 * sri_val)


def _collect_trainfree(
    trainfree_roots: List[Path], values: Dict[str, Dict[str, List[float]]]
) -> None:
    for root in trainfree_roots:
        for metrics_path in root.rglob("shah_metrics.json"):
            # Expected layout:
            # <root>/<method>/<beta_agent>/<seed>/shah_methods/<method>/shah_metrics.json
            if len(metrics_path.parts) < 7:
                continue
            method = metrics_path.parent.name
            if method not in TRAINFREE_METHODS:
                continue

            beta_agent_dir = metrics_path.parents[3]
            beta_label, agent = _parse_beta_agent(beta_agent_dir.name)
            if beta_label is None or agent is None:
                continue

            name = _row_name(agent, beta_label)
            if name is None or name not in values:
                continue

            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            val = payload.get("average_percent_reward")
            if val is None:
                continue
            values[name][method].append(100.0 * float(val))


def _build_average(values: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
    avg = {alg: [] for alg in ALGORITHMS}
    for row_name, rec in values.items():
        if row_name == "Average":
            continue
        for alg in ALGORITHMS:
            avg[alg].extend(rec[alg])
    return avg


def _write_table(path: Path, table: Dict[str, Dict[str, Optional[float]]]) -> None:
    fieldnames = ["Agent"] + ALGORITHMS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_name in ROW_ORDER:
            row = {"Agent": row_name}
            row.update(table[row_name])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export reward-means.csv and reward-sterrs.csv compatible with "
            "learning_biases/create_graphs.py for 5 methods: boltzmann_planner, "
            "optimal_planner, given_rewards, joint_with_init, sri_policy."
        )
    )
    parser.add_argument(
        "--sri-rows-csv",
        action="append",
        required=True,
        help="Path to trial_collation_rows.csv from SRI/policy runs. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--trainfree-root",
        action="append",
        required=True,
        help="Root directory of Shah train-free runs. Repeat for multiple roots.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    args = parser.parse_args()

    sri_rows_csvs = [Path(p).resolve() for p in args.sri_rows_csv]
    trainfree_roots = [Path(p).resolve() for p in args.trainfree_root]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    values = _init_values()
    _collect_from_sri_runs(sri_rows_csvs, args.checkpoint, values)
    _collect_trainfree(trainfree_roots, values)
    avg_values = _build_average(values)

    means: Dict[str, Dict[str, Optional[float]]] = {}
    sterrs: Dict[str, Dict[str, Optional[float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}

    for row_name in ROW_ORDER:
        means[row_name] = {}
        sterrs[row_name] = {}
        counts[row_name] = {}
        source = avg_values if row_name == "Average" else values[row_name]
        for alg in ALGORITHMS:
            mean, sem, n = _mean_sem(source[alg])
            means[row_name][alg] = mean
            sterrs[row_name][alg] = sem
            counts[row_name][alg] = n

    _write_table(out_dir / "reward-means.csv", means)
    _write_table(out_dir / "reward-sterrs.csv", sterrs)

    payload = {
        "checkpoint": args.checkpoint,
        "sri_rows_csv": [str(p) for p in sri_rows_csvs],
        "trainfree_roots": [str(p) for p in trainfree_roots],
        "counts": counts,
    }
    with (out_dir / "reward-counts.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": args.checkpoint,
                "out_dir": str(out_dir),
                "means_csv": str(out_dir / "reward-means.csv"),
                "sterrs_csv": str(out_dir / "reward-sterrs.csv"),
                "counts_json": str(out_dir / "reward-counts.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
