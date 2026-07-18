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
    ("given_rewards", "shah_average_percent_reward"),
    ("sri_policy", "sri_average_percent_reward"),
]


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


def _load_rows(path: Path, checkpoint: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("checkpoint") != checkpoint:
                continue
            rows.append(row)
    return rows


def _safe_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _write_table(path: Path, values: Dict[str, Dict[str, Optional[float]]]) -> None:
    fieldnames = ["Agent"] + [alg for alg, _ in ALGORITHMS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_name in ROW_ORDER:
            rec = {"Agent": row_name}
            rec.update(values.get(row_name, {}))
            writer.writerow(rec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export reward-means.csv and reward-sterrs.csv compatible with "
            "learning_biases/create_graphs.py, using two algorithms: "
            "given_rewards and sri_policy."
        )
    )
    parser.add_argument("--rows-csv", required=True, help="Path to trial_collation_rows.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for CSV tables")
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=["best", "last"],
        help="Which checkpoint rows to use from trial_collation_rows.csv",
    )
    args = parser.parse_args()

    rows_csv = Path(args.rows_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(rows_csv, checkpoint=args.checkpoint)

    by_row_alg: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        name = _row_name(str(row.get("agent", "")), str(row.get("beta_label", "")))
        if name is None:
            continue
        by_row_alg.setdefault(name, {alg: [] for alg, _ in ALGORITHMS})
        for alg, key in ALGORITHMS:
            v = _safe_float(row.get(key))
            if v is not None:
                by_row_alg[name][alg].append(100.0 * v)

    # "Average" matches Shah's top-left bar: aggregate over all 12 bias conditions.
    by_row_alg["Average"] = {alg: [] for alg, _ in ALGORITHMS}
    for name in ROW_ORDER:
        if name == "Average":
            continue
        if name not in by_row_alg:
            continue
        for alg, _ in ALGORITHMS:
            by_row_alg["Average"][alg].extend(by_row_alg[name][alg])

    means: Dict[str, Dict[str, Optional[float]]] = {}
    sterrs: Dict[str, Dict[str, Optional[float]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for name in ROW_ORDER:
        means[name] = {}
        sterrs[name] = {}
        counts[name] = {}
        alg_values = by_row_alg.get(name, {alg: [] for alg, _ in ALGORITHMS})
        for alg, _ in ALGORITHMS:
            mean, sem, n = _mean_sem(alg_values.get(alg, []))
            means[name][alg] = mean
            sterrs[name][alg] = sem
            counts[name][alg] = n

    _write_table(out_dir / "reward-means.csv", means)
    _write_table(out_dir / "reward-sterrs.csv", sterrs)

    with (out_dir / "reward-counts.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "rows_csv": str(rows_csv),
                "num_input_rows": len(rows),
                "counts": counts,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "means_csv": str(out_dir / "reward-means.csv"),
                "sterrs_csv": str(out_dir / "reward-sterrs.csv"),
                "counts_json": str(out_dir / "reward-counts.json"),
                "num_input_rows": len(rows),
                "checkpoint": args.checkpoint,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
