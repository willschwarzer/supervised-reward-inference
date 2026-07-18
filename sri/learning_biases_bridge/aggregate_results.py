import argparse
import csv
import json
import os
from typing import Dict, List


def _find_result_dirs(base_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(base_dir):
        return out
    for agent in sorted(os.listdir(base_dir)):
        agent_dir = os.path.join(base_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in sorted(os.listdir(agent_dir)):
            run_dir = os.path.join(agent_dir, seed)
            if os.path.isdir(run_dir):
                out.append(run_dir)
    return out


def aggregate(base_dir: str, out_dir: str) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for run_dir in _find_result_dirs(base_dir):
        sri_path = os.path.join(run_dir, "sri_metrics.json")
        comp_path = os.path.join(run_dir, "comparison.json")
        if not os.path.exists(sri_path):
            continue

        with open(sri_path, "r", encoding="utf-8") as f:
            sri = json.load(f)
        comp = None
        if os.path.exists(comp_path):
            with open(comp_path, "r", encoding="utf-8") as f:
                comp = json.load(f)

        row = {
            "run_dir": run_dir,
            "agent": comp.get("agent") if comp else None,
            "seed": comp.get("seed") if comp else None,
            "sri_average_percent_reward": sri.get("average_percent_reward"),
            "sri_average_regret": sri.get("average_regret"),
            "shah_average_percent_reward": None,
            "shah_average_regret": None,
            "delta_sri_minus_shah": None,
        }
        if comp and "shah_given_rewards" in comp:
            row["shah_average_percent_reward"] = comp["shah_given_rewards"]["average_percent_reward"]
            row["shah_average_regret"] = comp["shah_given_rewards"]["average_regret"]
            row["delta_sri_minus_shah"] = comp.get("delta", {}).get("sri_minus_shah")
        rows.append(row)

    summary = {
        "num_runs": len(rows),
        "rows": rows,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = os.path.join(out_dir, "summary.csv")
    fieldnames = [
        "run_dir",
        "agent",
        "seed",
        "sri_average_percent_reward",
        "sri_average_regret",
        "shah_average_percent_reward",
        "shah_average_regret",
        "delta_sri_minus_shah",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate bridge run metrics into summary files")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate(args.base_dir, args.out)
    print(json.dumps({"num_runs": summary["num_runs"], "out": os.path.abspath(args.out)}, indent=2))


if __name__ == "__main__":
    main()
