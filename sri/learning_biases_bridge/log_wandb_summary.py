import argparse
import json
import os
from typing import Dict, Iterable, List, Optional


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _group_by_agent(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        agent = row.get("agent")
        if not agent:
            continue
        out.setdefault(str(agent), []).append(row)
    return out


def _parse_tags(tags: str) -> List[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


def log_summary(args: argparse.Namespace) -> Dict[str, object]:
    import wandb

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = summary.get("rows", [])
    tags = _parse_tags(args.tags)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        group=args.group,
        tags=tags,
        job_type="learning_biases_bridge",
        config={
            "summary_path": os.path.abspath(args.summary),
            "base_dir": os.path.abspath(args.base_dir) if args.base_dir else None,
            "num_runs": summary.get("num_runs", 0),
        },
    )

    table = wandb.Table(
        columns=[
            "run_dir",
            "agent",
            "seed",
            "sri_average_percent_reward",
            "sri_average_regret",
            "shah_average_percent_reward",
            "shah_average_regret",
            "delta_sri_minus_shah",
        ]
    )
    for row in rows:
        table.add_data(
            row.get("run_dir"),
            row.get("agent"),
            row.get("seed"),
            row.get("sri_average_percent_reward"),
            row.get("sri_average_regret"),
            row.get("shah_average_percent_reward"),
            row.get("shah_average_regret"),
            row.get("delta_sri_minus_shah"),
        )

    logs: Dict[str, object] = {
        "bridge/num_runs": int(summary.get("num_runs", 0)),
        "bridge/runs_table": table,
    }
    logs["bridge/mean_sri_percent_reward"] = _mean(
        row.get("sri_average_percent_reward") for row in rows
    )
    logs["bridge/mean_sri_regret"] = _mean(
        row.get("sri_average_regret") for row in rows
    )
    logs["bridge/mean_shah_percent_reward"] = _mean(
        row.get("shah_average_percent_reward") for row in rows
    )
    logs["bridge/mean_shah_regret"] = _mean(
        row.get("shah_average_regret") for row in rows
    )
    logs["bridge/mean_delta_sri_minus_shah"] = _mean(
        row.get("delta_sri_minus_shah") for row in rows
    )
    wandb.log(logs)

    by_agent = _group_by_agent(rows)
    for agent, agent_rows in by_agent.items():
        wandb.log(
            {
                f"agent/{agent}/num_runs": int(len(agent_rows)),
                f"agent/{agent}/mean_sri_percent_reward": _mean(
                    r.get("sri_average_percent_reward") for r in agent_rows
                ),
                f"agent/{agent}/mean_shah_percent_reward": _mean(
                    r.get("shah_average_percent_reward") for r in agent_rows
                ),
                f"agent/{agent}/mean_delta_sri_minus_shah": _mean(
                    r.get("delta_sri_minus_shah") for r in agent_rows
                ),
            }
        )

    if args.base_dir:
        summary_json = os.path.join(args.base_dir, "summary.json")
        summary_csv = os.path.join(args.base_dir, "summary.csv")
        if os.path.exists(summary_json):
            wandb.save(summary_json, base_path=args.base_dir)
        if os.path.exists(summary_csv):
            wandb.save(summary_csv, base_path=args.base_dir)

    run.summary["num_runs"] = int(summary.get("num_runs", 0))
    run.summary["mean_delta_sri_minus_shah"] = logs["bridge/mean_delta_sri_minus_shah"]
    run.finish()

    return {
        "status": "ok",
        "project": args.project,
        "entity": args.entity,
        "num_runs": int(summary.get("num_runs", 0)),
        "run_url": run.url,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log learning-biases bridge summary to W&B.")
    parser.add_argument("--summary", required=True, help="Path to summary.json from aggregate_results.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags, e.g. 'bridge,policy,tf1'.",
    )
    parser.add_argument("--base-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = log_summary(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
