import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict:
    with np.load(path) as arr:
        return {key: arr[key] for key in arr.files}


def build_shuffled_dataset(clean_dir: Path, out_dir: Path, shuffle_seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(clean_dir / "manifest.json", out_dir / "manifest.json")
    shutil.copy2(clean_dir / "reward_infer.npz", out_dir / "reward_infer.npz")

    rng = np.random.default_rng(shuffle_seed)
    split_stats = {}
    for split in ("planner_train", "planner_val"):
        data = _load_npz(clean_dir / f"{split}.npz")
        rewards = data["rewards"]
        perm = rng.permutation(rewards.shape[0])
        data["rewards"] = rewards[perm]
        np.savez(out_dir / f"{split}.npz", **data)
        split_stats[split] = {"num_tasks": int(rewards.shape[0])}

    payload = {
        "clean_dir": str(clean_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "shuffle_seed": int(shuffle_seed),
        "splits": split_stats,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shuffled-label variant of a bridge dataset (planner train/val rewards only)."
    )
    parser.add_argument("--clean-dir", required=True, help="Directory with original manifest and planner/reward splits.")
    parser.add_argument("--out-dir", required=True, help="Output directory for shuffled dataset variant.")
    parser.add_argument("--shuffle-seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_shuffled_dataset(
        clean_dir=Path(args.clean_dir),
        out_dir=Path(args.out_dir),
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
