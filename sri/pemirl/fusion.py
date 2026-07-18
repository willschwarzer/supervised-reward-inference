from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

import numpy as np


class KeyedFusionBuffer:
    """Task-keyed replay buffer matching RamFusionDistrCustom semantics."""

    def __init__(self, buffer_size: int = 100, subsample_ratio: float = 0.5, seed: int = 0) -> None:
        self.buffer_size = int(buffer_size)
        self.subsample_ratio = float(subsample_ratio)
        self._rng = np.random.default_rng(seed)
        self._buffer: dict[Hashable, list[Any]] = defaultdict(list)

    def add_paths(self, keyed_paths: dict[Hashable, list[Any]], subsample: bool = True) -> None:
        for key, paths in keyed_paths.items():
            keep = paths
            if subsample and len(paths) > 0:
                n_keep = max(1, int(len(paths) * self.subsample_ratio))
                idxs = self._rng.choice(len(paths), size=n_keep, replace=False)
                keep = [paths[i] for i in idxs]
            buf = self._buffer[key]
            buf.extend(keep)
            overflow = len(buf) - self.buffer_size
            while overflow > 0 and len(buf) > 0:
                # Matches old behavior: preferentially drop older entries.
                probs = np.arange(len(buf), dtype=np.float64) + 1.0
                probs /= probs.sum()
                drop = int(self._rng.choice(np.arange(len(buf)), p=probs))
                buf.pop(drop)
                overflow -= 1

    def sample_paths(self, keys: list[Hashable], n: int) -> dict[Hashable, list[Any]] | None:
        ret: dict[Hashable, list[Any]] = {}
        for key in keys:
            buf = self._buffer.get(key, [])
            if len(buf) == 0:
                return None
            idxs = self._rng.integers(0, len(buf), size=n)
            ret[key] = [buf[i] for i in idxs]
        return ret

    def __len__(self) -> int:
        return int(sum(len(v) for v in self._buffer.values()))
