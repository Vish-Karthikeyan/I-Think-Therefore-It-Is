"""Activation storage. Activations are captured in the same forward pass that produces the
answer (HFBackend capture hooks, last prompt token), and stored per layer in float16
shards keyed by the request's cache key."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, tuple[str, int]] = {}
        for f in sorted(self.root.glob("shard_*.json")):
            for i, k in enumerate(json.loads(f.read_text())):
                self._index[k] = (f.stem, i)
        self._cache: dict[str, dict] = {}

    def has(self, key: str) -> bool:
        return key in self._index

    def put(self, acts: dict[str, dict[int, np.ndarray]]) -> None:
        if not acts:
            return
        import os, time
        stem = f"shard_{time.time_ns()}_{os.getpid()}"  # unique across concurrent processes
        keys = list(acts)
        layers = sorted(next(iter(acts.values())))
        np.savez(self.root / f"{stem}.npz",
                 **{f"L{l}": np.stack([acts[k][l] for k in keys]).astype(np.float16) for l in layers})
        (self.root / f"{stem}.json").write_text(json.dumps(keys))
        for i, k in enumerate(keys):
            self._index[k] = (stem, i)

    def _shard(self, stem: str):
        if stem not in self._cache:
            self._cache[stem] = dict(np.load(self.root / f"{stem}.npz"))
        return self._cache[stem]

    def layers(self) -> list[int]:
        if not self._index:
            return []
        stem = next(iter(self._index.values()))[0]
        return sorted(int(k[1:]) for k in self._shard(stem))

    def get(self, keys: list[str], layer: int) -> np.ndarray:
        """Stack activations for `keys` at `layer` -> [n, d] float32 (NaN rows if missing)."""
        rows = []
        d = None
        for k in keys:
            loc = self._index.get(k)
            if loc is None:
                rows.append(None)
                continue
            v = self._shard(loc[0])[f"L{layer}"][loc[1]]
            d = v.shape[0]
            rows.append(v)
        if d is None:
            raise KeyError("no activations found for these keys")
        return np.stack([r if r is not None else np.full(d, np.nan, np.float16) for r in rows]).astype(np.float32)
