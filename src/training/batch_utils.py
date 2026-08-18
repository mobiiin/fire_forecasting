"""Batch compatibility helpers for tuple and terrain-enabled mapping batches."""

from __future__ import annotations

from typing import Any


def unpack_batch(batch: Any):
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError("Mapping batch must contain 'x' and 'y'.")
        return batch["x"], batch["y"], {"terrain": batch.get("terrain"), "metadata": batch.get("metadata")}
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Batch must be a mapping or tuple/list containing x and y.")
    if len(batch) == 2:
        return batch[0], batch[1], {"terrain": None, "metadata": None}
    if len(batch) == 3:
        return batch[0], batch[1], {"terrain": None, "metadata": batch[2]}
    raise ValueError(f"Unsupported batch tuple length: {len(batch)}")
