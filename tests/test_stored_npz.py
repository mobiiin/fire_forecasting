from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.cache import _read_shard_shapes
from src.data.stored_npz import open_stored_npz_array, stored_npz_array_info


def test_open_stored_npz_array_memmaps_member_without_loading_archive(tmp_path: Path) -> None:
	shard_path = tmp_path / "shard.npz"
	x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
	y = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
	np.savez(shard_path, X=x, y=y)

	info = stored_npz_array_info(shard_path, "X")
	mapped = open_stored_npz_array(shard_path, "X")

	assert info.shape == x.shape
	assert info.dtype == x.dtype
	assert isinstance(mapped, np.memmap)
	assert mapped.shape == x.shape
	np.testing.assert_array_equal(mapped[1], x[1])


def test_cache_shape_reader_uses_stored_npz_metadata(tmp_path: Path) -> None:
	shard_path = tmp_path / "shard.npz"
	np.savez(
		shard_path,
		X=np.zeros((5, 2, 3, 4, 4), dtype=np.float32),
		y=np.zeros((5, 2, 4, 4), dtype=np.float32),
	)

	assert _read_shard_shapes(shard_path) == ((5, 2, 3, 4, 4), (5, 2, 4, 4))


def test_open_stored_npz_array_rejects_compressed_members(tmp_path: Path) -> None:
	shard_path = tmp_path / "compressed.npz"
	np.savez_compressed(shard_path, X=np.zeros((2, 3), dtype=np.float32))

	with pytest.raises(ValueError, match="compressed"):
		open_stored_npz_array(shard_path, "X")
