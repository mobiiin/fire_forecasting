"""Optional CUDA prefetcher for overlapping host-to-device transfer with compute."""

from __future__ import annotations

from typing import Any

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional training dependency
	torch = None


def _move_to_device(value: Any, device, non_blocking: bool):
	if torch is not None and torch.is_tensor(value):
		return value.to(device, non_blocking=non_blocking)
	if isinstance(value, tuple):
		return tuple(_move_to_device(item, device, non_blocking) for item in value)
	if isinstance(value, list):
		return [_move_to_device(item, device, non_blocking) for item in value]
	if isinstance(value, dict):
		return {key: _move_to_device(item, device, non_blocking) for key, item in value.items()}
	return value


class CUDAPrefetcher:
	"""Iterate a DataLoader while preloading the next batch on a CUDA stream."""

	def __init__(self, loader, device, non_blocking: bool = True) -> None:
		if torch is None:
			raise ImportError("PyTorch is required to use CUDAPrefetcher.")
		if getattr(device, "type", str(device)).lower() != "cuda":
			raise ValueError("CUDAPrefetcher requires a CUDA device.")
		self.loader = loader
		self.device = device
		self.non_blocking = bool(non_blocking)
		self.stream = torch.cuda.Stream(device=device)

	def __iter__(self):
		iterator = iter(self.loader)
		next_batch = None

		def preload():
			try:
				batch = next(iterator)
			except StopIteration:
				return None
			with torch.cuda.stream(self.stream):
				return _move_to_device(batch, self.device, self.non_blocking)

		next_batch = preload()
		while next_batch is not None:
			torch.cuda.current_stream(self.device).wait_stream(self.stream)
			current_batch = next_batch
			next_batch = preload()
			yield current_batch

	def __len__(self) -> int:
		return len(self.loader)
