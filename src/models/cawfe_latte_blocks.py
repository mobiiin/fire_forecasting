"""Shared blocks for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

from src.models.st_mamba_lite_blocks import _make_group_norm


def apply_per_timestep(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
	"""Apply a 2D module independently to every timestep."""

	if x.ndim != 5:
		raise ValueError(f"apply_per_timestep expects (B, T, C, H, W), got {tuple(x.shape)}.")
	batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
	y = module(x.reshape(batch_size * time_steps, channels, height, width))
	return y.reshape(batch_size, time_steps, int(y.shape[1]), int(y.shape[2]), int(y.shape[3]))


class SequenceConvBlock(nn.Module):
	"""Small per-timestep Conv2d block for sequence tensors."""

	def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
		super().__init__()
		padding = int(kernel_size) // 2
		self.net = nn.Sequential(
			nn.Conv2d(int(input_channels), int(output_channels), kernel_size=int(kernel_size), padding=padding),
			_make_group_norm(int(output_channels)),
			nn.GELU(),
			nn.Conv2d(int(output_channels), int(output_channels), kernel_size=3, padding=1),
			_make_group_norm(int(output_channels)),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return apply_per_timestep(x, self.net)


class TemporalReadout(nn.Module):
	"""Collapse ``(B, T, C, H, W)`` to ``(B, C, H, W)``."""

	def __init__(self, channels: int, mode: str = "attention_pool") -> None:
		super().__init__()
		self.channels = int(channels)
		self.mode = str(mode).lower()
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"Unsupported temporal_readout: {mode!r}.")
		if self.mode == "attention_pool":
			self.scorer = nn.Conv2d(self.channels, 1, kernel_size=1)
		self.last_weights: torch.Tensor | None = None

	def forward(self, x: torch.Tensor, return_weights: bool = False):
		if x.ndim != 5:
			raise ValueError(f"TemporalReadout expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"TemporalReadout expected channel dim {self.channels}, got {int(x.shape[2])}.")
		if self.mode == "last":
			weights = None
			out = x[:, -1]
		elif self.mode == "mean":
			weights = torch.full(
				(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]),
				1.0 / float(x.shape[1]),
				dtype=x.dtype,
				device=x.device,
			)
			out = torch.mean(x, dim=1)
		else:
			batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
			scores = self.scorer(x.reshape(batch_size * time_steps, channels, height, width))
			weights = torch.softmax(scores.reshape(batch_size, time_steps, 1, height, width), dim=1)
			out = torch.sum(x * weights, dim=1)
		self.last_weights = weights
		if return_weights:
			return out, weights
		return out


__all__ = ["SequenceConvBlock", "TemporalReadout", "apply_per_timestep", "_make_group_norm"]
