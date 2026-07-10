"""Input-adapter modules for sequence forecasting models."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)


class SequenceConv2dInputAdapter(nn.Module):
	"""Project each input timestep with the same 2D convolution."""

	def __init__(self, input_channels: int, embed_dim: int, kernel_size: int = 3) -> None:
		super().__init__()
		if input_channels <= 0:
			raise ValueError(f"input_channels must be positive, got {input_channels}.")
		if embed_dim <= 0:
			raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
		if kernel_size <= 0:
			raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
		self.input_channels = int(input_channels)
		self.embed_dim = int(embed_dim)
		padding = int(kernel_size) // 2
		self.proj = nn.Conv2d(self.input_channels, self.embed_dim, kernel_size=int(kernel_size), padding=padding)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Project ``(B, T, C, H, W)`` to ``(B, T, E, H, W)``."""

		if x.ndim != 5:
			raise ValueError(f"SequenceConv2dInputAdapter expects a 5D tensor, got {tuple(x.shape)}.")
		if x.shape[2] != self.input_channels:
			raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[2]}.")
		batch_size, time_steps, _, height, width = tuple(int(value) for value in x.shape)
		projected = self.proj(x.reshape(batch_size * time_steps, self.input_channels, height, width))
		return projected.reshape(batch_size, time_steps, self.embed_dim, height, width)


class IdentitySequenceInputAdapter(nn.Module):
	"""Return the canonical sequence input unchanged."""

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"IdentitySequenceInputAdapter expects a 5D tensor, got {tuple(x.shape)}.")
		return x


def adapt_input_for_architecture(x: torch.Tensor, architecture: str) -> torch.Tensor:
	"""Apply any architecture-specific input adaptation.

	The canonical dataset shape already matches ``st_mamba_lite``, so that path
	returns ``x`` unchanged.
	"""

	architecture_name = str(architecture).lower()
	if architecture_name == "st_mamba_lite":
		return x
	return x
