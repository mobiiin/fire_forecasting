"""Positional encoding utilities for WeatherFormer-lite."""

from __future__ import annotations

import math
from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None


class WeatherFormerPositionalEncoding(nn.Module):
	"""Add learnable time/space embeddings and optional Fourier space features."""

	def __init__(
		self,
		embed_dim: int,
		input_sequence_length: int,
		patch_size: int,
		use_time_pos_embed: bool,
		use_2d_space_pos_embed: bool,
		use_fourier_space_encoding: bool,
	) -> None:
		super().__init__()
		self.embed_dim = int(embed_dim)
		self.input_sequence_length = int(input_sequence_length)
		self.patch_size = int(patch_size)
		self.use_time_pos_embed = bool(use_time_pos_embed)
		self.use_2d_space_pos_embed = bool(use_2d_space_pos_embed)
		self.use_fourier_space_encoding = bool(use_fourier_space_encoding)

		if self.use_time_pos_embed:
			self.time_pos = nn.Parameter(torch.zeros(1, self.input_sequence_length, self.embed_dim, 1, 1))
			nn.init.trunc_normal_(self.time_pos, std=0.02)
		else:
			self.register_parameter("time_pos", None)

		if self.use_2d_space_pos_embed:
			self.space_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim, self.patch_size, self.patch_size))
			nn.init.trunc_normal_(self.space_pos, std=0.02)
		else:
			self.register_parameter("space_pos", None)

		if self.use_fourier_space_encoding:
			self.fourier_proj = nn.Conv2d(8, self.embed_dim, kernel_size=1)
		else:
			self.fourier_proj = None

	def _space_bias(self, x: torch.Tensor) -> torch.Tensor:
		if self.space_pos is None:
			return torch.zeros_like(x)
		height, width = tuple(int(value) for value in x.shape[-2:])
		bias = self.space_pos.reshape(1, self.embed_dim, self.patch_size, self.patch_size)
		if height != self.patch_size or width != self.patch_size:
			bias = F.interpolate(bias, size=(height, width), mode="bilinear", align_corners=False)
		return bias.unsqueeze(1)

	def _fourier_bias(self, x: torch.Tensor) -> torch.Tensor:
		if self.fourier_proj is None:
			return torch.zeros_like(x)
		_, _, _, height, width = tuple(int(value) for value in x.shape)
		y_coords = torch.linspace(-1.0, 1.0, steps=height, device=x.device, dtype=x.dtype)
		x_coords = torch.linspace(-1.0, 1.0, steps=width, device=x.device, dtype=x.dtype)
		yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
		features = torch.stack(
			[
				torch.sin(math.pi * xx),
				torch.cos(math.pi * xx),
				torch.sin(math.pi * yy),
				torch.cos(math.pi * yy),
				torch.sin(2.0 * math.pi * xx),
				torch.cos(2.0 * math.pi * xx),
				torch.sin(2.0 * math.pi * yy),
				torch.cos(2.0 * math.pi * yy),
			],
			dim=0,
		).unsqueeze(0)
		projected = self.fourier_proj(features)
		return projected.unsqueeze(1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"WeatherFormerPositionalEncoding expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.embed_dim:
			raise ValueError(f"WeatherFormerPositionalEncoding expected channel dim {self.embed_dim}, got {int(x.shape[2])}.")
		y = x
		if self.time_pos is not None:
			if int(x.shape[1]) != self.input_sequence_length:
				raise ValueError(
					"WeatherFormerPositionalEncoding expected the configured input sequence length. "
					f"Got T={int(x.shape[1])}, expected {self.input_sequence_length}."
				)
			y = y + self.time_pos
		if self.space_pos is not None:
			y = y + self._space_bias(x)
		if self.fourier_proj is not None:
			y = y + self._fourier_bias(x)
		return y
