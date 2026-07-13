"""Vertical atmospheric encoder for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

from src.models.cawfe_latte_blocks import SequenceConvBlock, apply_per_timestep


class VerticalAtmosphereEncoder(nn.Module):
	"""Encode raw atmospheric channels as vertical-level tokens."""

	def __init__(
		self,
		num_levels: int,
		vars_per_level: int,
		atm_embed_dim: int,
		encoder_type: str = "attention",
		num_heads: int = 4,
		num_layers: int = 1,
		dropout: float = 0.0,
		pool_mode: str = "attention_pool",
		attention_chunk_size: int = 8192,
	) -> None:
		super().__init__()
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.atm_embed_dim = int(atm_embed_dim)
		self.encoder_type = str(encoder_type).lower()
		self.pool_mode = str(pool_mode).lower()
		self.attention_chunk_size = int(attention_chunk_size)
		self.input_channels = self.num_levels * self.vars_per_level
		if self.num_levels <= 0 or self.vars_per_level <= 0 or self.atm_embed_dim <= 0:
			raise ValueError("num_levels, vars_per_level, and atm_embed_dim must be positive.")
		if self.encoder_type not in {"attention", "mlp", "conv1d", "flatten_conv"}:
			raise ValueError(f"Unsupported vertical_encoder_type: {encoder_type!r}.")

		if self.encoder_type == "attention":
			if self.atm_embed_dim % int(num_heads) != 0:
				raise ValueError(f"atm_embed_dim={self.atm_embed_dim} must be divisible by num_heads={num_heads}.")
			self.var_projection = nn.Linear(self.vars_per_level, self.atm_embed_dim)
			self.vertical_pos_embed = nn.Parameter(torch.zeros(1, self.num_levels, self.atm_embed_dim))
			layer = nn.TransformerEncoderLayer(
				d_model=self.atm_embed_dim,
				nhead=int(num_heads),
				dim_feedforward=max(self.atm_embed_dim * 2, 1),
				dropout=float(dropout),
				activation="gelu",
				batch_first=True,
				norm_first=True,
			)
			self.vertical_transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))
			if self.pool_mode == "attention_pool":
				self.level_pool = nn.Linear(self.atm_embed_dim, 1)
			elif self.pool_mode != "mean":
				raise ValueError(f"Unsupported vertical pool_mode: {pool_mode!r}.")
			nn.init.trunc_normal_(self.vertical_pos_embed, std=0.02)
		elif self.encoder_type in {"mlp", "flatten_conv"}:
			self.flat_encoder = SequenceConvBlock(self.input_channels, self.atm_embed_dim, kernel_size=1)
		else:
			self.level_conv = nn.Sequential(
				nn.Conv1d(self.vars_per_level, self.atm_embed_dim, kernel_size=3, padding=1),
				nn.GELU(),
				nn.Conv1d(self.atm_embed_dim, self.atm_embed_dim, kernel_size=3, padding=1),
				nn.GELU(),
			)
			self.level_pool = nn.Linear(self.atm_embed_dim, 1)

	def _apply_vertical_transformer(self, tokens: torch.Tensor) -> torch.Tensor:
		chunk_size = int(self.attention_chunk_size)
		if chunk_size <= 0 or int(tokens.shape[0]) <= chunk_size:
			return self.vertical_transformer(tokens)
		return torch.cat(
			[
				self.vertical_transformer(tokens[start : start + chunk_size])
				for start in range(0, int(tokens.shape[0]), chunk_size)
			],
			dim=0,
		)

	def forward(self, x_atm_raw: torch.Tensor) -> torch.Tensor:
		if x_atm_raw.ndim != 5:
			raise ValueError(f"VerticalAtmosphereEncoder expects (B, T, C, H, W), got {tuple(x_atm_raw.shape)}.")
		if int(x_atm_raw.shape[2]) != self.input_channels:
			raise ValueError(f"Expected {self.input_channels} atmospheric channels, got {int(x_atm_raw.shape[2])}.")
		if self.encoder_type in {"mlp", "flatten_conv"}:
			return self.flat_encoder(x_atm_raw)

		batch_size, time_steps, _, height, width = tuple(int(value) for value in x_atm_raw.shape)
		x_levels = x_atm_raw.reshape(batch_size, time_steps, self.num_levels, self.vars_per_level, height, width)
		tokens = x_levels.permute(0, 1, 4, 5, 2, 3).reshape(batch_size * time_steps * height * width, self.num_levels, self.vars_per_level)
		if self.encoder_type == "attention":
			tokens = self.var_projection(tokens) + self.vertical_pos_embed
			tokens = self._apply_vertical_transformer(tokens)
			if self.pool_mode == "attention_pool":
				weights = torch.softmax(self.level_pool(tokens), dim=1)
				pooled = torch.sum(tokens * weights, dim=1)
			else:
				pooled = torch.mean(tokens, dim=1)
		else:
			tokens = self.level_conv(tokens.transpose(1, 2)).transpose(1, 2)
			weights = torch.softmax(self.level_pool(tokens), dim=1)
			pooled = torch.sum(tokens * weights, dim=1)
		return pooled.reshape(batch_size, time_steps, height, width, self.atm_embed_dim).permute(0, 1, 4, 2, 3).contiguous()


class FlatAtmosphereFallback(nn.Module):
	"""Ablation fallback that treats atmospheric channels as a flat image tensor."""

	def __init__(self, input_channels: int, output_channels: int) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.encoder = SequenceConvBlock(self.input_channels, int(output_channels), kernel_size=1)

	def forward(self, x_atm_raw: torch.Tensor) -> torch.Tensor:
		if int(x_atm_raw.shape[2]) != self.input_channels:
			raise ValueError(f"Expected {self.input_channels} atmospheric channels, got {int(x_atm_raw.shape[2])}.")
		return self.encoder(x_atm_raw)


__all__ = ["FlatAtmosphereFallback", "VerticalAtmosphereEncoder"]
