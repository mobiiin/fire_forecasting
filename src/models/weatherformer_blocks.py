"""Factorized transformer blocks for WeatherFormer-lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	from torch.utils.checkpoint import checkpoint  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	checkpoint = None

from src.models.earthformer_blocks import StochasticDepth
from src.models.st_mamba_lite_blocks import ChannelLayerNorm5D, _make_group_norm
from src.models.window_attention import WindowSelfAttention


class TemporalSelfAttention(nn.Module):
	"""Temporal attention applied independently at each spatial cell."""

	def __init__(self, dim: int, num_heads: int, attention_dropout: float = 0.0) -> None:
		super().__init__()
		if dim <= 0:
			raise ValueError(f"dim must be positive, got {dim}.")
		if num_heads <= 0:
			raise ValueError(f"num_heads must be positive, got {num_heads}.")
		if dim % num_heads != 0:
			raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
		self.dim = int(dim)
		self.num_heads = int(num_heads)
		self.attn = nn.MultiheadAttention(
			embed_dim=self.dim,
			num_heads=self.num_heads,
			dropout=float(attention_dropout),
			batch_first=True,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalSelfAttention expects a 5D tensor, got {tuple(x.shape)}.")
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
		if channels != self.dim:
			raise ValueError(f"TemporalSelfAttention expected channel dim {self.dim}, got {channels}.")
		sequence = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, time_steps, channels)
		attended, _ = self.attn(query=sequence, key=sequence, value=sequence, need_weights=False)
		return attended.reshape(batch_size, height, width, time_steps, channels).permute(0, 3, 4, 1, 2).contiguous()


class ChannelMLP3D(nn.Module):
	"""Pointwise Conv3d MLP over the channel dimension."""

	def __init__(self, channels: int, mlp_ratio: float, dropout: float = 0.0) -> None:
		super().__init__()
		hidden_channels = max(int(round(float(channels) * float(mlp_ratio))), 1)
		self.net = nn.Sequential(
			nn.Conv3d(channels, hidden_channels, kernel_size=1),
			nn.GELU(),
			nn.Dropout(float(dropout)),
			nn.Conv3d(hidden_channels, channels, kernel_size=1),
			nn.Dropout(float(dropout)),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"ChannelMLP3D expects a 5D tensor, got {tuple(x.shape)}.")
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.net(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class TemporalReadout2D(nn.Module):
	"""Collapse time into one 2D feature map per sample."""

	def __init__(self, channels: int, mode: str = "attention_pool") -> None:
		super().__init__()
		self.channels = int(channels)
		self.mode = str(mode).lower()
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"Unsupported temporal_readout: {mode!r}.")
		if self.mode == "attention_pool":
			self.scorer = nn.Conv2d(self.channels, 1, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalReadout2D expects a 5D tensor, got {tuple(x.shape)}.")
		if self.mode == "last":
			return x[:, -1]
		if self.mode == "mean":
			return torch.mean(x, dim=1)
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
		tokens = x.reshape(batch_size * time_steps, channels, height, width)
		scores = self.scorer(tokens).reshape(batch_size, time_steps, 1, height, width)
		weights = torch.softmax(scores, dim=1)
		return torch.sum(x * weights, dim=1)


class FactorizedWeatherFormerBlock(nn.Module):
	"""Temporal-then-spatial factorized transformer block."""

	def __init__(
		self,
		channels: int,
		num_heads: int,
		mlp_ratio: float,
		window_size: int,
		shifted_window: bool,
		use_global_tokens: bool,
		num_global_tokens: int,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		drop_path: float = 0.0,
		gradient_checkpointing: bool = False,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.gradient_checkpointing = bool(gradient_checkpointing)

		self.norm1 = ChannelLayerNorm5D(self.channels)
		self.temporal_attention = TemporalSelfAttention(self.channels, num_heads=int(num_heads), attention_dropout=float(attention_dropout))
		self.norm2 = ChannelLayerNorm5D(self.channels)
		self.spatial_window_attention = WindowSelfAttention(
			dim=self.channels,
			num_heads=int(num_heads),
			window_size=int(window_size),
			attention_dropout=float(attention_dropout),
			shift_size=int(window_size) // 2 if bool(shifted_window) else 0,
			use_global_tokens=bool(use_global_tokens),
			num_global_tokens=int(num_global_tokens),
		)
		self.norm3 = ChannelLayerNorm5D(self.channels)
		self.channel_mlp = ChannelMLP3D(self.channels, mlp_ratio=float(mlp_ratio), dropout=float(dropout))
		self.drop_path = StochasticDepth(float(drop_path))

	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.drop_path(self.temporal_attention(self.norm1(x)))
		x = x + self.drop_path(self.spatial_window_attention(self.norm2(x)))
		x = x + self.drop_path(self.channel_mlp(self.norm3(x)))
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FactorizedWeatherFormerBlock expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"FactorizedWeatherFormerBlock expected channel dim {self.channels}, got {int(x.shape[2])}.")
		if self.gradient_checkpointing and self.training and checkpoint is not None:
			return checkpoint(self._forward_impl, x, use_reentrant=False)
		return self._forward_impl(x)


__all__ = [
	"FactorizedWeatherFormerBlock",
	"TemporalReadout2D",
	"_make_group_norm",
]
