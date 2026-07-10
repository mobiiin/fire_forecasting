"""Simplified Earthformer-inspired building blocks."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
	from torch.utils.checkpoint import checkpoint  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, ModuleList=list)
	F = None
	checkpoint = None


class StochasticDepth(nn.Module):
	"""Per-sample residual-path dropout."""

	def __init__(self, drop_prob: float = 0.0) -> None:
		super().__init__()
		if not 0.0 <= float(drop_prob) < 1.0:
			raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")
		self.drop_prob = float(drop_prob)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.drop_prob == 0.0 or not self.training:
			return x
		keep_prob = 1.0 - self.drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()
		return x * random_tensor / keep_prob


class FeedForward(nn.Module):
	"""Transformer MLP block."""

	def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
		super().__init__()
		hidden_dim = max(int(round(float(dim) * float(mlp_ratio))), 1)
		self.net = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(float(dropout)),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(float(dropout)),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class PatchMerge2D(nn.Module):
	"""Merge non-overlapping 2x2 spatial neighborhoods."""

	def __init__(self, in_dim: int, out_dim: int) -> None:
		super().__init__()
		self.in_dim = int(in_dim)
		self.out_dim = int(out_dim)
		self.norm = nn.LayerNorm(self.in_dim * 4)
		self.reduction = nn.Linear(self.in_dim * 4, self.out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"PatchMerge2D expects a 5D tensor, got {tuple(x.shape)}.")
		batch_size, time_steps, height, width, channels = tuple(int(value) for value in x.shape)
		if channels != self.in_dim:
			raise ValueError(f"PatchMerge2D expected channel dim {self.in_dim}, got {channels}.")
		if height % 2 != 0 or width % 2 != 0:
			raise ValueError(f"PatchMerge2D requires even H/W, got H={height}, W={width}.")
		x00 = x[:, :, 0::2, 0::2, :]
		x01 = x[:, :, 0::2, 1::2, :]
		x10 = x[:, :, 1::2, 0::2, :]
		x11 = x[:, :, 1::2, 1::2, :]
		merged = torch.cat([x00, x01, x10, x11], dim=-1)
		return self.reduction(self.norm(merged))


class TemporalReadout(nn.Module):
	"""Collapse time for each spatial cell."""

	def __init__(self, dim: int, num_heads: int, mode: str) -> None:
		super().__init__()
		self.dim = int(dim)
		self.mode = str(mode).lower()
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"Unsupported temporal_readout: {mode!r}.")
		if self.mode == "attention_pool":
			self.norm = nn.LayerNorm(self.dim)
			self.query = nn.Parameter(torch.zeros(1, 1, self.dim))
			self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=int(num_heads), batch_first=True)
			nn.init.trunc_normal_(self.query, std=0.02)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalReadout expects a 5D tensor, got {tuple(x.shape)}.")
		if self.mode == "last":
			return x[:, -1]
		if self.mode == "mean":
			return torch.mean(x, dim=1)

		batch_size, time_steps, height, width, channels = tuple(int(value) for value in x.shape)
		tokens = self.norm(x).permute(0, 2, 3, 1, 4).reshape(batch_size * height * width, time_steps, channels)
		query = self.query.expand(batch_size * height * width, -1, -1)
		pooled, _ = self.attn(query=query, key=tokens, value=tokens, need_weights=False)
		return pooled.reshape(batch_size, height, width, channels)


class ConvGELUBlock(nn.Module):
	"""Small spatial decoder block."""

	def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.GELU(),
			nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class AxialCuboidAttentionBlock(nn.Module):
	"""Axial attention over time, height, and width."""

	def __init__(
		self,
		dim: int,
		num_heads: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		drop_path: float = 0.0,
		use_global_vectors: bool = False,
		num_global_vectors: int = 0,
		gradient_checkpointing: bool = False,
	) -> None:
		super().__init__()
		if dim <= 0:
			raise ValueError(f"dim must be positive, got {dim}.")
		if num_heads <= 0:
			raise ValueError(f"num_heads must be positive, got {num_heads}.")
		if dim % num_heads != 0:
			raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
		self.dim = int(dim)
		self.num_heads = int(num_heads)
		self.use_global_vectors = bool(use_global_vectors)
		self.num_global_vectors = int(num_global_vectors)
		self.gradient_checkpointing = bool(gradient_checkpointing)

		self.temporal_norm = nn.LayerNorm(self.dim)
		self.height_norm = nn.LayerNorm(self.dim)
		self.width_norm = nn.LayerNorm(self.dim)
		self.ffn_norm = nn.LayerNorm(self.dim)

		self.temporal_attn = nn.MultiheadAttention(
			embed_dim=self.dim,
			num_heads=self.num_heads,
			dropout=float(attention_dropout),
			batch_first=True,
		)
		self.height_attn = nn.MultiheadAttention(
			embed_dim=self.dim,
			num_heads=self.num_heads,
			dropout=float(attention_dropout),
			batch_first=True,
		)
		self.width_attn = nn.MultiheadAttention(
			embed_dim=self.dim,
			num_heads=self.num_heads,
			dropout=float(attention_dropout),
			batch_first=True,
		)
		self.ffn = FeedForward(self.dim, mlp_ratio=float(mlp_ratio), dropout=float(dropout))
		self.drop_path = StochasticDepth(float(drop_path))

		if self.use_global_vectors:
			if self.num_global_vectors <= 0:
				raise ValueError("num_global_vectors must be positive when use_global_vectors=true.")
			self.global_tokens = nn.Parameter(torch.zeros(1, self.num_global_vectors, self.dim))
			self.global_update = nn.MultiheadAttention(
				embed_dim=self.dim,
				num_heads=self.num_heads,
				dropout=float(attention_dropout),
				batch_first=True,
			)
			nn.init.trunc_normal_(self.global_tokens, std=0.02)

	def _temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, time_steps, height, width, channels = tuple(int(value) for value in x.shape)
		normalized = self.temporal_norm(x)
		queries = normalized.permute(0, 2, 3, 1, 4).reshape(batch_size * height * width, time_steps, channels)
		keys = queries
		values = queries
		if self.use_global_vectors:
			pooled = normalized.mean(dim=(2, 3))
			global_tokens = self.global_tokens.expand(batch_size, -1, -1)
			global_context, _ = self.global_update(query=global_tokens, key=pooled, value=pooled, need_weights=False)
			global_context = global_context[:, None, None, :, :].expand(batch_size, height, width, -1, -1)
			global_context = global_context.reshape(batch_size * height * width, self.num_global_vectors, channels)
			keys = torch.cat([queries, global_context], dim=1)
			values = torch.cat([values, global_context], dim=1)
		attended, _ = self.temporal_attn(query=queries, key=keys, value=values, need_weights=False)
		return attended.reshape(batch_size, height, width, time_steps, channels).permute(0, 3, 1, 2, 4)

	def _height_attention(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, time_steps, height, width, channels = tuple(int(value) for value in x.shape)
		normalized = self.height_norm(x)
		queries = normalized.permute(0, 1, 3, 2, 4).reshape(batch_size * time_steps * width, height, channels)
		attended, _ = self.height_attn(query=queries, key=queries, value=queries, need_weights=False)
		return attended.reshape(batch_size, time_steps, width, height, channels).permute(0, 1, 3, 2, 4)

	def _width_attention(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, time_steps, height, width, channels = tuple(int(value) for value in x.shape)
		normalized = self.width_norm(x)
		queries = normalized.reshape(batch_size * time_steps * height, width, channels)
		attended, _ = self.width_attn(query=queries, key=queries, value=queries, need_weights=False)
		return attended.reshape(batch_size, time_steps, height, width, channels)

	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.drop_path(self._temporal_attention(x))
		x = x + self.drop_path(self._height_attention(x))
		x = x + self.drop_path(self._width_attention(x))
		x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"AxialCuboidAttentionBlock expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[-1]) != self.dim:
			raise ValueError(f"Expected channel dim {self.dim}, got {x.shape[-1]}.")
		if self.gradient_checkpointing and self.training and checkpoint is not None:
			return checkpoint(self._forward_impl, x, use_reentrant=False)
		return self._forward_impl(x)
