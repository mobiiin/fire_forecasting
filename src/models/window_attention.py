"""Windowed attention helpers for WeatherFormer-lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
	"""Partition ``(B, H, W, C)`` into flattened windows."""

	if x.ndim != 4:
		raise ValueError(f"window_partition expects a 4D tensor, got {tuple(x.shape)}.")
	batch_size, height, width, channels = tuple(int(value) for value in x.shape)
	if height % int(window_size) != 0 or width % int(window_size) != 0:
		raise ValueError(f"window_partition requires H/W divisible by window_size={window_size}, got H={height}, W={width}.")
	windows = x.reshape(
		batch_size,
		height // int(window_size),
		int(window_size),
		width // int(window_size),
		int(window_size),
		channels,
	)
	windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
	return windows.reshape(-1, int(window_size) * int(window_size), channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
	"""Reverse ``window_partition`` back to ``(B, H, W, C)``."""

	if windows.ndim != 3:
		raise ValueError(f"window_reverse expects a 3D tensor, got {tuple(windows.shape)}.")
	if height % int(window_size) != 0 or width % int(window_size) != 0:
		raise ValueError(f"window_reverse requires H/W divisible by window_size={window_size}, got H={height}, W={width}.")
	num_windows_per_sample = (int(height) // int(window_size)) * (int(width) // int(window_size))
	if int(windows.shape[0]) % num_windows_per_sample != 0:
		raise ValueError(
			"window_reverse received a window count that is incompatible with the target shape. "
			f"windows={tuple(windows.shape)} height={height} width={width} window_size={window_size}."
		)
	batch_size = int(windows.shape[0]) // num_windows_per_sample
	channels = int(windows.shape[-1])
	x = windows.reshape(
		batch_size,
		int(height) // int(window_size),
		int(width) // int(window_size),
		int(window_size),
		int(window_size),
		channels,
	)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
	return x.reshape(batch_size, int(height), int(width), channels)


def cyclic_shift(x: torch.Tensor, shift_size: int) -> torch.Tensor:
	"""Cyclically shift spatial dimensions of channel-last tensors."""

	if x.ndim < 3:
		raise ValueError(f"cyclic_shift expects at least 3 dims, got {tuple(x.shape)}.")
	if int(shift_size) == 0:
		return x
	return torch.roll(x, shifts=(-int(shift_size), -int(shift_size)), dims=(-3, -2))


class WindowSelfAttention(nn.Module):
	"""Local spatial self-attention over fixed-size windows."""

	def __init__(
		self,
		dim: int,
		num_heads: int,
		window_size: int,
		attention_dropout: float = 0.0,
		shift_size: int = 0,
		use_global_tokens: bool = False,
		num_global_tokens: int = 0,
	) -> None:
		super().__init__()
		if dim <= 0:
			raise ValueError(f"dim must be positive, got {dim}.")
		if num_heads <= 0:
			raise ValueError(f"num_heads must be positive, got {num_heads}.")
		if dim % num_heads != 0:
			raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
		if window_size <= 0:
			raise ValueError(f"window_size must be positive, got {window_size}.")
		if shift_size < 0 or shift_size >= window_size:
			raise ValueError(f"shift_size must be in [0, window_size), got shift_size={shift_size} window_size={window_size}.")
		self.dim = int(dim)
		self.num_heads = int(num_heads)
		self.window_size = int(window_size)
		self.shift_size = int(shift_size)
		self.use_global_tokens = bool(use_global_tokens)
		self.num_global_tokens = int(num_global_tokens)

		self.attn = nn.MultiheadAttention(
			embed_dim=self.dim,
			num_heads=self.num_heads,
			dropout=float(attention_dropout),
			batch_first=True,
		)
		if self.use_global_tokens:
			if self.num_global_tokens <= 0:
				raise ValueError("num_global_tokens must be positive when use_global_tokens=true.")
			self.global_tokens = nn.Parameter(torch.zeros(1, self.num_global_tokens, self.dim))
			self.global_update = nn.MultiheadAttention(
				embed_dim=self.dim,
				num_heads=self.num_heads,
				dropout=float(attention_dropout),
				batch_first=True,
			)
			nn.init.trunc_normal_(self.global_tokens, std=0.02)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"WindowSelfAttention expects a 5D tensor, got {tuple(x.shape)}.")
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
		if channels != self.dim:
			raise ValueError(f"WindowSelfAttention expected channel dim {self.dim}, got {channels}.")
		if height % self.window_size != 0 or width % self.window_size != 0:
			raise ValueError(
				f"WindowSelfAttention requires H/W divisible by window_size={self.window_size}, got H={height}, W={width}."
			)

		x_4d = x.permute(0, 1, 3, 4, 2).reshape(batch_size * time_steps, height, width, channels)
		if self.shift_size > 0:
			x_4d = cyclic_shift(x_4d, shift_size=self.shift_size)

		windows = window_partition(x_4d, window_size=self.window_size)
		keys = windows
		values = windows
		if self.use_global_tokens:
			pooled = x_4d.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
			global_tokens = self.global_tokens.expand(batch_size * time_steps, -1, -1)
			global_context, _ = self.global_update(query=global_tokens, key=pooled, value=pooled, need_weights=False)
			num_windows_per_frame = (height // self.window_size) * (width // self.window_size)
			global_context = global_context.unsqueeze(1).expand(-1, num_windows_per_frame, -1, -1).reshape(
				batch_size * time_steps * num_windows_per_frame,
				self.num_global_tokens,
				channels,
			)
			keys = torch.cat([windows, global_context], dim=1)
			values = torch.cat([windows, global_context], dim=1)

		attended, _ = self.attn(query=windows, key=keys, value=values, need_weights=False)
		merged = window_reverse(attended, window_size=self.window_size, height=height, width=width)
		if self.shift_size > 0:
			merged = torch.roll(merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
		return merged.reshape(batch_size, time_steps, height, width, channels).permute(0, 1, 4, 2, 3).contiguous()
