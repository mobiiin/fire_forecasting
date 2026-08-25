"""New CAWFE-Latte encoder and fusion stem.

This is intentionally not a complete forecasting model yet. It implements only
atmosphere, wind, fire/fuel, flux/energy encoders and fire-query modality fusion.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from typing import Any

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None


def _group_count(channels: int, preferred: int = 8) -> int:
	for groups in range(min(preferred, channels), 0, -1):
		if channels % groups == 0:
			return groups
	return 1


def _as_int_list(values: Sequence[Any] | None, default: Sequence[int]) -> list[int]:
	if values is None:
		return [int(value) for value in default]
	return [int(value) for value in values]


def _as_str_list(values: Sequence[Any] | None) -> list[str]:
	if values is None:
		return []
	return [str(value).lower() for value in values]


def _manual_multihead_attention(
	module: nn.MultiheadAttention,
	query: torch.Tensor,
	key: torch.Tensor,
	value: torch.Tensor,
	*,
	training: bool,
	need_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
	"""Small-sequence MHA using explicit matmul/softmax instead of SDPA kernels."""
	if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
		raise ValueError("Manual MHA expects batch-first B,L,D tensors.")
	if key.shape != value.shape:
		raise ValueError(f"Manual MHA key/value shapes differ: key={tuple(key.shape)} value={tuple(value.shape)}")
	embed_dim = int(module.embed_dim)
	num_heads = int(module.num_heads)
	head_dim = embed_dim // num_heads
	if embed_dim % num_heads != 0:
		raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
	if int(query.shape[-1]) != embed_dim or int(key.shape[-1]) != embed_dim:
		raise ValueError(f"Manual MHA expected D={embed_dim}, got query={tuple(query.shape)} key={tuple(key.shape)}")
	weight = module.in_proj_weight
	bias = module.in_proj_bias
	if weight is None:
		raise ValueError("Manual MHA requires in_proj_weight.")
	q_bias = None if bias is None else bias[:embed_dim]
	k_bias = None if bias is None else bias[embed_dim : 2 * embed_dim]
	v_bias = None if bias is None else bias[2 * embed_dim :]
	q_proj = F.linear(query, weight[:embed_dim], q_bias)
	k_proj = F.linear(key, weight[embed_dim : 2 * embed_dim], k_bias)
	v_proj = F.linear(value, weight[2 * embed_dim :], v_bias)
	batch = int(query.shape[0])
	q_len = int(query.shape[1])
	k_len = int(key.shape[1])
	q_proj = q_proj.reshape(batch, q_len, num_heads, head_dim).transpose(1, 2)
	k_proj = k_proj.reshape(batch, k_len, num_heads, head_dim).transpose(1, 2)
	v_proj = v_proj.reshape(batch, k_len, num_heads, head_dim).transpose(1, 2)
	# These attentions have tiny sequence lengths (vertical: 8, fusion: 1x3).
	# Use broadcasted multiply/reduce instead of matmul/bmm so backward does not route
	# through Triton-backed bmm kernels on clusters where libcuda discovery is brittle.
	scores = (q_proj.unsqueeze(3) * k_proj.unsqueeze(2)).sum(dim=-1) / math.sqrt(float(head_dim))
	weights = torch.softmax(scores, dim=-1)
	weights_for_values = F.dropout(weights, p=float(module.dropout), training=training) if float(module.dropout) > 0.0 else weights
	out = (weights_for_values.unsqueeze(-1) * v_proj.unsqueeze(2)).sum(dim=3)
	out = out.transpose(1, 2).contiguous().reshape(batch, q_len, embed_dim)
	out = module.out_proj(out)
	if need_weights:
		return out, weights.mean(dim=1)
	return out, None


def select_channels_by_keywords(channel_names: Mapping[int, str] | None, keywords: Sequence[str], *, input_channels: int | None = None) -> list[int]:
	"""Return channel indices whose names contain any keyword."""

	if not channel_names or not keywords:
		return []
	lowered_keywords = [str(keyword).lower() for keyword in keywords]
	selected: list[int] = []
	for index, name in channel_names.items():
		channel_index = int(index)
		if input_channels is not None and not 0 <= channel_index < int(input_channels):
			continue
		lowered_name = str(name).lower()
		if any(keyword in lowered_name for keyword in lowered_keywords):
			selected.append(channel_index)
	return sorted(set(selected))


def _dedupe_indices(indices: Sequence[int], *, input_channels: int) -> list[int]:
	seen: set[int] = set()
	result: list[int] = []
	for index in indices:
		channel_index = int(index)
		if not 0 <= channel_index < int(input_channels):
			continue
		if channel_index not in seen:
			seen.add(channel_index)
			result.append(channel_index)
	return result


def _apply_per_timestep(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
	if x.ndim != 5:
		raise ValueError(f"Expected B x T x C x H x W tensor, got shape {tuple(x.shape)}.")
	batch, time_steps, channels, height, width = (int(value) for value in x.shape)
	y = module(x.reshape(batch * time_steps, channels, height, width))
	out_channels = int(y.shape[1])
	return y.reshape(batch, time_steps, out_channels, height, width)


class ConvBlock2d(nn.Module):
	"""Small channel-first Conv2d block."""

	def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
		super().__init__()
		padding = int(kernel_size) // 2
		self.block = nn.Sequential(
			nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=padding),
			nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
			nn.SiLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class TerrainEncoder(nn.Module):
	"""Encode static four-channel terrain features without a time axis."""
	def __init__(self, in_channels: int = 4, hidden_dim: int = 32, out_dim: int = 64, activation: str = "gelu", norm: str = "groupnorm") -> None:
		super().__init__(); self.out_dim=int(out_dim)
		act = nn.GELU() if str(activation).lower() == "gelu" else nn.SiLU()
		def block(cin, cout):
			return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(_group_count(cout), cout) if str(norm).lower() == "groupnorm" else nn.BatchNorm2d(cout), act)
		self.net=nn.Sequential(block(int(in_channels), int(hidden_dim)), block(int(hidden_dim), int(out_dim)), block(int(out_dim), int(out_dim)))
	def forward(self, terrain: torch.Tensor) -> torch.Tensor:
		if terrain.ndim != 4: raise ValueError(f"TerrainEncoder expects B x C x H x W, got {tuple(terrain.shape)}")
		return self.net(terrain)


class TerrainFiLMConditioner(nn.Module):
	"""Zero-initialized FiLM modulation for static terrain conditioning."""
	def __init__(self, dim: int = 64, use_scale: bool = True, use_shift: bool = True, scale_init_zero: bool = True, shift_init_zero: bool = True) -> None:
		super().__init__(); self.use_scale=bool(use_scale); self.use_shift=bool(use_shift)
		self.scale=nn.Conv2d(dim, dim, 1) if self.use_scale else None; self.shift=nn.Conv2d(dim, dim, 1) if self.use_shift else None
		if scale_init_zero and self.scale is not None: nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.scale.bias)
		if shift_init_zero and self.shift is not None: nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)
	def forward(self, z: torch.Tensor, terrain_embedding: torch.Tensor) -> torch.Tensor:
		if z.ndim != 5 or terrain_embedding.ndim != 4 or int(z.shape[2]) != int(terrain_embedding.shape[1]) or tuple(z.shape[-2:]) != tuple(terrain_embedding.shape[-2:]): raise ValueError("Terrain FiLM spatial/channel dimensions must match fused representation.")
		scale=self.scale(terrain_embedding) if self.scale is not None else torch.zeros_like(terrain_embedding)
		shift=self.shift(terrain_embedding) if self.shift is not None else torch.zeros_like(terrain_embedding)
		return z * (1 + scale[:, None]) + shift[:, None]


class ResidualBlock2d(nn.Module):
	"""Residual spatial refinement block for one timestep."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
			nn.GroupNorm(_group_count(int(channels)), int(channels)),
			nn.SiLU(inplace=True),
			nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
			nn.GroupNorm(_group_count(int(channels)), int(channels)),
		)
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.activation(x + self.block(x))


class SequenceSpatialEncoder(nn.Module):
	"""Apply a 2D projection and residual blocks independently per timestep."""

	def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, num_blocks: int) -> None:
		super().__init__()
		layers: list[nn.Module] = [ConvBlock2d(int(in_channels), int(hidden_dim), kernel_size=1)]
		for _ in range(max(0, int(num_blocks))):
			layers.append(ResidualBlock2d(int(hidden_dim)))
		if int(hidden_dim) != int(out_dim):
			layers.append(ConvBlock2d(int(hidden_dim), int(out_dim), kernel_size=1))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return _apply_per_timestep(x, self.net)


class TemporalConvBlock(nn.Module):
	"""Lightweight temporal mixing over B x T x C x H x W."""

	def __init__(self, channels: int) -> None:
		super().__init__()
		self.conv = nn.Conv3d(int(channels), int(channels), kernel_size=(3, 1, 1), padding=(1, 0, 0))
		self.norm = nn.GroupNorm(_group_count(int(channels)), int(channels))
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalConvBlock expects B x T x C x H x W, got {tuple(x.shape)}.")
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.activation(self.norm(self.conv(y)))
		return y.permute(0, 2, 1, 3, 4).contiguous()


class AtmosphereEncoder(nn.Module):
	"""Encode raw 8-level x 10-variable atmospheric structure."""

	def __init__(
		self,
		*,
		out_dim: int = 64,
		num_levels: int = 8,
		vars_per_level: int = 10,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		vertical_mixing: str = "attention",
		vertical_attention_chunk_size: int | None = 32768,
		vertical_attention_force_math: bool = True,
	) -> None:
		super().__init__()
		self.out_dim = int(out_dim)
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.hidden_dim = int(hidden_dim)
		self.required_channels = self.num_levels * self.vars_per_level
		self.vertical_mixing = str(vertical_mixing).lower()
		self.vertical_attention_chunk_size = None if vertical_attention_chunk_size in (None, 0, "", "null") else int(vertical_attention_chunk_size)
		self.vertical_attention_force_math = bool(vertical_attention_force_math)
		if self.vertical_mixing not in {"attention", "mlp"}:
			raise ValueError(f"vertical_mixing must be 'attention' or 'mlp', got {vertical_mixing!r}.")
		self.variable_projection = nn.Linear(self.vars_per_level, self.hidden_dim)
		if self.vertical_mixing == "attention":
			num_heads = max(1, min(4, self.hidden_dim // 16))
			while self.hidden_dim % num_heads != 0:
				num_heads -= 1
			self.vertical_attention = nn.MultiheadAttention(self.hidden_dim, num_heads, batch_first=True)
			self.pool_score = nn.Linear(self.hidden_dim, 1)
			self.vertical_mlp = None
		else:
			self.vertical_attention = None
			self.pool_score = None
			self.vertical_mlp = nn.Sequential(
				nn.Linear(self.num_levels * self.hidden_dim, self.hidden_dim),
				nn.SiLU(),
				nn.Linear(self.hidden_dim, self.hidden_dim),
			)
		self.refine = SequenceSpatialEncoder(self.hidden_dim, self.out_dim, self.out_dim, int(num_blocks))

	def _vertical_attention_context(self):
		if not self.vertical_attention_force_math or torch is None:
			return contextlib.nullcontext()
		try:
			from torch.nn.attention import SDPBackend, sdpa_kernel

			return sdpa_kernel(SDPBackend.MATH)
		except Exception:
			pass
		if not hasattr(torch.backends, "cuda"):
			return contextlib.nullcontext()
		try:
			return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
		except Exception:
			return contextlib.nullcontext()

	def _apply_vertical_attention(self, tokens: torch.Tensor) -> torch.Tensor:
		assert self.vertical_attention is not None
		chunk_size = self.vertical_attention_chunk_size
		if chunk_size is None or int(chunk_size) <= 0 or int(tokens.shape[0]) <= int(chunk_size):
			out, _ = _manual_multihead_attention(self.vertical_attention, tokens, tokens, tokens, training=self.training, need_weights=False)
			return out
		chunks = []
		for start in range(0, int(tokens.shape[0]), int(chunk_size)):
			chunk = tokens[start : start + int(chunk_size)]
			out, _ = _manual_multihead_attention(self.vertical_attention, chunk, chunk, chunk, training=self.training, need_weights=False)
			chunks.append(out)
		return torch.cat(chunks, dim=0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"AtmosphereEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) < self.required_channels:
			raise ValueError(
				f"AtmosphereEncoder requires at least {self.required_channels} channels for "
				f"{self.num_levels} levels x {self.vars_per_level} variables, got {int(x.shape[2])}."
			)
		batch, time_steps, _, height, width = (int(value) for value in x.shape)
		x_atm = x[:, :, : self.required_channels].reshape(batch, time_steps, self.num_levels, self.vars_per_level, height, width)
		tokens = x_atm.permute(0, 1, 4, 5, 2, 3).contiguous().reshape(batch * time_steps * height * width, self.num_levels, self.vars_per_level)
		tokens = self.variable_projection(tokens)
		if self.vertical_mixing == "attention":
			assert self.vertical_attention is not None and self.pool_score is not None
			tokens = self._apply_vertical_attention(tokens)
			weights = torch.softmax(self.pool_score(tokens), dim=1)
			mixed = (tokens * weights).sum(dim=1)
		else:
			assert self.vertical_mlp is not None
			mixed = self.vertical_mlp(tokens.reshape(tokens.shape[0], self.num_levels * self.hidden_dim))
		maps = mixed.reshape(batch, time_steps, height, width, self.hidden_dim).permute(0, 1, 4, 2, 3).contiguous()
		return self.refine(maps)


class WindEncoder(nn.Module):
	"""Encode wind transport features derived from U/V/W and optional engineered channels."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		num_levels: int = 8,
		vars_per_level: int = 10,
		u_var_index: int = 0,
		v_var_index: int = 1,
		w_var_index: int = 2,
		use_engineered_wind_channels: bool = True,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
		eps: float = 1.0e-6,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.u_var_index = int(u_var_index)
		self.v_var_index = int(v_var_index)
		self.w_var_index = int(w_var_index)
		self.eps = float(eps)
		for name, index in {"u_var_index": self.u_var_index, "v_var_index": self.v_var_index, "w_var_index": self.w_var_index}.items():
			if not 0 <= index < self.vars_per_level:
				raise ValueError(f"{name} must be in [0, {self.vars_per_level - 1}], got {index}.")
		keywords = _as_str_list(engineered_channel_keywords)
		engineered = select_channels_by_keywords(channel_names, keywords, input_channels=self.input_channels) if use_engineered_wind_channels else []
		self.engineered_indices = _dedupe_indices([index for index in engineered if index >= self.num_levels * self.vars_per_level], input_channels=self.input_channels)
		if use_engineered_wind_channels and keywords and channel_names is None:
			warnings.warn("WindEncoder channel manifest unavailable; using derived U/V/W wind features only.", stacklevel=2)
		derived_channels = 7 * self.num_levels
		self.encoder = SequenceSpatialEncoder(derived_channels + len(self.engineered_indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def _level_indices(self, var_index: int) -> list[int]:
		return [level * self.vars_per_level + int(var_index) for level in range(self.num_levels)]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"WindEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		min_required = (self.num_levels - 1) * self.vars_per_level + max(self.u_var_index, self.v_var_index, self.w_var_index) + 1
		if int(x.shape[2]) < min_required:
			raise ValueError(f"WindEncoder requires at least {min_required} channels for configured U/V/W extraction, got {int(x.shape[2])}.")
		u = x[:, :, self._level_indices(self.u_var_index)]
		v = x[:, :, self._level_indices(self.v_var_index)]
		w = x[:, :, self._level_indices(self.w_var_index)]
		speed = torch.sqrt(u.square() + v.square() + self.eps)
		dir_cos = u / speed.clamp_min(self.eps)
		dir_sin = v / speed.clamp_min(self.eps)
		updraft = torch.relu(w)
		parts = [u, v, w, speed, dir_cos, dir_sin, updraft]
		if self.engineered_indices:
			parts.append(x[:, :, self.engineered_indices])
		features = torch.cat(parts, dim=2)
		return self.temporal(self.encoder(features))


class FireFuelEncoder(nn.Module):
	"""Encode fuel and fire-history features with extra local capacity."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 4,
		raw_fuel_channels: Sequence[int] | None = None,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		raw = _as_int_list(raw_fuel_channels, [84, 85])
		engineered = select_channels_by_keywords(channel_names, _as_str_list(engineered_channel_keywords), input_channels=self.input_channels)
		self.indices = _dedupe_indices(raw + [index for index in engineered if index not in raw], input_channels=self.input_channels)
		if len(self.indices) < len(raw):
			raise ValueError(f"FireFuelEncoder requires raw fuel channels {raw}, but input_channels={self.input_channels}.")
		if engineered_channel_keywords and channel_names is None:
			warnings.warn("FireFuelEncoder channel manifest unavailable; using raw fuel channels only.", stacklevel=2)
		self.encoder = SequenceSpatialEncoder(len(self.indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FireFuelEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) <= max(self.indices):
			raise ValueError(f"FireFuelEncoder requires channels up to {max(self.indices)}, got {int(x.shape[2])}.")
		return self.temporal(self.encoder(x[:, :, self.indices]))


class FluxEnergyEncoder(nn.Module):
	"""Encode flux and energy-driver channels."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		out_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		raw_flux_channels: Sequence[int] | None = None,
		engineered_channel_keywords: Sequence[str] | None = None,
		channel_names: Mapping[int, str] | None = None,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		raw = _as_int_list(raw_flux_channels, [80, 81, 82, 83])
		engineered = select_channels_by_keywords(channel_names, _as_str_list(engineered_channel_keywords), input_channels=self.input_channels)
		self.indices = _dedupe_indices(raw + [index for index in engineered if index not in raw], input_channels=self.input_channels)
		if len(self.indices) < len(raw):
			raise ValueError(f"FluxEnergyEncoder requires raw flux channels {raw}, but input_channels={self.input_channels}.")
		if engineered_channel_keywords and channel_names is None:
			warnings.warn("FluxEnergyEncoder channel manifest unavailable; using raw flux channels only.", stacklevel=2)
		self.encoder = SequenceSpatialEncoder(len(self.indices), int(hidden_dim), int(out_dim), int(num_blocks))
		self.temporal = TemporalConvBlock(int(out_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FluxEnergyEncoder expects B x T x C x H x W, got {tuple(x.shape)}.")
		if int(x.shape[2]) <= max(self.indices):
			raise ValueError(f"FluxEnergyEncoder requires channels up to {max(self.indices)}, got {int(x.shape[2])}.")
		return self.temporal(self.encoder(x[:, :, self.indices]))


def tokens_to_grid(tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
	"""Convert B x T x N x D row-major tokens back to B x T x D x H x W."""
	if tokens.ndim != 4:
		raise ValueError(f"tokens_to_grid expects B x T x N x D tokens, got {tuple(tokens.shape)}.")
	height, width = (int(value) for value in spatial_shape)
	batch, time_steps, num_tokens, dim = (int(value) for value in tokens.shape)
	if num_tokens != height * width:
		raise ValueError(f"Token count N={num_tokens} does not match spatial shape {(height, width)}.")
	return tokens.transpose(-1, -2).reshape(batch, time_steps, dim, height, width).contiguous()


class MultimodalAlignment(nn.Module):
	"""Align encoded CAWFE-Latte modalities before token fusion."""

	def __init__(
		self,
		*,
		dim: int = 64,
		max_time: int = 16,
		use_spatial_pos: bool = True,
		use_temporal_pos: bool = True,
		separate_layernorms: bool = True,
		eps: float = 1.0e-5,
		learned_spatial_pos: bool = True,
		learned_temporal_pos: bool = True,
		flatten_order: str = "row_major",
	) -> None:
		super().__init__()
		self.dim = int(dim)
		self.max_time = int(max_time)
		self.use_spatial_pos = bool(use_spatial_pos)
		self.use_temporal_pos = bool(use_temporal_pos)
		self.separate_layernorms = bool(separate_layernorms)
		self.learned_spatial_pos = bool(learned_spatial_pos)
		self.learned_temporal_pos = bool(learned_temporal_pos)
		self.flatten_order = str(flatten_order).lower()
		if self.flatten_order != "row_major":
			raise ValueError(f"CAWFE-Latte alignment supports only flatten_order='row_major', got {flatten_order!r}.")
		if self.dim < 1:
			raise ValueError(f"alignment.dim must be positive, got {self.dim}.")
		if self.max_time < 1:
			raise ValueError(f"alignment.max_time must be positive, got {self.max_time}.")
		if not self.learned_spatial_pos and self.use_spatial_pos:
			raise ValueError("CAWFE-Latte alignment v1 currently supports only learned spatial positional embeddings.")
		if not self.learned_temporal_pos and self.use_temporal_pos:
			raise ValueError("CAWFE-Latte alignment v1 currently supports only learned temporal positional embeddings.")
		if self.separate_layernorms:
			self.norm_atm = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_wind = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_fire = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_flux = nn.LayerNorm(self.dim, eps=float(eps))
		else:
			shared = nn.LayerNorm(self.dim, eps=float(eps))
			self.norm_atm = shared
			self.norm_wind = shared
			self.norm_fire = shared
			self.norm_flux = shared
		if self.use_spatial_pos:
			self.register_parameter("spatial_pos", None)
			self._spatial_shape: tuple[int, int] | None = None
		else:
			self.register_parameter("spatial_pos", None)
			self._spatial_shape = None
		if self.use_temporal_pos:
			self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_time, 1, self.dim))
			nn.init.trunc_normal_(self.temporal_pos, std=0.02)
		else:
			self.register_parameter("temporal_pos", None)

	@staticmethod
	def to_tokens(x: torch.Tensor) -> torch.Tensor:
		# B,T,D,H,W -> B,T,N,D. flatten(-2) preserves row-major token order:
		# token index = y * W + x.
		return x.flatten(-2).transpose(-1, -2).contiguous()

	def _validate_inputs(self, tensors: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
		shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
		for name, tensor in tensors.items():
			if tensor.ndim != 5:
				raise ValueError(f"Alignment input {name} must be B x T x D x H x W, got {tuple(tensor.shape)}.")
		first = next(iter(tensors.values()))
		batch, time_steps, dim, height, width = (int(value) for value in first.shape)
		for name, tensor in tensors.items():
			if tuple(tensor.shape[:2]) != (batch, time_steps):
				raise ValueError(f"Alignment inputs must share B,T dimensions; got {shapes}.")
			if int(tensor.shape[2]) != self.dim:
				raise ValueError(f"Alignment input {name} has D={int(tensor.shape[2])}; expected {self.dim}.")
			if tuple(tensor.shape[-2:]) != (height, width):
				raise ValueError(f"Alignment inputs must share the same spatial grid; got {shapes}.")
		if time_steps > self.max_time:
			raise ValueError(f"Alignment received T={time_steps}, but max_time={self.max_time}.")
		return batch, time_steps, dim, height, width

	def _spatial_position(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		if not self.use_spatial_pos:
			return torch.zeros(1, 1, int(height) * int(width), self.dim, device=device, dtype=dtype)
		shape = (int(height), int(width))
		num_tokens = shape[0] * shape[1]
		if self.spatial_pos is None:
			param = nn.Parameter(torch.zeros(1, 1, num_tokens, self.dim, device=device, dtype=dtype))
			nn.init.trunc_normal_(param, std=0.02)
			self.spatial_pos = param
			self._spatial_shape = shape
		elif self._spatial_shape != shape:
			raise ValueError(f"Alignment learned spatial_pos was initialized for {self._spatial_shape}, got {shape}.")
		return self.spatial_pos.to(device=device, dtype=dtype)

	def forward(self, A: torch.Tensor, W: torch.Tensor, F: torch.Tensor, Q: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
		_, time_steps, _, height, width = self._validate_inputs({"atmosphere": A, "wind": W, "fire_fuel": F, "flux_energy": Q})
		A_tokens = self.norm_atm(self.to_tokens(A))
		W_tokens = self.norm_wind(self.to_tokens(W))
		F_tokens = self.norm_fire(self.to_tokens(F))
		Q_tokens = self.norm_flux(self.to_tokens(Q))
		spatial_pos = self._spatial_position(height, width, device=A_tokens.device, dtype=A_tokens.dtype)
		if self.use_temporal_pos:
			assert self.temporal_pos is not None
			temporal_pos = self.temporal_pos[:, :time_steps].to(device=A_tokens.device, dtype=A_tokens.dtype)
		else:
			temporal_pos = torch.zeros(1, time_steps, 1, self.dim, device=A_tokens.device, dtype=A_tokens.dtype)
		return {
			"atmosphere": A_tokens + spatial_pos + temporal_pos,
			"wind": W_tokens + spatial_pos + temporal_pos,
			"fire_fuel": F_tokens + spatial_pos + temporal_pos,
			"flux_energy": Q_tokens + spatial_pos + temporal_pos,
			"spatial_shape": (height, width),
		}


class FireQueryCrossAttentionFusion(nn.Module):
	"""Fuse modalities per spatial token/time using fire/fuel as the query."""

	modalities = ("atmosphere", "wind", "flux_energy")

	def __init__(
		self,
		*,
		dim: int = 64,
		num_heads: int = 4,
		dropout: float = 0.1,
		use_layer_norm: bool = True,
		residual: bool = True,
		attention_chunk_size: int | None = 32768,
		attention_force_math: bool = True,
	) -> None:
		super().__init__()
		self.dim = int(dim)
		self.residual = bool(residual)
		self.use_layer_norm = bool(use_layer_norm)
		self.attention_chunk_size = None if attention_chunk_size in (None, 0, "", "null") else int(attention_chunk_size)
		self.attention_force_math = bool(attention_force_math)
		if self.dim % int(num_heads) != 0:
			raise ValueError(f"fusion.num_heads={num_heads} must divide fusion.dim={self.dim}.")
		self.q_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.kv_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.attention = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
		self.dropout = nn.Dropout(float(dropout))
		self.out_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(self.dim * 2, self.dim))

	def _attention_context(self):
		if not self.attention_force_math or torch is None:
			return contextlib.nullcontext()
		try:
			from torch.nn.attention import SDPBackend, sdpa_kernel

			return sdpa_kernel(SDPBackend.MATH)
		except Exception:
			pass
		if not hasattr(torch.backends, "cuda"):
			return contextlib.nullcontext()
		try:
			return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
		except Exception:
			return contextlib.nullcontext()

	def _apply_attention(self, q: torch.Tensor, kv: torch.Tensor, *, return_attention: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
		chunk_size = self.attention_chunk_size
		if chunk_size is None or int(chunk_size) <= 0 or int(q.shape[0]) <= int(chunk_size):
			attended, weights = _manual_multihead_attention(self.attention, q, kv, kv, training=self.training, need_weights=return_attention)
			return attended, weights if return_attention else None
		attended_chunks = []
		weight_chunks = [] if return_attention else None
		for start in range(0, int(q.shape[0]), int(chunk_size)):
			end = start + int(chunk_size)
			attended, weights = _manual_multihead_attention(
				self.attention,
				q[start:end],
				kv[start:end],
				kv[start:end],
				training=self.training,
				need_weights=return_attention,
			)
			attended_chunks.append(attended)
			if return_attention and weights is not None:
				assert weight_chunks is not None
				weight_chunks.append(weights)
		attended_all = torch.cat(attended_chunks, dim=0)
		weights_all = torch.cat(weight_chunks, dim=0) if weight_chunks else None
		return attended_all, weights_all

	def forward(
		self,
		atmosphere: torch.Tensor,
		wind: torch.Tensor,
		fire_fuel: torch.Tensor,
		flux_energy: torch.Tensor,
		*,
		return_attention: bool = False,
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		for name, tensor in {"atmosphere": atmosphere, "wind": wind, "fire_fuel": fire_fuel, "flux_energy": flux_energy}.items():
			if tensor.ndim != 4:
				raise ValueError(f"Fusion input {name} must be B x T x N x D tokens, got {tuple(tensor.shape)}.")
			if int(tensor.shape[-1]) != self.dim:
				raise ValueError(f"Fusion input {name} has D={int(tensor.shape[-1])}; expected {self.dim}.")
		if not (atmosphere.shape == wind.shape == fire_fuel.shape == flux_energy.shape):
			raise ValueError("Fusion token inputs must all have the same B x T x N x D shape.")
		batch, time_steps, num_tokens, dim = (int(value) for value in fire_fuel.shape)
		q = self.q_norm(fire_fuel.reshape(batch * time_steps * num_tokens, 1, dim))
		kv = torch.stack([atmosphere, wind, flux_energy], dim=3).reshape(batch * time_steps * num_tokens, 3, dim)
		kv = self.kv_norm(kv)
		attended, weights = self._apply_attention(q, kv, return_attention=return_attention)
		attended = attended.reshape(batch, time_steps, num_tokens, dim)
		z = fire_fuel + self.dropout(attended) if self.residual else self.dropout(attended)
		z = z + self.dropout(self.mlp(self.out_norm(z)))
		if return_attention:
			attention_tokens = weights.squeeze(1).reshape(batch, time_steps, num_tokens, 3)
			return z, attention_tokens
		return z


class TemporalSpatialResidualBlock(nn.Module):
	"""Efficient residual block mixing time then local space."""

	def __init__(self, dim: int, *, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, residual: bool = True) -> None:
		super().__init__()
		self.residual = bool(residual)
		t_padding = int(temporal_kernel_size) // 2
		s_padding = int(spatial_kernel_size) // 2
		self.temporal = nn.Conv3d(int(dim), int(dim), kernel_size=(int(temporal_kernel_size), 1, 1), padding=(t_padding, 0, 0))
		self.temporal_norm = nn.GroupNorm(_group_count(int(dim)), int(dim))
		self.spatial = nn.Conv3d(int(dim), int(dim), kernel_size=(1, int(spatial_kernel_size), int(spatial_kernel_size)), padding=(0, s_padding, s_padding))
		self.spatial_norm = nn.GroupNorm(_group_count(int(dim)), int(dim))
		self.dropout = nn.Dropout3d(float(dropout)) if float(dropout) > 0 else nn.Identity()
		self.activation = nn.SiLU(inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalSpatialResidualBlock expects B x T x C x H x W, got {tuple(x.shape)}.")
		x3d = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.activation(self.temporal_norm(self.temporal(x3d)))
		y = self.spatial_norm(self.spatial(y))
		y = self.dropout(y)
		if self.residual:
			y = x3d + y
		y = self.activation(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class ResidualSpatiotemporalBlock(TemporalSpatialResidualBlock):
	"""CAWFE-Latte v1.1 residual spatiotemporal block.

	Input and output layout is B x T x D x H x W. Internally the block uses
	Conv3d over B x D x T x H x W, first mixing time and then local space.
	"""

	def __init__(self, dim: int, *, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, norm: str = "groupnorm", activation: str = "silu", residual: bool = True) -> None:
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"CAWFE-Latte v1.1 supports only post_fusion_backbone.norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"CAWFE-Latte v1.1 supports only post_fusion_backbone.activation='silu', got {activation!r}.")
		super().__init__(
			dim=int(dim),
			temporal_kernel_size=int(temporal_kernel_size),
			spatial_kernel_size=int(spatial_kernel_size),
			dropout=float(dropout),
			residual=bool(residual),
		)


class ResidualSpatiotemporalBackbone(nn.Module):
	"""Six-block same-resolution post-fusion backbone for CAWFE-Latte v1.1."""

	def __init__(self, *, dim: int = 64, num_blocks: int = 6, temporal_kernel_size: int = 3, spatial_kernel_size: int = 3, dropout: float = 0.1, residual: bool = True, type: str = "residual_spatiotemporal", norm: str = "groupnorm", activation: str = "silu") -> None:
		super().__init__()
		if str(type).lower() != "residual_spatiotemporal":
			raise ValueError(f"CAWFE-Latte v1.1 supports only post_fusion_backbone.type='residual_spatiotemporal', got {type!r}.")
		self.blocks = nn.Sequential(
			*(
				ResidualSpatiotemporalBlock(
					int(dim),
					temporal_kernel_size=int(temporal_kernel_size),
					spatial_kernel_size=int(spatial_kernel_size),
					dropout=float(dropout),
					norm=str(norm),
					activation=str(activation),
					residual=bool(residual),
				)
				for _ in range(max(0, int(num_blocks)))
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.blocks(x)


class TemporalCNNBackbone(nn.Module):
	"""Small temporal CNN backbone over fused sequence features."""

	def __init__(
		self,
		*,
		dim: int = 64,
		num_blocks: int = 3,
		temporal_kernel_size: int = 3,
		spatial_kernel_size: int = 3,
		dropout: float = 0.1,
		residual: bool = True,
		type: str = "temporal_cnn",
		norm: str = "groupnorm",
		activation: str = "silu",
	) -> None:
		super().__init__()
		if str(type).lower() != "temporal_cnn":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.type='temporal_cnn', got {type!r}.")
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"CAWFE-Latte v1 supports only backbone.activation='silu', got {activation!r}.")
		self.blocks = nn.Sequential(
			*(
				TemporalSpatialResidualBlock(
					int(dim),
					temporal_kernel_size=int(temporal_kernel_size),
					spatial_kernel_size=int(spatial_kernel_size),
					dropout=float(dropout),
					residual=bool(residual),
				)
				for _ in range(max(0, int(num_blocks)))
			)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.blocks(x)


class TemporalAggregator(nn.Module):
	"""Aggregate B x T x C x H x W features to B x C x H x W."""

	def __init__(self, *, mode: str = "last", dim: int = 64, input_sequence_length: int = 5) -> None:
		super().__init__()
		self.mode = str(mode).lower()
		self.dim = int(dim)
		if self.mode not in {"last", "mean", "attention_pool"}:
			raise ValueError(f"temporal_aggregation.mode must be last, mean, or attention_pool; got {mode!r}.")
		self.attention_score = nn.Conv2d(self.dim, 1, kernel_size=1) if self.mode == "attention_pool" else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TemporalAggregator expects B x T x C x H x W, got {tuple(x.shape)}.")
		if self.mode == "last":
			return x[:, -1]
		if self.mode == "mean":
			return x.mean(dim=1)
		assert self.attention_score is not None
		batch, time_steps, channels, height, width = (int(value) for value in x.shape)
		scores = self.attention_score(x.reshape(batch * time_steps, channels, height, width)).reshape(batch, time_steps, 1, height, width)
		weights = torch.softmax(scores, dim=1)
		return (x * weights).sum(dim=1)


class ShallowDecoder(nn.Module):
	"""Small same-resolution decoder for CAWFE-Latte v1."""

	def __init__(
		self,
		*,
		in_dim: int = 64,
		hidden_dim: int = 64,
		num_blocks: int = 2,
		dropout: float = 0.1,
		type: str = "shallow_cnn",
		norm: str = "groupnorm",
		activation: str = "silu",
	) -> None:
		super().__init__()
		if str(type).lower() != "shallow_cnn":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.type='shallow_cnn', got {type!r}.")
		if str(norm).lower() != "groupnorm":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.norm='groupnorm', got {norm!r}.")
		if str(activation).lower() != "silu":
			raise ValueError(f"CAWFE-Latte v1 supports only decoder.activation='silu', got {activation!r}.")
		layers: list[nn.Module] = [ConvBlock2d(int(in_dim), int(hidden_dim))]
		for _ in range(max(0, int(num_blocks))):
			layers.append(ResidualBlock2d(int(hidden_dim)))
		if float(dropout) > 0:
			layers.append(nn.Dropout2d(float(dropout)))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4:
			raise ValueError(f"ShallowDecoder expects B x C x H x W, got {tuple(x.shape)}.")
		return self.net(x)


def _activation_from_name(name: str) -> nn.Module:
	key = str(name).lower()
	if key in {"none", "identity", "logits"}:
		return nn.Identity()
	if key == "relu":
		return nn.ReLU(inplace=True)
	if key == "softplus":
		return nn.Softplus()
	raise ValueError(f"Unsupported CAWFE-Latte head activation {name!r}; expected none, relu, softplus, or logits.")


class PredictionHead(nn.Module):
	"""Single-channel prediction head with optional activation."""

	def __init__(self, in_dim: int, *, activation: str = "none") -> None:
		super().__init__()
		self.proj = nn.Conv2d(int(in_dim), 1, kernel_size=1)
		self.activation = _activation_from_name(activation)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.activation(self.proj(x))


class CAWFELatte(nn.Module):
	"""CAWFE-Latte v1 end-to-end model."""

	def __init__(
		self,
		*,
		input_channels: int = 129,
		input_sequence_length: int = 5,
		output_channels: int = 4,
		output_dim: int = 64,
		version: str = "v1_end_to_end",
		atmosphere: Mapping[str, Any] | None = None,
		wind: Mapping[str, Any] | None = None,
		fire_fuel: Mapping[str, Any] | None = None,
		flux_energy: Mapping[str, Any] | None = None,
		fusion: Mapping[str, Any] | None = None,
		alignment: Mapping[str, Any] | None = None,
		backbone: Mapping[str, Any] | None = None,
		temporal_aggregation: Mapping[str, Any] | None = None,
		decoder: Mapping[str, Any] | None = None,
		heads: Mapping[str, Any] | None = None,
		auxiliary: Mapping[str, Any] | None = None,
		use_terrain_conditioning: bool = False,
		terrain_encoder: Mapping[str, Any] | None = None,
		terrain_film: Mapping[str, Any] | None = None,
		channel_names: Mapping[int, str] | None = None,
		debug_prediction_head: bool = False,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.input_sequence_length = int(input_sequence_length)
		self.output_channels = int(output_channels)
		self.output_dim = int(output_dim)
		self.version = str(version)
		self.debug_prediction_head = bool(debug_prediction_head)
		if self.output_channels != 4:
			raise ValueError(f"CAWFE-Latte v1 expects output_channels=4, got {self.output_channels}.")
		if self.input_channels < 86:
			raise ValueError(f"CAWFE-Latte requires at least 86 input channels, got {self.input_channels}.")
		atmosphere_config = dict(atmosphere or {})
		wind_config = dict(wind or {})
		fire_config = dict(fire_fuel or {})
		flux_config = dict(flux_energy or {})
		fusion_config = dict(fusion or {})
		alignment_config = dict(alignment or {})
		backbone_config = dict(backbone or {})
		aggregation_config = dict(temporal_aggregation or {})
		decoder_config = dict(decoder or {})
		head_config = dict(heads or {})
		auxiliary_config = dict(auxiliary or {})
		self.use_terrain_conditioning = bool(use_terrain_conditioning)
		terrain_encoder_config = dict(terrain_encoder or {})
		terrain_film_config = dict(terrain_film or {})
		if self.use_terrain_conditioning:
			self.terrain_encoder = TerrainEncoder(**terrain_encoder_config)
			self.terrain_film = TerrainFiLMConditioner(dim=int(terrain_film_config.get("dim", self.output_dim)), **{key: value for key, value in terrain_film_config.items() if key != "dim"})
		else:
			self.terrain_encoder = None; self.terrain_film = None
		self.atmosphere_encoder = AtmosphereEncoder(out_dim=int(atmosphere_config.get("out_dim", self.output_dim)), **{key: value for key, value in atmosphere_config.items() if key != "out_dim"})
		self.wind_encoder = WindEncoder(input_channels=self.input_channels, out_dim=int(wind_config.get("out_dim", self.output_dim)), channel_names=channel_names, **{key: value for key, value in wind_config.items() if key != "out_dim"})
		self.fire_fuel_encoder = FireFuelEncoder(input_channels=self.input_channels, out_dim=int(fire_config.get("out_dim", self.output_dim)), channel_names=channel_names, **{key: value for key, value in fire_config.items() if key != "out_dim"})
		self.flux_energy_encoder = FluxEnergyEncoder(input_channels=self.input_channels, out_dim=int(flux_config.get("out_dim", self.output_dim)), channel_names=channel_names, **{key: value for key, value in flux_config.items() if key != "out_dim"})
		alignment_enabled = bool(alignment_config.get("enabled", True))
		if not alignment_enabled:
			raise ValueError("CAWFE-Latte multimodal alignment is mandatory before fusion.")
		spatial_alignment = dict(alignment_config.get("spatial", {})) if isinstance(alignment_config.get("spatial", {}), Mapping) else {}
		distribution_alignment = dict(alignment_config.get("distribution", {})) if isinstance(alignment_config.get("distribution", {}), Mapping) else {}
		temporal_alignment = dict(alignment_config.get("temporal", {})) if isinstance(alignment_config.get("temporal", {}), Mapping) else {}
		if str(distribution_alignment.get("norm_type", "layernorm")).lower() != "layernorm":
			raise ValueError("CAWFE-Latte alignment.distribution.norm_type must be 'layernorm'.")
		self.alignment = MultimodalAlignment(
			dim=self.output_dim,
			max_time=int(temporal_alignment.get("max_time", max(16, self.input_sequence_length))),
			use_spatial_pos=bool(spatial_alignment.get("add_positional_embedding", True)),
			use_temporal_pos=bool(temporal_alignment.get("add_positional_embedding", True)),
			separate_layernorms=bool(distribution_alignment.get("separate_norm_per_modality", True)),
			eps=float(distribution_alignment.get("eps", 1.0e-5)),
			learned_spatial_pos=bool(spatial_alignment.get("learned_positional_embedding", True)),
			learned_temporal_pos=bool(temporal_alignment.get("learned_positional_embedding", True)),
			flatten_order=str(spatial_alignment.get("flatten_order", "row_major")),
		)
		fusion_dim = int(fusion_config.get("dim", self.output_dim))
		if fusion_dim != self.output_dim:
			raise ValueError(f"fusion.dim must match output_dim for CAWFE-Latte v1, got {fusion_dim} and {self.output_dim}.")
		input_format = str(fusion_config.get("input_format", "tokens")).lower()
		if input_format != "tokens":
			raise ValueError(f"CAWFE-Latte fusion.input_format must be 'tokens', got {input_format!r}.")
		query_source = str(fusion_config.get("query_source", "fire_fuel"))
		if query_source != "fire_fuel":
			raise ValueError("CAWFE-Latte fusion currently supports only query_source='fire_fuel'.")
		key_value_sources = list(fusion_config.get("key_value_sources", ["atmosphere", "wind", "flux_energy"]))
		if key_value_sources != ["atmosphere", "wind", "flux_energy"]:
			raise ValueError("CAWFE-Latte fusion currently expects key_value_sources=['atmosphere', 'wind', 'flux_energy'].")
		self.fusion = FireQueryCrossAttentionFusion(
			dim=fusion_dim,
			num_heads=int(fusion_config.get("num_heads", 4)),
			dropout=float(fusion_config.get("dropout", 0.1)),
			use_layer_norm=bool(fusion_config.get("use_layer_norm", True)),
			residual=bool(fusion_config.get("residual", True)),
		)
		backbone_config.setdefault("dim", self.output_dim)
		self.temporal_backbone = TemporalCNNBackbone(**backbone_config)
		self.temporal_aggregator = TemporalAggregator(
			mode=str(aggregation_config.get("mode", "last")),
			dim=self.output_dim,
			input_sequence_length=self.input_sequence_length,
		)
		decoder_config.setdefault("in_dim", self.output_dim)
		decoder_config.setdefault("hidden_dim", self.output_dim)
		self.decoder = ShallowDecoder(**decoder_config)
		decoder_out_dim = int(decoder_config.get("hidden_dim", self.output_dim))
		self.surface_head = PredictionHead(decoder_out_dim, activation=dict(head_config.get("surface", {})).get("activation", "none"))
		self.canopy_head = PredictionHead(decoder_out_dim, activation=dict(head_config.get("canopy", {})).get("activation", "none"))
		self.mask_head = PredictionHead(decoder_out_dim, activation=dict(head_config.get("mask", {})).get("activation", "logits"))
		self.energy_head = PredictionHead(decoder_out_dim, activation=dict(head_config.get("energy", {})).get("activation", "none"))
		fire_support_config = dict(auxiliary_config.get("fire_support_head", {})) if isinstance(auxiliary_config.get("fire_support_head", {}), Mapping) else {}
		self.aux_fire_support_enabled = bool(fire_support_config.get("enabled", True))
		self.aux_fire_support_detach_source = bool(fire_support_config.get("detach_source", False))
		self.aux_fire_support_head = nn.Conv2d(self.output_dim, 1, kernel_size=1) if self.aux_fire_support_enabled else None

	def _validate_input(self, x: torch.Tensor) -> None:
		if x.ndim != 5:
			raise ValueError(f"CAWFE-Latte expects B x T x C x H x W input, got {tuple(x.shape)}.")
		if int(x.shape[1]) != self.input_sequence_length:
			raise ValueError(f"CAWFE-Latte expected T={self.input_sequence_length}, got T={int(x.shape[1])}.")
		if int(x.shape[2]) < 86:
			raise ValueError(f"CAWFE-Latte requires at least 86 input channels, got {int(x.shape[2])}.")
		if int(x.shape[2]) < self.input_channels:
			raise ValueError(f"CAWFE-Latte was configured for input_channels={self.input_channels}, got {int(x.shape[2])}.")

	def _predict_heads(self, decoded: torch.Tensor) -> torch.Tensor:
		return torch.cat(
			[
				self.surface_head(decoded),
				self.canopy_head(decoded),
				self.mask_head(decoded),
				self.energy_head(decoded),
			],
			dim=1,
		)

	def forward(self, x: torch.Tensor, terrain: torch.Tensor | None = None, *, return_features: bool = False, return_attention: bool = False):
		self._validate_input(x)
		atmosphere = self.atmosphere_encoder(x)
		wind = self.wind_encoder(x)
		fire_fuel = self.fire_fuel_encoder(x)
		flux_energy = self.flux_energy_encoder(x)
		aligned = self.alignment(atmosphere, wind, fire_fuel, flux_energy)
		fusion_result = self.fusion(
			aligned["atmosphere"],
			aligned["wind"],
			aligned["fire_fuel"],
			aligned["flux_energy"],
			return_attention=return_attention,
		)
		if return_attention:
			fused_tokens, attention = fusion_result
		else:
			fused_tokens = fusion_result
			attention = None
		spatial_shape = aligned["spatial_shape"]
		fused_dynamic = tokens_to_grid(fused_tokens, spatial_shape)
		fused = fused_dynamic
		terrain_embedding = None
		if self.use_terrain_conditioning:
			if terrain is None: raise ValueError("CAWFE-Latte terrain conditioning is enabled but terrain input is missing.")
			terrain_embedding = self.terrain_encoder(terrain)
			if tuple(terrain_embedding.shape[-2:]) != tuple(fused_dynamic.shape[-2:]):
				raise ValueError(f"Terrain encoder grid {tuple(terrain_embedding.shape[-2:])} does not match fused dynamic grid {tuple(fused_dynamic.shape[-2:])}.")
			fused = self.terrain_film(fused_dynamic, terrain_embedding)
		local = self.temporal_backbone(fused)
		aux_source = local[:, -1]
		if self.aux_fire_support_detach_source:
			aux_source = aux_source.detach()
		aux_logits = self.aux_fire_support_head(aux_source) if self.aux_fire_support_head is not None else None
		aggregated = self.temporal_aggregator(local)
		decoded = self.decoder(aggregated)
		prediction = self._predict_heads(decoded)
		if return_features:
			features = {
				"prediction": prediction,
				"aux_fire_support_logits": aux_logits,
				"atmosphere": atmosphere,
				"wind": wind,
				"fire_fuel": fire_fuel,
				"flux_energy": flux_energy,
				"aligned_atmosphere": aligned["atmosphere"],
				"aligned_wind": aligned["wind"],
				"aligned_fire_fuel": aligned["fire_fuel"],
				"aligned_flux_energy": aligned["flux_energy"],
				"fused_tokens": fused_tokens,
				"fused_grid": fused_dynamic,
				"spatial_shape": spatial_shape,
				"fused": fused,
				"fused_dynamic": fused_dynamic,
				"terrain_features": terrain_embedding,
				"fused_after_terrain": fused if self.use_terrain_conditioning else None,
				"fused_terrain": fused,
				"local": local,
				"aggregated": aggregated,
				"decoded": decoded,
			}
			if return_attention:
				features["fusion_attention"] = attention
			return features
		if aux_logits is not None:
			return {"prediction": prediction, "aux_fire_support_logits": aux_logits}
		return prediction


class TemporalAttentionPooling(nn.Module):
	"""Per-pixel temporal attention pooling for B x T x D x H x W features."""

	def __init__(self, *, dim: int = 64, hidden_dim: int | None = None, score_kernel_size: int = 1, type: str = "attention", enabled: bool = True, softmax_dim: str = "time", initialize_uniform: bool = True, return_attention: bool = True) -> None:
		super().__init__()
		if str(type).lower() != "attention":
			raise ValueError(f"TemporalAttentionPooling supports only type='attention', got {type!r}.")
		if str(softmax_dim).lower() != "time":
			raise ValueError(f"TemporalAttentionPooling supports only softmax_dim='time', got {softmax_dim!r}.")
		self.dim = int(dim)
		self.hidden_dim = int(hidden_dim or max(1, self.dim // 4))
		self.enabled = bool(enabled)
		self.return_attention = bool(return_attention)
		padding = int(score_kernel_size) // 2
		self.score_head = nn.Sequential(
			nn.Conv2d(self.dim, self.hidden_dim, kernel_size=int(score_kernel_size), padding=padding),
			nn.SiLU(inplace=True),
			nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
		)
		if bool(initialize_uniform):
			final = self.score_head[-1]
			if isinstance(final, nn.Conv2d):
				nn.init.zeros_(final.weight)
				nn.init.zeros_(final.bias)

	def forward(self, z: torch.Tensor, *, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		if z.ndim != 5:
			raise ValueError(f"TemporalAttentionPooling expects B x T x D x H x W, got {tuple(z.shape)}.")
		batch, time_steps, dim, height, width = (int(value) for value in z.shape)
		if dim != self.dim:
			raise ValueError(f"TemporalAttentionPooling expected D={self.dim}, got D={dim}.")
		if not self.enabled:
			pooled = z[:, -1]
			alpha = torch.zeros(batch, time_steps, 1, height, width, dtype=z.dtype, device=z.device)
			alpha[:, -1] = 1.0
			return (pooled, alpha) if return_attention else pooled
		score = self.score_head(z.reshape(batch * time_steps, dim, height, width)).reshape(batch, time_steps, 1, height, width)
		alpha = torch.softmax(score, dim=1)
		pooled = (alpha * z).sum(dim=1)
		return (pooled, alpha) if return_attention else pooled


class CAWFELatteV11(CAWFELatte):
	"""CAWFE-Latte v1.1 ablation with residual spatiotemporal backbone and support-gated regression heads."""

	def __init__(self, *, post_fusion_backbone: Mapping[str, Any] | None = None, support_gate: Mapping[str, Any] | None = None, **kwargs) -> None:
		kwargs = dict(kwargs)
		kwargs.setdefault("version", "v1_1_resblocks_support_gate")
		super().__init__(**kwargs)

		backbone_config = dict(post_fusion_backbone or {})
		backbone_config.setdefault("type", "residual_spatiotemporal")
		backbone_config.setdefault("dim", self.output_dim)
		backbone_config.setdefault("num_blocks", 6)
		self.temporal_backbone = ResidualSpatiotemporalBackbone(**backbone_config)

		support_config = dict(support_gate or {})
		self.support_gate_enabled = bool(support_config.get("enabled", True))
		self.support_gate_min = float(support_config.get("gate_min", 0.05))
		self.support_gate_max = float(support_config.get("gate_max", 1.0))
		if not 0.0 <= self.support_gate_min <= self.support_gate_max:
			raise ValueError(f"support_gate requires 0 <= gate_min <= gate_max, got {self.support_gate_min}, {self.support_gate_max}.")
		apply_to = support_config.get("apply_to", ["surface", "canopy", "energy"])
		self.support_gate_apply_to = {str(name).lower() for name in apply_to}
		if "mask" in self.support_gate_apply_to:
			raise ValueError("CAWFE-Latte v1.1 support gate must not be applied to mask logits.")
		support_head_config = dict(support_config.get("support_head", {})) if isinstance(support_config.get("support_head", {}), Mapping) else {}
		decoder_out_dim = int(getattr(self.surface_head.proj, "in_channels", self.output_dim))
		in_dim = int(support_head_config.get("in_dim", decoder_out_dim))
		if in_dim != decoder_out_dim:
			raise ValueError(f"support_gate.support_head.in_dim={in_dim} must match decoder feature dim={decoder_out_dim}.")
		out_channels = int(support_head_config.get("out_channels", 1))
		if out_channels != 1:
			raise ValueError("CAWFE-Latte v1.1 support head must output one channel.")
		kernel_size = int(support_head_config.get("kernel_size", 1))
		self.support_head = nn.Conv2d(in_dim, 1, kernel_size=kernel_size, padding=kernel_size // 2)
		# v1.1 uses support_head(decoded) for both the gate and auxiliary fire-support loss.
		self.aux_fire_support_head = None

	def _gate_regression_heads(self, decoded: torch.Tensor) -> dict[str, torch.Tensor]:
		raw_surface = self.surface_head(decoded)
		raw_canopy = self.canopy_head(decoded)
		raw_mask_logits = self.mask_head(decoded)
		raw_energy = self.energy_head(decoded)
		support_logits = self.support_head(decoded)
		support_prob = torch.sigmoid(support_logits)
		support_gate = self.support_gate_min + (self.support_gate_max - self.support_gate_min) * support_prob
		if self.support_gate_enabled:
			surface = raw_surface * support_gate if "surface" in self.support_gate_apply_to else raw_surface
			canopy = raw_canopy * support_gate if "canopy" in self.support_gate_apply_to else raw_canopy
			energy = raw_energy * support_gate if "energy" in self.support_gate_apply_to else raw_energy
		else:
			surface, canopy, energy = raw_surface, raw_canopy, raw_energy
		prediction = torch.cat([surface, canopy, raw_mask_logits, energy], dim=1)
		return {
			"prediction": prediction,
			"support_logits": support_logits,
			"support_prob": support_prob,
			"support_gate": support_gate,
			"raw_surface": raw_surface,
			"raw_canopy": raw_canopy,
			"raw_mask_logits": raw_mask_logits,
			"raw_energy": raw_energy,
		}

	def forward(self, x: torch.Tensor, terrain: torch.Tensor | None = None, *, return_features: bool = False, return_attention: bool = False):
		self._validate_input(x)
		atmosphere = self.atmosphere_encoder(x)
		wind = self.wind_encoder(x)
		fire_fuel = self.fire_fuel_encoder(x)
		flux_energy = self.flux_energy_encoder(x)
		aligned = self.alignment(atmosphere, wind, fire_fuel, flux_energy)
		fusion_result = self.fusion(
			aligned["atmosphere"],
			aligned["wind"],
			aligned["fire_fuel"],
			aligned["flux_energy"],
			return_attention=return_attention,
		)
		if return_attention:
			fused_tokens, attention = fusion_result
		else:
			fused_tokens = fusion_result
			attention = None
		spatial_shape = aligned["spatial_shape"]
		fused_dynamic = tokens_to_grid(fused_tokens, spatial_shape)
		fused = fused_dynamic
		terrain_embedding = None
		if self.use_terrain_conditioning:
			if terrain is None:
				raise ValueError("CAWFE-Latte terrain conditioning is enabled but terrain input is missing.")
			terrain_embedding = self.terrain_encoder(terrain)
			if tuple(terrain_embedding.shape[-2:]) != tuple(fused_dynamic.shape[-2:]):
				raise ValueError(f"Terrain encoder grid {tuple(terrain_embedding.shape[-2:])} does not match fused dynamic grid {tuple(fused_dynamic.shape[-2:])}.")
			fused = self.terrain_film(fused_dynamic, terrain_embedding)
		local = self.temporal_backbone(fused)
		aggregated = self.temporal_aggregator(local)
		decoded = self.decoder(aggregated)
		head_outputs = self._gate_regression_heads(decoded)
		prediction = head_outputs["prediction"]
		support_logits = head_outputs["support_logits"]
		aux_logits = support_logits if self.aux_fire_support_enabled else None
		if return_features:
			features = {
				"prediction": prediction,
				"aux_fire_support_logits": aux_logits,
				"support_logits": support_logits,
				"support_prob": head_outputs["support_prob"],
				"support_gate": head_outputs["support_gate"],
				"raw_surface": head_outputs["raw_surface"],
				"raw_canopy": head_outputs["raw_canopy"],
				"raw_mask_logits": head_outputs["raw_mask_logits"],
				"raw_energy": head_outputs["raw_energy"],
				"atmosphere": atmosphere,
				"wind": wind,
				"fire_fuel": fire_fuel,
				"flux_energy": flux_energy,
				"aligned_atmosphere": aligned["atmosphere"],
				"aligned_wind": aligned["wind"],
				"aligned_fire_fuel": aligned["fire_fuel"],
				"aligned_flux_energy": aligned["flux_energy"],
				"fused_tokens": fused_tokens,
				"fused_grid": fused_dynamic,
				"spatial_shape": spatial_shape,
				"fused": fused,
				"fused_dynamic": fused_dynamic,
				"terrain_features": terrain_embedding,
				"fused_after_terrain": fused if self.use_terrain_conditioning else None,
				"fused_terrain": fused,
				"local": local,
				"post_fusion_features": local,
				"aggregated": aggregated,
				"decoded": decoded,
			}
			if return_attention:
				features["fusion_attention"] = attention
			return features
		if aux_logits is not None:
			return {"prediction": prediction, "aux_fire_support_logits": aux_logits, "support_logits": support_logits}
		return prediction


class CAWFELatteV12(CAWFELatteV11):
	"""CAWFE-Latte v1.2 ablation: v1.1 plus temporal attention pooling after ResBlocks."""

	def __init__(self, *, temporal_pooling: Mapping[str, Any] | None = None, **kwargs) -> None:
		kwargs = dict(kwargs)
		kwargs.setdefault("version", "v1_2_temporal_attention_pooling")
		super().__init__(**kwargs)
		pooling_config = dict(temporal_pooling or {})
		pooling_config.setdefault("type", "attention")
		pooling_config.setdefault("dim", self.output_dim)
		pooling_config.setdefault("hidden_dim", max(1, self.output_dim // 4))
		self.temporal_pooling = TemporalAttentionPooling(**pooling_config)

	@staticmethod
	def _temporal_attention_entropy(alpha: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
		return -(alpha * torch.log(alpha.clamp_min(float(eps)))).sum(dim=1)

	def forward(self, x: torch.Tensor, terrain: torch.Tensor | None = None, *, return_features: bool = False, return_attention: bool = False):
		self._validate_input(x)
		atmosphere = self.atmosphere_encoder(x)
		wind = self.wind_encoder(x)
		fire_fuel = self.fire_fuel_encoder(x)
		flux_energy = self.flux_energy_encoder(x)
		aligned = self.alignment(atmosphere, wind, fire_fuel, flux_energy)
		fusion_result = self.fusion(
			aligned["atmosphere"],
			aligned["wind"],
			aligned["fire_fuel"],
			aligned["flux_energy"],
			return_attention=return_attention,
		)
		if return_attention:
			fused_tokens, attention = fusion_result
		else:
			fused_tokens = fusion_result
			attention = None
		spatial_shape = aligned["spatial_shape"]
		fused_dynamic = tokens_to_grid(fused_tokens, spatial_shape)
		fused = fused_dynamic
		terrain_embedding = None
		if self.use_terrain_conditioning:
			if terrain is None:
				raise ValueError("CAWFE-Latte terrain conditioning is enabled but terrain input is missing.")
			terrain_embedding = self.terrain_encoder(terrain)
			if tuple(terrain_embedding.shape[-2:]) != tuple(fused_dynamic.shape[-2:]):
				raise ValueError(f"Terrain encoder grid {tuple(terrain_embedding.shape[-2:])} does not match fused dynamic grid {tuple(fused_dynamic.shape[-2:])}.")
			fused = self.terrain_film(fused_dynamic, terrain_embedding)
		local = self.temporal_backbone(fused)
		pooled, temporal_alpha = self.temporal_pooling(local, return_attention=True)
		decoded = self.decoder(pooled)
		head_outputs = self._gate_regression_heads(decoded)
		prediction = head_outputs["prediction"]
		support_logits = head_outputs["support_logits"]
		aux_logits = support_logits if self.aux_fire_support_enabled else None
		if return_features:
			features = {
				"prediction": prediction,
				"aux_fire_support_logits": aux_logits,
				"support_logits": support_logits,
				"support_prob": head_outputs["support_prob"],
				"support_gate": head_outputs["support_gate"],
				"temporal_attention_alpha": temporal_alpha,
				"temporal_attention_entropy": self._temporal_attention_entropy(temporal_alpha),
				"raw_surface": head_outputs["raw_surface"],
				"raw_canopy": head_outputs["raw_canopy"],
				"raw_mask_logits": head_outputs["raw_mask_logits"],
				"raw_energy": head_outputs["raw_energy"],
				"atmosphere": atmosphere,
				"wind": wind,
				"fire_fuel": fire_fuel,
				"flux_energy": flux_energy,
				"aligned_atmosphere": aligned["atmosphere"],
				"aligned_wind": aligned["wind"],
				"aligned_fire_fuel": aligned["fire_fuel"],
				"aligned_flux_energy": aligned["flux_energy"],
				"fused_tokens": fused_tokens,
				"fused_grid": fused_dynamic,
				"spatial_shape": spatial_shape,
				"fused": fused,
				"fused_dynamic": fused_dynamic,
				"terrain_features": terrain_embedding,
				"fused_after_terrain": fused if self.use_terrain_conditioning else None,
				"fused_terrain": fused,
				"local": local,
				"post_fusion_features": local,
				"aggregated": pooled,
				"temporal_pooled_features": pooled,
				"decoded": decoded,
			}
			if return_attention:
				features["fusion_attention"] = attention
			return features
		if aux_logits is not None:
			return {"prediction": prediction, "aux_fire_support_logits": aux_logits, "support_logits": support_logits}
		return prediction


__all__ = [
	"AtmosphereEncoder",
	"CAWFELatte",
	"CAWFELatteV11",
	"CAWFELatteV12",
	"TerrainEncoder",
	"TerrainFiLMConditioner",
	"FireFuelEncoder",
	"FireQueryCrossAttentionFusion",
	"FluxEnergyEncoder",
	"MultimodalAlignment",
	"PredictionHead",
	"ShallowDecoder",
	"TemporalAggregator",
	"TemporalAttentionPooling",
	"TemporalCNNBackbone",
	"ResidualSpatiotemporalBackbone",
	"ResidualSpatiotemporalBlock",
	"TemporalSpatialResidualBlock",
	"TemporalConvBlock",
	"WindEncoder",
	"select_channels_by_keywords",
	"tokens_to_grid",
]
