"""New CAWFE-Latte encoder and fusion stem.

This is intentionally not a complete forecasting model yet. It implements only
atmosphere, wind, fire/fuel, flux/energy encoders and fire-query modality fusion.
"""

from __future__ import annotations

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
	) -> None:
		super().__init__()
		self.out_dim = int(out_dim)
		self.num_levels = int(num_levels)
		self.vars_per_level = int(vars_per_level)
		self.hidden_dim = int(hidden_dim)
		self.required_channels = self.num_levels * self.vars_per_level
		self.vertical_mixing = str(vertical_mixing).lower()
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
			tokens, _ = self.vertical_attention(tokens, tokens, tokens, need_weights=False)
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


class FireQueryCrossAttentionFusion(nn.Module):
	"""Fuse modalities per pixel/time using fire/fuel as the query."""

	modalities = ("atmosphere", "wind", "flux_energy")

	def __init__(self, *, dim: int = 64, num_heads: int = 4, dropout: float = 0.1, use_layer_norm: bool = True, residual: bool = True) -> None:
		super().__init__()
		self.dim = int(dim)
		self.residual = bool(residual)
		self.use_layer_norm = bool(use_layer_norm)
		if self.dim % int(num_heads) != 0:
			raise ValueError(f"fusion.num_heads={num_heads} must divide fusion.dim={self.dim}.")
		self.q_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.kv_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.attention = nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
		self.dropout = nn.Dropout(float(dropout))
		self.out_norm = nn.LayerNorm(self.dim) if self.use_layer_norm else nn.Identity()
		self.mlp = nn.Sequential(nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(float(dropout)), nn.Linear(self.dim * 2, self.dim))

	def _flatten(self, x: torch.Tensor) -> torch.Tensor:
		return x.permute(0, 1, 3, 4, 2).contiguous().reshape(-1, self.dim)

	def _unflatten(self, x: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
		batch, time_steps, height, width = shape
		return x.reshape(batch, time_steps, height, width, self.dim).permute(0, 1, 4, 2, 3).contiguous()

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
			if tensor.ndim != 5:
				raise ValueError(f"Fusion input {name} must be B x T x D x H x W, got {tuple(tensor.shape)}.")
			if int(tensor.shape[2]) != self.dim:
				raise ValueError(f"Fusion input {name} has dim={int(tensor.shape[2])}; expected {self.dim}.")
		if not (atmosphere.shape == wind.shape == fire_fuel.shape == flux_energy.shape):
			raise ValueError("Fusion inputs must all have the same B x T x D x H x W shape.")
		batch, time_steps, _, height, width = (int(value) for value in fire_fuel.shape)
		q = self.q_norm(self._flatten(fire_fuel)).unsqueeze(1)
		kv = torch.stack([self._flatten(atmosphere), self._flatten(wind), self._flatten(flux_energy)], dim=1)
		kv = self.kv_norm(kv)
		attended, weights = self.attention(q, kv, kv, need_weights=return_attention, average_attn_weights=True)
		f_flat = self._flatten(fire_fuel)
		z = f_flat + self.dropout(attended.squeeze(1)) if self.residual else self.dropout(attended.squeeze(1))
		z = z + self.dropout(self.mlp(self.out_norm(z)))
		z_maps = self._unflatten(z, (batch, time_steps, height, width))
		if return_attention:
			attention_maps = weights.squeeze(1).reshape(batch, time_steps, height, width, 3)
			return z_maps, attention_maps
		return z_maps


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
		fusion_dim = int(fusion_config.get("dim", self.output_dim))
		if fusion_dim != self.output_dim:
			raise ValueError(f"fusion.dim must match output_dim for CAWFE-Latte v1, got {fusion_dim} and {self.output_dim}.")
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
		fusion_result = self.fusion(atmosphere, wind, fire_fuel, flux_energy, return_attention=return_attention)
		if return_attention:
			fused, attention = fusion_result
		else:
			fused = fusion_result
			attention = None
		terrain_embedding = None
		fused_dynamic = fused
		if self.use_terrain_conditioning:
			if terrain is None: raise ValueError("CAWFE-Latte terrain conditioning is enabled but terrain input is missing.")
			terrain_embedding = self.terrain_encoder(terrain)
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
				"fused": fused,
				"fused_dynamic": fused_dynamic,
				"terrain_features": terrain_embedding,
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


__all__ = [
	"AtmosphereEncoder",
	"CAWFELatte",
	"TerrainEncoder",
	"TerrainFiLMConditioner",
	"FireFuelEncoder",
	"FireQueryCrossAttentionFusion",
	"FluxEnergyEncoder",
	"PredictionHead",
	"ShallowDecoder",
	"TemporalAggregator",
	"TemporalCNNBackbone",
	"TemporalSpatialResidualBlock",
	"TemporalConvBlock",
	"WindEncoder",
	"select_channels_by_keywords",
]
