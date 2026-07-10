"""WeatherFormer-lite model for wildfire forecasting.

This is an in-project WeatherFormer-inspired architecture tailored to CAWFE
wildfire forecasting. It is not the official WeatherFormer implementation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, ModuleList=list)
	F = None

from src.models.weatherformer_blocks import FactorizedWeatherFormerBlock, TemporalReadout2D, _make_group_norm
from src.models.weatherformer_positional import WeatherFormerPositionalEncoding


def _to_int_list(values: Sequence[int] | int, name: str) -> list[int]:
	if isinstance(values, (list, tuple)):
		return [int(value) for value in values]
	raise TypeError(f"{name} must be a sequence of integers, got {type(values)!r}.")


class WeatherFormerLite(nn.Module):
	"""Factorized transformer with WeatherFormer-inspired input scaling."""

	def __init__(
		self,
		input_channels: int,
		output_channels: int,
		input_sequence_length: int,
		patch_size: int,
		embed_dim: int,
		encoder_channels: Sequence[int],
		decoder_channels: Sequence[int],
		depths: Sequence[int],
		num_heads: Sequence[int],
		mlp_ratio: float,
		use_channel_scaler: bool,
		use_feature_gate: bool,
		scaler_init: float,
		use_time_pos_embed: bool,
		use_2d_space_pos_embed: bool,
		use_fourier_space_encoding: bool,
		window_size: int,
		shifted_window: bool,
		use_global_tokens: bool,
		num_global_tokens: int,
		temporal_readout: str,
		dropout: float,
		attention_dropout: float,
		drop_path: float,
		gradient_checkpointing: bool,
		use_unet_decoder: bool = True,
		use_skip_connections: bool = True,
		upsample_mode: str = "bilinear",
		downsample_stages: int = 2,
		patch_merge_factor: int = 2,
		required_patch_divisibility: int = 16,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.output_channels = int(output_channels)
		self.input_sequence_length = int(input_sequence_length)
		self.patch_size = int(patch_size)
		self.embed_dim = int(embed_dim)
		self.encoder_channels = _to_int_list(encoder_channels, "encoder_channels")
		self.decoder_channels = _to_int_list(decoder_channels, "decoder_channels")
		self.depths = _to_int_list(depths, "depths")
		self.num_heads = _to_int_list(num_heads, "num_heads")
		self.window_size = int(window_size)
		self.shifted_window = bool(shifted_window)
		self.temporal_readout_name = str(temporal_readout).lower()
		self.upsample_mode = str(upsample_mode).lower()
		self.downsample_stages = int(downsample_stages)
		self.patch_merge_factor = int(patch_merge_factor)
		self.required_patch_divisibility = int(required_patch_divisibility)
		self.use_channel_scaler = bool(use_channel_scaler)
		self.use_feature_gate = bool(use_feature_gate)

		if self.input_channels <= 0:
			raise ValueError(f"input_channels must be positive, got {self.input_channels}.")
		if self.output_channels <= 0:
			raise ValueError(f"output_channels must be positive, got {self.output_channels}.")
		if self.input_sequence_length <= 0:
			raise ValueError(f"input_sequence_length must be positive, got {self.input_sequence_length}.")
		if self.patch_size <= 0:
			raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
		if self.window_size <= 0:
			raise ValueError(f"window_size must be positive, got {self.window_size}.")
		if self.patch_size % self.required_patch_divisibility != 0:
			raise ValueError(
				"weatherformer_lite.patch_size must be divisible by the required patch divisibility. "
				f"Got patch_size={self.patch_size}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if self.patch_size % self.window_size != 0:
			raise ValueError(
				"weatherformer_lite.patch_size must be divisible by window_size. "
				f"Got patch_size={self.patch_size}, window_size={self.window_size}."
			)
		if len(self.encoder_channels) != 2:
			raise ValueError(f"encoder_channels must contain exactly 2 values, got {self.encoder_channels}.")
		if len(self.decoder_channels) != 2:
			raise ValueError(f"decoder_channels must contain exactly 2 values, got {self.decoder_channels}.")
		if len(self.depths) != 2:
			raise ValueError(f"depths must contain exactly 2 values for the current implementation, got {self.depths}.")
		if len(self.num_heads) != 2:
			raise ValueError(f"num_heads must contain exactly 2 values for the current implementation, got {self.num_heads}.")
		if any(depth <= 0 for depth in self.depths):
			raise ValueError(f"All weatherformer_lite depths must be positive, got {self.depths}.")
		if any(heads <= 0 for heads in self.num_heads):
			raise ValueError(f"All weatherformer_lite num_heads must be positive, got {self.num_heads}.")
		if self.downsample_stages != 2:
			raise NotImplementedError(f"weatherformer_lite currently requires downsample_stages=2, got {self.downsample_stages}.")
		if self.patch_merge_factor != 2:
			raise NotImplementedError(f"weatherformer_lite currently requires patch_merge_factor=2, got {self.patch_merge_factor}.")
		if not bool(use_unet_decoder):
			raise NotImplementedError("weatherformer_lite currently requires use_unet_decoder=true.")
		if self.upsample_mode not in {"nearest", "bilinear", "convtranspose"}:
			raise ValueError(f"Unsupported upsample_mode: {upsample_mode!r}.")

		if self.use_channel_scaler:
			self.channel_scaler = nn.Parameter(torch.full((1, 1, self.input_channels, 1, 1), float(scaler_init)))
		else:
			self.register_parameter("channel_scaler", None)
		if self.use_feature_gate:
			self.feature_gate = nn.Parameter(torch.zeros(1, 1, self.input_channels, 1, 1))
		else:
			self.register_parameter("feature_gate", None)

		self.stem = nn.Sequential(
			nn.Conv2d(self.input_channels, self.embed_dim, kernel_size=3, padding=1),
			_make_group_norm(self.embed_dim),
			nn.GELU(),
			nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
			_make_group_norm(self.embed_dim),
			nn.GELU(),
		)
		self.stage1_input_project = (
			nn.Identity() if self.embed_dim == self.encoder_channels[0] else nn.Conv2d(self.embed_dim, self.encoder_channels[0], kernel_size=1)
		)
		self.positional_encoding = WeatherFormerPositionalEncoding(
			embed_dim=self.encoder_channels[0],
			input_sequence_length=self.input_sequence_length,
			patch_size=self.patch_size,
			use_time_pos_embed=bool(use_time_pos_embed),
			use_2d_space_pos_embed=bool(use_2d_space_pos_embed),
			use_fourier_space_encoding=bool(use_fourier_space_encoding),
		)

		total_blocks = sum(self.depths)
		block_index = 0
		self.stage1_blocks = nn.ModuleList()
		for local_index in range(self.depths[0]):
			block_drop_path = float(drop_path) * block_index / max(total_blocks - 1, 1)
			self.stage1_blocks.append(
				FactorizedWeatherFormerBlock(
					channels=self.encoder_channels[0],
					num_heads=self.num_heads[0],
					mlp_ratio=float(mlp_ratio),
					window_size=self.window_size,
					shifted_window=bool(self.shifted_window and (local_index % 2 == 1)),
					use_global_tokens=bool(use_global_tokens),
					num_global_tokens=int(num_global_tokens),
					dropout=float(dropout),
					attention_dropout=float(attention_dropout),
					drop_path=block_drop_path,
					gradient_checkpointing=bool(gradient_checkpointing),
				)
			)
			block_index += 1

		self.downsample = nn.Sequential(
			nn.Conv2d(self.encoder_channels[0], self.encoder_channels[1], kernel_size=3, stride=2, padding=1),
			_make_group_norm(self.encoder_channels[1]),
			nn.GELU(),
		)

		self.stage2_blocks = nn.ModuleList()
		for local_index in range(self.depths[1]):
			block_drop_path = float(drop_path) * block_index / max(total_blocks - 1, 1)
			self.stage2_blocks.append(
				FactorizedWeatherFormerBlock(
					channels=self.encoder_channels[1],
					num_heads=self.num_heads[1],
					mlp_ratio=float(mlp_ratio),
					window_size=self.window_size,
					shifted_window=bool(self.shifted_window and (local_index % 2 == 1)),
					use_global_tokens=bool(use_global_tokens),
					num_global_tokens=int(num_global_tokens),
					dropout=float(dropout),
					attention_dropout=float(attention_dropout),
					drop_path=block_drop_path,
					gradient_checkpointing=bool(gradient_checkpointing),
				)
			)
			block_index += 1

		self.skip_readout = TemporalReadout2D(self.encoder_channels[0], mode=self.temporal_readout_name)
		self.bottleneck_readout = TemporalReadout2D(self.encoder_channels[1], mode=self.temporal_readout_name)

		self.use_skip_connections = bool(use_skip_connections)
		if self.upsample_mode == "convtranspose":
			self.upsample = nn.ConvTranspose2d(self.encoder_channels[1], self.decoder_channels[0], kernel_size=2, stride=2)
			self.low_res_project = None
		else:
			self.upsample = None
			self.low_res_project = nn.Conv2d(self.encoder_channels[1], self.decoder_channels[0], kernel_size=1)

		decoder_input_channels = self.decoder_channels[0] + (self.encoder_channels[0] if self.use_skip_connections else 0)
		self.decoder = nn.Sequential(
			nn.Conv2d(decoder_input_channels, self.decoder_channels[1], kernel_size=3, padding=1),
			_make_group_norm(self.decoder_channels[1]),
			nn.GELU(),
			nn.Conv2d(self.decoder_channels[1], self.decoder_channels[1], kernel_size=3, padding=1),
			_make_group_norm(self.decoder_channels[1]),
			nn.GELU(),
		)
		self.output_head = nn.Sequential(
			nn.Conv2d(self.decoder_channels[1], self.decoder_channels[1], kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(self.decoder_channels[1], self.output_channels, kernel_size=1),
		)

	def _apply_input_scalers(self, x: torch.Tensor) -> torch.Tensor:
		y = x
		if self.channel_scaler is not None:
			y = y * self.channel_scaler
		if self.feature_gate is not None:
			y = y * torch.sigmoid(self.feature_gate)
		return y

	def _apply_per_timestep(self, x: torch.Tensor, module: nn.Module) -> torch.Tensor:
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
		y = module(x.reshape(batch_size * time_steps, channels, height, width))
		return y.reshape(batch_size, time_steps, int(y.shape[1]), int(y.shape[2]), int(y.shape[3]))

	def _upsample_bottleneck(self, z2: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
		if self.upsample is not None:
			return self.upsample(z2)
		projected = self.low_res_project(z2)  # type: ignore[operator]
		if self.upsample_mode == "bilinear":
			return F.interpolate(projected, size=target_size, mode=self.upsample_mode, align_corners=False)
		return F.interpolate(projected, size=target_size, mode=self.upsample_mode)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"WeatherFormerLite expects a 5D tensor, got shape {tuple(x.shape)}.")
		if int(x.shape[1]) != self.input_sequence_length:
			raise ValueError(
				f"WeatherFormerLite expected input_sequence_length={self.input_sequence_length}, got T={int(x.shape[1])}."
			)
		if int(x.shape[2]) != self.input_channels:
			raise ValueError(f"WeatherFormerLite expected input_channels={self.input_channels}, got {int(x.shape[2])}.")
		height, width = tuple(int(value) for value in x.shape[-2:])
		if height % self.window_size != 0 or width % self.window_size != 0:
			raise ValueError(f"WeatherFormerLite requires H/W divisible by window_size={self.window_size}, got H={height}, W={width}.")
		if height % self.required_patch_divisibility != 0 or width % self.required_patch_divisibility != 0:
			raise ValueError(
				"WeatherFormerLite requires H and W to be divisible by required_patch_divisibility. "
				f"Got H={height}, W={width}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if (height // 2) % self.window_size != 0 or (width // 2) % self.window_size != 0:
			raise ValueError(
				"WeatherFormerLite requires the downsampled spatial size to remain divisible by window_size. "
				f"Got H={height}, W={width}, window_size={self.window_size}."
			)

		scaled = self._apply_input_scalers(x)
		stem = self._apply_per_timestep(scaled, self.stem)
		stage1 = self._apply_per_timestep(stem, self.stage1_input_project)
		stage1 = self.positional_encoding(stage1)
		for block in self.stage1_blocks:
			stage1 = block(stage1)

		stage2 = self._apply_per_timestep(stage1, self.downsample)
		for block in self.stage2_blocks:
			stage2 = block(stage2)

		z1 = self.skip_readout(stage1)
		z2 = self.bottleneck_readout(stage2)
		upsampled = self._upsample_bottleneck(z2, target_size=tuple(int(value) for value in z1.shape[-2:]))
		if self.use_skip_connections:
			decoded = self.decoder(torch.cat([upsampled, z1], dim=1))
		else:
			decoded = self.decoder(upsampled)
		return self.output_head(decoded)
