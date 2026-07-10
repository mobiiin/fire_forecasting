"""ST-Mamba-Lite model for CAWFE wildfire forecasting.

This is an in-project CAWFE-tailored spatial-temporal Mamba architecture. It
is inspired by MetMamba and ST-Mamba ideas, but it is not an official
reproduction of either paper.
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

from src.models.st_mamba_lite_blocks import STMambaBlock, TemporalReadout2D, _make_group_norm


def _to_int_list(values: Sequence[int] | int, name: str) -> list[int]:
	if isinstance(values, (list, tuple)):
		return [int(value) for value in values]
	raise TypeError(f"{name} must be a sequence of integers, got {type(values)!r}.")


class STMamba(nn.Module):
	"""ST-Mamba-Lite dense forecasting model."""

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
		mamba_backend: str,
		d_state: int,
		d_conv: int,
		expand: int,
		scan_mode: str,
		scan_routes: Sequence[str],
		bidirectional_scan: bool,
		use_depthwise_conv3d: bool,
		temporal_readout: str,
		dropout: float,
		drop_path: float,
		mlp_ratio: float,
		gradient_checkpointing: bool,
		depthwise_conv3d_kernel_size: Sequence[int] = (3, 3, 3),
		use_st_mixer: bool = True,
		st_mixer_sequence_order: str = "THW",
		use_unet_decoder: bool = True,
		use_skip_connections: bool = True,
		upsample_mode: str = "bilinear",
		use_adaln_conditioning: bool = False,
		use_fire_static_embedding: bool = False,
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
		self.mamba_backend = str(mamba_backend).lower()
		self.scan_mode = str(scan_mode).lower()
		self.scan_routes = [str(route) for route in scan_routes]
		self.bidirectional_scan = bool(bidirectional_scan)
		self.temporal_readout_mode = str(temporal_readout).lower()
		self.upsample_mode = str(upsample_mode).lower()
		self.required_patch_divisibility = int(required_patch_divisibility)

		if self.input_channels <= 0:
			raise ValueError(f"input_channels must be positive, got {self.input_channels}.")
		if self.output_channels <= 0:
			raise ValueError(f"output_channels must be positive, got {self.output_channels}.")
		if self.input_sequence_length <= 0:
			raise ValueError(f"input_sequence_length must be positive, got {self.input_sequence_length}.")
		if self.patch_size <= 0:
			raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
		if self.patch_size % self.required_patch_divisibility != 0:
			raise ValueError(
				"st_mamba_lite.patch_size must be divisible by the required patch divisibility. "
				f"Got patch_size={self.patch_size}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if len(self.encoder_channels) != 2:
			raise ValueError(f"encoder_channels must contain exactly 2 values, got {self.encoder_channels}.")
		if len(self.decoder_channels) != 2:
			raise ValueError(f"decoder_channels must contain exactly 2 values, got {self.decoder_channels}.")
		if len(self.depths) != 2:
			raise ValueError(f"depths must contain exactly 2 values for the current implementation, got {self.depths}.")
		if any(depth <= 0 for depth in self.depths):
			raise ValueError(f"All st_mamba_lite depths must be positive, got {self.depths}.")
		if self.encoder_channels[0] <= 0 or self.encoder_channels[1] <= 0:
			raise ValueError(f"encoder_channels must be positive, got {self.encoder_channels}.")
		if not bool(use_unet_decoder):
			raise NotImplementedError("st_mamba_lite currently requires use_unet_decoder=true.")
		if bool(use_adaln_conditioning):
			raise NotImplementedError("st_mamba_lite.use_adaln_conditioning is a placeholder and is not implemented yet.")
		if bool(use_fire_static_embedding):
			raise NotImplementedError("st_mamba_lite.use_fire_static_embedding is a placeholder and is not implemented yet.")
		if self.upsample_mode not in {"nearest", "bilinear", "convtranspose"}:
			raise ValueError(f"Unsupported upsample_mode: {upsample_mode!r}.")

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

		total_blocks = sum(self.depths)
		block_index = 0
		self.stage1_blocks = nn.ModuleList()
		for _ in range(self.depths[0]):
			block_drop_path = float(drop_path) * block_index / max(total_blocks - 1, 1)
			self.stage1_blocks.append(
				STMambaBlock(
					channels=self.encoder_channels[0],
					mamba_backend=self.mamba_backend,
					d_state=int(d_state),
					d_conv=int(d_conv),
					expand=int(expand),
					scan_mode=self.scan_mode,
					scan_routes=self.scan_routes,
					bidirectional_scan=self.bidirectional_scan,
					use_depthwise_conv3d=bool(use_depthwise_conv3d),
					depthwise_conv3d_kernel_size=depthwise_conv3d_kernel_size,
					use_st_mixer=bool(use_st_mixer),
					st_mixer_sequence_order=str(st_mixer_sequence_order),
					dropout=float(dropout),
					drop_path=block_drop_path,
					mlp_ratio=float(mlp_ratio),
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
		for _ in range(self.depths[1]):
			block_drop_path = float(drop_path) * block_index / max(total_blocks - 1, 1)
			self.stage2_blocks.append(
				STMambaBlock(
					channels=self.encoder_channels[1],
					mamba_backend=self.mamba_backend,
					d_state=int(d_state),
					d_conv=int(d_conv),
					expand=int(expand),
					scan_mode=self.scan_mode,
					scan_routes=self.scan_routes,
					bidirectional_scan=self.bidirectional_scan,
					use_depthwise_conv3d=bool(use_depthwise_conv3d),
					depthwise_conv3d_kernel_size=depthwise_conv3d_kernel_size,
					use_st_mixer=bool(use_st_mixer),
					st_mixer_sequence_order=str(st_mixer_sequence_order),
					dropout=float(dropout),
					drop_path=block_drop_path,
					mlp_ratio=float(mlp_ratio),
					gradient_checkpointing=bool(gradient_checkpointing),
				)
			)
			block_index += 1

		self.skip_readout = TemporalReadout2D(self.encoder_channels[0], mode=self.temporal_readout_mode)
		self.bottleneck_readout = TemporalReadout2D(self.encoder_channels[1], mode=self.temporal_readout_mode)

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

		self.mamba_backend_used = next(iter(self.stage1_blocks)).backend_name if self.stage1_blocks else "unknown"

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
			raise ValueError(f"STMamba expects a 5D tensor, got shape {tuple(x.shape)}.")
		if int(x.shape[1]) != self.input_sequence_length:
			raise ValueError(f"STMamba expected input_sequence_length={self.input_sequence_length}, got T={int(x.shape[1])}.")
		if int(x.shape[2]) != self.input_channels:
			raise ValueError(f"STMamba expected input_channels={self.input_channels}, got {int(x.shape[2])}.")
		height, width = tuple(int(value) for value in x.shape[-2:])
		if height % self.required_patch_divisibility != 0 or width % self.required_patch_divisibility != 0:
			raise ValueError(
				"STMamba requires H and W to be divisible by the required patch divisibility. "
				f"Got H={height}, W={width}, required_patch_divisibility={self.required_patch_divisibility}."
			)

		stem = self._apply_per_timestep(x, self.stem)
		stage1 = self._apply_per_timestep(stem, self.stage1_input_project)
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
