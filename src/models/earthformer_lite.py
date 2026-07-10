"""Earthformer-lite model for wildfire forecasting.

This is an in-project simplified Earthformer-inspired model. It is not the
full official Earthformer implementation.
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

from src.models.earthformer_blocks import AxialCuboidAttentionBlock, ConvGELUBlock, PatchMerge2D, TemporalReadout
from src.models.input_adapters import SequenceConv2dInputAdapter


def _to_int_list(values: Sequence[int] | int, name: str) -> list[int]:
	if isinstance(values, (list, tuple)):
		return [int(value) for value in values]
	raise TypeError(f"{name} must be a sequence of integers, got {type(values)!r}.")


class EarthformerLite(nn.Module):
	"""Simplified Earthformer-style axial-attention encoder-decoder."""

	def __init__(
		self,
		input_channels: int,
		output_channels: int,
		input_sequence_length: int,
		patch_size: int,
		embed_dim: int,
		depths: Sequence[int],
		num_heads: Sequence[int],
		mlp_ratio: float,
		dropout: float,
		attention_dropout: float,
		drop_path: float,
		use_global_vectors: bool,
		num_global_vectors: int,
		use_time_pos_embed: bool,
		use_space_pos_embed: bool,
		gradient_checkpointing: bool,
		temporal_readout: str = "attention_pool",
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
		self.depths = _to_int_list(depths, "depths")
		self.num_heads = _to_int_list(num_heads, "num_heads")
		self.temporal_readout_name = str(temporal_readout).lower()
		self.downsample_stages = int(downsample_stages)
		self.patch_merge_factor = int(patch_merge_factor)
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
				"earthformer_lite.patch_size must be divisible by the required patch divisibility. "
				f"Got patch_size={self.patch_size}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if len(self.depths) != len(self.num_heads):
			raise ValueError(f"depths and num_heads must have the same length, got {self.depths} vs {self.num_heads}.")
		if len(self.depths) < 1:
			raise ValueError("earthformer_lite requires at least one hierarchy stage.")
		if any(depth <= 0 for depth in self.depths):
			raise ValueError(f"All earthformer_lite depths must be positive, got {self.depths}.")
		if any(heads <= 0 for heads in self.num_heads):
			raise ValueError(f"All earthformer_lite num_heads must be positive, got {self.num_heads}.")

		self.stage_dims = [self.embed_dim * (self.patch_merge_factor ** index) for index in range(len(self.depths))]
		self.input_adapter = SequenceConv2dInputAdapter(input_channels=self.input_channels, embed_dim=self.stage_dims[0], kernel_size=3)

		self.use_time_pos_embed = bool(use_time_pos_embed)
		self.use_space_pos_embed = bool(use_space_pos_embed)
		if self.use_time_pos_embed:
			self.time_pos_embed = nn.Parameter(torch.zeros(1, self.input_sequence_length, 1, 1, self.stage_dims[0]))
			nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
		else:
			self.register_parameter("time_pos_embed", None)
		if self.use_space_pos_embed:
			self.space_pos_embed = nn.Parameter(torch.zeros(1, self.stage_dims[0], self.patch_size, self.patch_size))
			nn.init.trunc_normal_(self.space_pos_embed, std=0.02)
		else:
			self.register_parameter("space_pos_embed", None)

		total_blocks = sum(self.depths)
		block_index = 0
		self.encoder_stages = nn.ModuleList()
		self.patch_merges = nn.ModuleList()
		for stage_index, depth in enumerate(self.depths):
			stage_blocks = []
			for _ in range(depth):
				stage_drop_path = float(drop_path) * block_index / max(total_blocks - 1, 1)
				stage_blocks.append(
					AxialCuboidAttentionBlock(
						dim=self.stage_dims[stage_index],
						num_heads=self.num_heads[stage_index],
						mlp_ratio=float(mlp_ratio),
						dropout=float(dropout),
						attention_dropout=float(attention_dropout),
						drop_path=stage_drop_path,
						use_global_vectors=bool(use_global_vectors),
						num_global_vectors=int(num_global_vectors),
						gradient_checkpointing=bool(gradient_checkpointing),
					)
				)
				block_index += 1
			self.encoder_stages.append(nn.ModuleList(stage_blocks))
			if stage_index < len(self.depths) - 1:
				self.patch_merges.append(PatchMerge2D(self.stage_dims[stage_index], self.stage_dims[stage_index + 1]))

		self.stage_readouts = nn.ModuleList(
			[TemporalReadout(dim=self.stage_dims[index], num_heads=self.num_heads[index], mode=self.temporal_readout_name) for index in range(len(self.depths))]
		)

		self.decoder_fuse_blocks = nn.ModuleList()
		for stage_index in reversed(range(len(self.stage_dims) - 1)):
			in_channels = self.stage_dims[stage_index + 1] + self.stage_dims[stage_index]
			out_channels = self.stage_dims[stage_index]
			self.decoder_fuse_blocks.append(ConvGELUBlock(in_channels=in_channels, out_channels=out_channels, dropout=float(dropout)))

		self.output_head = nn.Sequential(
			nn.Conv2d(self.stage_dims[0], self.stage_dims[0], kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(self.stage_dims[0], self.output_channels, kernel_size=1),
		)

	def _space_position_bias(self, x: torch.Tensor) -> torch.Tensor:
		if self.space_pos_embed is None:
			return torch.zeros_like(x)
		height, width = tuple(int(value) for value in x.shape[-2:])
		bias = self.space_pos_embed
		if height != self.patch_size or width != self.patch_size:
			bias = F.interpolate(bias, size=(height, width), mode="bilinear", align_corners=False)
		return bias.unsqueeze(1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"EarthformerLite expects a 5D tensor, got shape {tuple(x.shape)}.")
		if int(x.shape[1]) != self.input_sequence_length:
			raise ValueError(
				f"EarthformerLite expected input_sequence_length={self.input_sequence_length}, got T={int(x.shape[1])}."
			)
		if int(x.shape[2]) != self.input_channels:
			raise ValueError(f"EarthformerLite expected input_channels={self.input_channels}, got {int(x.shape[2])}.")
		height, width = tuple(int(value) for value in x.shape[-2:])
		if height % self.required_patch_divisibility != 0 or width % self.required_patch_divisibility != 0:
			raise ValueError(
				"EarthformerLite requires H and W to be divisible by required_patch_divisibility. "
				f"Got H={height}, W={width}, required_patch_divisibility={self.required_patch_divisibility}."
			)

		features = self.input_adapter(x)
		if self.use_time_pos_embed and self.time_pos_embed is not None:
			features = features + self.time_pos_embed.permute(0, 1, 4, 2, 3)
		if self.use_space_pos_embed:
			features = features + self._space_position_bias(features)
		features = features.permute(0, 1, 3, 4, 2)

		stage_features: list[torch.Tensor] = []
		current = features
		for stage_index, stage_blocks in enumerate(self.encoder_stages):
			for block in stage_blocks:
				current = block(current)
			stage_features.append(current)
			if stage_index < len(self.patch_merges):
				current = self.patch_merges[stage_index](current)

		current_2d = self.stage_readouts[-1](stage_features[-1]).permute(0, 3, 1, 2)
		for reverse_index, stage_index in enumerate(reversed(range(len(stage_features) - 1))):
			skip = self.stage_readouts[stage_index](stage_features[stage_index]).permute(0, 3, 1, 2)
			current_2d = F.interpolate(current_2d, size=skip.shape[-2:], mode="nearest")
			current_2d = self.decoder_fuse_blocks[reverse_index](torch.cat([skip, current_2d], dim=1))
		return self.output_head(current_2d)
