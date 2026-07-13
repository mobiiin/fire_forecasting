"""CAWFE-Latte-Lite model for wildfire forecasting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping, Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

from src.models.cawfe_latte_backbone import HybridTransformerMambaBackbone
from src.models.cawfe_latte_blocks import SequenceConvBlock, TemporalReadout
from src.models.cawfe_latte_constraints import PhysicalOutputConstraintLayer
from src.models.cawfe_latte_decoder import MultiTaskFireDecoder
from src.models.cawfe_latte_fire import FireFrontAttentionGate, FireFuelStateEncoder
from src.models.cawfe_latte_vertical import FlatAtmosphereFallback, VerticalAtmosphereEncoder


def _to_int_list(values: Sequence[int] | int, name: str) -> list[int]:
	if isinstance(values, (list, tuple)):
		return [int(value) for value in values]
	raise TypeError(f"{name} must be a sequence of integers, got {type(values)!r}.")


class CAWFELatteLite(nn.Module):
	"""Layer-aware temporal Transformer/Mamba architecture for CAWFE patches."""

	def __init__(
		self,
		input_channels: int,
		output_channels: int,
		input_sequence_length: int,
		patch_size: int,
		embed_dim: int,
		atm_embed_dim: int,
		fire_embed_dim: int,
		fused_dim: int,
		backbone_dim: int,
		atmosphere_start_channel: int,
		atmosphere_num_levels: int,
		atmosphere_vars_per_level: int,
		atmosphere_num_channels: int,
		flux_channels: Sequence[int],
		fuel_channels: Sequence[int],
		engineered_start_channel: int,
		engineered_end_channel: int,
		use_vertical_atmosphere_encoder: bool,
		vertical_encoder_type: str,
		use_fire_fuel_encoder: bool,
		use_fire_front_gate: bool,
		backbone_type: str,
		backbone_depths: Sequence[int],
		num_heads: Sequence[int],
		window_size: int,
		shifted_window: bool,
		mamba_backend: str,
		mamba_d_state: int,
		mamba_d_conv: int,
		mamba_expand: int,
		mamba_scan_mode: str,
		fire_gate_hidden_dim: int,
		fire_gate_strength: float,
		fire_gate_mode: str,
		fire_gate_channels: Mapping[str, bool] | None,
		temporal_readout: str,
		decoder_channels: Sequence[int],
		use_skip_connections: bool,
		upsample_mode: str,
		decoder_task_heads: str,
		use_physical_output_constraints: bool,
		constrain_consumed_nonnegative: bool,
		constrain_energy_nonnegative: bool,
		mask_output_is_logits: bool,
		dropout: float = 0.0,
		attention_dropout: float = 0.0,
		drop_path: float = 0.1,
		mlp_ratio: float = 4.0,
		gradient_checkpointing: bool = False,
		required_patch_divisibility: int = 16,
		vertical_attention_chunk_size: int = 8192,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.output_channels = int(output_channels)
		self.input_sequence_length = int(input_sequence_length)
		self.patch_size = int(patch_size)
		self.embed_dim = int(embed_dim)
		self.atm_embed_dim = int(atm_embed_dim)
		self.fire_embed_dim = int(fire_embed_dim)
		self.fused_dim = int(fused_dim)
		self.backbone_dim = int(backbone_dim)
		self.atmosphere_start_channel = int(atmosphere_start_channel)
		self.atmosphere_num_levels = int(atmosphere_num_levels)
		self.atmosphere_vars_per_level = int(atmosphere_vars_per_level)
		self.atmosphere_num_channels = int(atmosphere_num_channels)
		self.flux_channels = _to_int_list(flux_channels, "flux_channels")
		self.fuel_channels = _to_int_list(fuel_channels, "fuel_channels")
		self.engineered_start_channel = int(engineered_start_channel)
		self.engineered_end_channel = int(engineered_end_channel)
		self.use_vertical_atmosphere_encoder = bool(use_vertical_atmosphere_encoder)
		self.use_fire_fuel_encoder = bool(use_fire_fuel_encoder)
		self.use_fire_front_gate = bool(use_fire_front_gate)
		self.backbone_type = str(backbone_type).lower()
		self.temporal_readout_name = str(temporal_readout).lower()
		self.required_patch_divisibility = int(required_patch_divisibility)
		self.fire_gate_channel_flags = dict(fire_gate_channels or {"flux": True, "fuel": True, "engineered": True})

		self._validate_init()

		if self.use_vertical_atmosphere_encoder:
			self.atmosphere_encoder = VerticalAtmosphereEncoder(
				num_levels=self.atmosphere_num_levels,
				vars_per_level=self.atmosphere_vars_per_level,
				atm_embed_dim=self.atm_embed_dim,
				encoder_type=str(vertical_encoder_type),
				num_heads=4,
				num_layers=1,
				dropout=float(dropout),
				pool_mode="attention_pool",
				attention_chunk_size=int(vertical_attention_chunk_size),
			)
		else:
			self.atmosphere_encoder = FlatAtmosphereFallback(self.atmosphere_num_channels, self.atm_embed_dim)

		self.fire_encoder = FireFuelStateEncoder(
			input_channels=self.input_channels,
			fire_embed_dim=self.fire_embed_dim,
			flux_channels=self.flux_channels,
			fuel_channels=self.fuel_channels,
			engineered_start_channel=self.engineered_start_channel,
			engineered_end_channel=self.engineered_end_channel,
			use_fire_fuel_encoder=self.use_fire_fuel_encoder,
		)
		self.fire_gate_selected_channels = self._build_fire_gate_channels()
		self.fusion = SequenceConvBlock(self.atm_embed_dim + self.fire_embed_dim, self.fused_dim, kernel_size=1)
		self.fusion_project = SequenceConvBlock(self.fused_dim, self.backbone_dim, kernel_size=3)
		self.fire_front_gate = (
			FireFrontAttentionGate(
				gate_input_channels=len(self.fire_gate_selected_channels),
				fused_channels=self.backbone_dim,
				hidden_dim=int(fire_gate_hidden_dim),
				gate_strength=float(fire_gate_strength),
				gate_mode=str(fire_gate_mode),
			)
			if self.use_fire_front_gate
			else None
		)
		self.backbone = HybridTransformerMambaBackbone(
			backbone_dim=self.backbone_dim,
			backbone_depths=backbone_depths,
			num_heads=num_heads,
			window_size=int(window_size),
			shifted_window=bool(shifted_window),
			backbone_type=self.backbone_type,
			mamba_backend=str(mamba_backend),
			mamba_d_state=int(mamba_d_state),
			mamba_d_conv=int(mamba_d_conv),
			mamba_expand=int(mamba_expand),
			mamba_scan_mode=str(mamba_scan_mode),
			mlp_ratio=float(mlp_ratio),
			dropout=float(dropout),
			attention_dropout=float(attention_dropout),
			drop_path=float(drop_path),
			gradient_checkpointing=bool(gradient_checkpointing),
		)
		self.skip_readout = TemporalReadout(self.backbone_dim, mode=self.temporal_readout_name)
		self.bottleneck_readout = TemporalReadout(self.backbone_dim * 2, mode=self.temporal_readout_name)
		self.decoder = MultiTaskFireDecoder(
			stage1_channels=self.backbone_dim,
			stage2_channels=self.backbone_dim * 2,
			decoder_channels=list(decoder_channels),
			output_channels=self.output_channels,
			use_skip_connections=bool(use_skip_connections),
			upsample_mode=str(upsample_mode),
			decoder_task_heads=str(decoder_task_heads),
		)
		self.constraints = (
			PhysicalOutputConstraintLayer(
				constrain_consumed_nonnegative=bool(constrain_consumed_nonnegative),
				constrain_energy_nonnegative=bool(constrain_energy_nonnegative),
				mask_output_is_logits=bool(mask_output_is_logits),
			)
			if bool(use_physical_output_constraints)
			else nn.Identity()
		)
		self.mamba_backend_used = self.backbone.mamba_backend_used

	def _validate_init(self) -> None:
		if self.input_channels <= 0 or self.output_channels <= 0:
			raise ValueError("input_channels and output_channels must be positive.")
		if self.output_channels != 4:
			raise ValueError(f"CAWFELatteLite expects output_channels=4, got {self.output_channels}.")
		if self.input_sequence_length <= 0 or self.patch_size <= 0:
			raise ValueError("input_sequence_length and patch_size must be positive.")
		if self.patch_size % self.required_patch_divisibility != 0:
			raise ValueError(
				"cawfe_latte_lite.patch_size must be divisible by required_patch_divisibility. "
				f"Got patch_size={self.patch_size}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if self.atmosphere_num_channels != self.atmosphere_num_levels * self.atmosphere_vars_per_level:
			raise ValueError(
				"atmosphere_num_channels must equal atmosphere_num_levels * atmosphere_vars_per_level. "
				f"Got {self.atmosphere_num_channels}, {self.atmosphere_num_levels}, {self.atmosphere_vars_per_level}."
			)
		self._validate_channel_layout(self.input_channels)

	def _validate_channel_layout(self, channels: int) -> None:
		atm_end = self.atmosphere_start_channel + self.atmosphere_num_channels
		if self.atmosphere_start_channel < 0 or atm_end > int(channels):
			raise ValueError(
				f"Atmospheric channel range [{self.atmosphere_start_channel}, {atm_end}) is incompatible with C={channels}."
			)
		for channel in self.flux_channels + self.fuel_channels:
			if channel < 0 or channel >= int(channels):
				raise ValueError(f"Configured flux/fuel channel {channel} is outside input channel count C={channels}.")
		if self.engineered_start_channel <= self.engineered_end_channel and self.engineered_end_channel >= int(channels):
			raise ValueError(
				f"engineered_end_channel={self.engineered_end_channel} is outside input channel count C={channels}."
			)

	def _build_fire_gate_channels(self) -> list[int]:
		selected: list[int] = []
		if bool(self.fire_gate_channel_flags.get("flux", True)):
			selected.extend(self.flux_channels)
		if bool(self.fire_gate_channel_flags.get("fuel", True)):
			selected.extend(self.fuel_channels)
		if bool(self.fire_gate_channel_flags.get("engineered", True)):
			selected.extend(range(self.engineered_start_channel, self.engineered_end_channel + 1))
		selected = sorted(dict.fromkeys(channel for channel in selected if 0 <= int(channel) < self.input_channels))
		if not selected:
			selected = list(self.fire_encoder.selected_channels)
		return selected

	def _validate_forward_input(self, x: torch.Tensor) -> None:
		if x.ndim != 5:
			raise ValueError(f"CAWFELatteLite expects a 5D tensor, got shape {tuple(x.shape)}.")
		if int(x.shape[1]) != self.input_sequence_length:
			raise ValueError(f"CAWFELatteLite expected input_sequence_length={self.input_sequence_length}, got T={int(x.shape[1])}.")
		if int(x.shape[2]) != self.input_channels:
			raise ValueError(f"CAWFELatteLite expected input_channels={self.input_channels}, got C={int(x.shape[2])}.")
		height, width = tuple(int(value) for value in x.shape[-2:])
		if height % self.required_patch_divisibility != 0 or width % self.required_patch_divisibility != 0:
			raise ValueError(
				"CAWFELatteLite requires H and W to be divisible by required_patch_divisibility. "
				f"Got H={height}, W={width}, required_patch_divisibility={self.required_patch_divisibility}."
			)
		if height % 2 != 0 or width % 2 != 0:
			raise ValueError(f"CAWFELatteLite requires even H/W for the two-stage decoder, got H={height}, W={width}.")

	def _atmosphere_slice(self, x: torch.Tensor) -> torch.Tensor:
		start = self.atmosphere_start_channel
		end = start + self.atmosphere_num_channels
		return x[:, :, start:end]

	def enabled_modules(self) -> dict[str, object]:
		return {
			"use_vertical_atmosphere_encoder": self.use_vertical_atmosphere_encoder,
			"use_fire_fuel_encoder": self.use_fire_fuel_encoder,
			"use_fire_front_gate": self.use_fire_front_gate,
			"use_transformer_backbone": self.backbone_type in {"transformer_only", "hybrid_transformer_mamba"},
			"use_mamba_backbone": self.backbone_type in {"mamba_only", "hybrid_transformer_mamba"},
			"use_hybrid_backbone": self.backbone_type == "hybrid_transformer_mamba",
			"backbone_type": self.backbone_type,
			"use_physical_output_constraints": not isinstance(self.constraints, nn.Identity),
		}

	def forward(self, x: torch.Tensor, return_aux: bool = False):
		self._validate_forward_input(x)
		x_atm = self._atmosphere_slice(x)
		atm_features = self.atmosphere_encoder(x_atm)
		fire_features = self.fire_encoder(x)
		fused = self.fusion(torch.cat([atm_features, fire_features], dim=2))
		fused = self.fusion_project(fused)

		fire_gate_map = None
		if self.fire_front_gate is not None:
			gate_input = x[:, :, self.fire_gate_selected_channels]
			fused, fire_gate_map = self.fire_front_gate(fused, gate_input)

		stage1, stage2 = self.backbone(fused)
		z1, skip_weights = self.skip_readout(stage1, return_weights=True)
		z2, bottleneck_weights = self.bottleneck_readout(stage2, return_weights=True)
		pred = self.decoder(z2, z1)
		pred = self.constraints(pred)
		if not return_aux:
			return pred
		aux = {
			"fire_gate_map": fire_gate_map,
			"temporal_attention_weights": {
				"skip": skip_weights,
				"bottleneck": bottleneck_weights,
			},
			"module_enabled_flags": self.enabled_modules(),
			"mamba_backend_used": self.mamba_backend_used,
		}
		return pred, aux


__all__ = ["CAWFELatteLite"]
