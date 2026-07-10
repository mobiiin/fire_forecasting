"""Factory for building wildfire forecasting models from config."""

from __future__ import annotations

from typing import Any, Mapping

from src.models.architecture_registry import get_architecture_spec, resolve_model_architecture
from src.models.cawfe_latte_lite import CAWFELatteLite
from src.models.convlstm_unet import ConvLSTMUNet, build_convlstm_unet_from_config
from src.models.earthformer_lite import EarthformerLite
from src.models.st_mamba_lite import STMamba
from src.models.weatherformer_lite import WeatherFormerLite


def build_model_from_config(config: Mapping[str, Any], input_channels: int):
	"""Build the configured architecture."""

	model_config = config.get("model", config)
	if not isinstance(model_config, Mapping):
		model_config = {}
	architecture = resolve_model_architecture(config)
	_ = get_architecture_spec(architecture)
	if architecture == "convlstm_unet":
		return build_convlstm_unet_from_config(config, input_channels=input_channels)
	if architecture == "earthformer_lite":
		section = config.get("earthformer_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		attention_pattern = str(section.get("attention_pattern", "axial")).lower()
		if attention_pattern != "axial":
			raise ValueError(
				"earthformer_lite currently supports only attention_pattern='axial'. "
				f"Got {attention_pattern!r}."
			)
		if bool(section.get("use_shifted_windows", False)):
			raise ValueError("earthformer_lite does not implement shifted windows in this simplified version.")
		return EarthformerLite(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			depths=section.get("depths", [2, 2]),
			num_heads=section.get("num_heads", [4, 8]),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			dropout=float(section.get("dropout", 0.0)),
			attention_dropout=float(section.get("attention_dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			use_global_vectors=bool(section.get("use_global_vectors", True)),
			num_global_vectors=int(section.get("num_global_vectors", 8)),
			use_time_pos_embed=bool(section.get("use_time_pos_embed", True)),
			use_space_pos_embed=bool(section.get("use_space_pos_embed", True)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			downsample_stages=int(section.get("downsample_stages", 2)),
			patch_merge_factor=int(section.get("patch_merge_factor", 2)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "st_mamba_lite":
		section = config.get("st_mamba_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		return STMamba(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			encoder_channels=section.get("encoder_channels", [64, 128]),
			decoder_channels=section.get("decoder_channels", [128, 64]),
			depths=section.get("depths", [2, 2]),
			mamba_backend=str(section.get("mamba_backend", "auto")),
			d_state=int(section.get("d_state", 16)),
			d_conv=int(section.get("d_conv", 4)),
			expand=int(section.get("expand", 2)),
			scan_mode=str(section.get("scan_mode", "route_pair")),
			scan_routes=section.get("scan_routes", ["HVT", "TVH"]),
			bidirectional_scan=bool(section.get("bidirectional_scan", True)),
			use_depthwise_conv3d=bool(section.get("use_depthwise_conv3d", True)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			dropout=float(section.get("dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			depthwise_conv3d_kernel_size=section.get("depthwise_conv3d_kernel_size", [3, 3, 3]),
			use_st_mixer=bool(section.get("use_st_mixer", True)),
			st_mixer_sequence_order=str(section.get("st_mixer_sequence_order", "THW")),
			use_unet_decoder=bool(section.get("use_unet_decoder", True)),
			use_skip_connections=bool(section.get("use_skip_connections", True)),
			upsample_mode=str(section.get("upsample_mode", "bilinear")),
			use_adaln_conditioning=bool(section.get("use_adaln_conditioning", False)),
			use_fire_static_embedding=bool(section.get("use_fire_static_embedding", False)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "weatherformer_lite":
		section = config.get("weatherformer_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		attention_type = str(section.get("attention_type", "factorized")).lower()
		if attention_type != "factorized":
			raise ValueError(
				"weatherformer_lite currently supports only attention_type='factorized'. "
				f"Got {attention_type!r}."
			)
		if not bool(section.get("temporal_attention", True)):
			raise ValueError("weatherformer_lite currently requires temporal_attention=true.")
		spatial_attention = str(section.get("spatial_attention", "window")).lower()
		if spatial_attention != "window":
			raise ValueError(
				"weatherformer_lite currently supports only spatial_attention='window'. "
				f"Got {spatial_attention!r}."
			)
		return WeatherFormerLite(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			encoder_channels=section.get("encoder_channels", [64, 128]),
			decoder_channels=section.get("decoder_channels", [128, 64]),
			depths=section.get("depths", [2, 2]),
			num_heads=section.get("num_heads", [4, 8]),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			use_channel_scaler=bool(section.get("use_channel_scaler", True)),
			use_feature_gate=bool(section.get("use_feature_gate", True)),
			scaler_init=float(section.get("scaler_init", 1.0)),
			use_time_pos_embed=bool(section.get("use_time_pos_embed", True)),
			use_2d_space_pos_embed=bool(section.get("use_2d_space_pos_embed", True)),
			use_fourier_space_encoding=bool(section.get("use_fourier_space_encoding", True)),
			window_size=int(section.get("window_size", 8)),
			shifted_window=bool(section.get("shifted_window", True)),
			use_global_tokens=bool(section.get("use_global_tokens", True)),
			num_global_tokens=int(section.get("num_global_tokens", 4)),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			dropout=float(section.get("dropout", 0.0)),
			attention_dropout=float(section.get("attention_dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			use_unet_decoder=bool(section.get("use_unet_decoder", True)),
			use_skip_connections=bool(section.get("use_skip_connections", True)),
			upsample_mode=str(section.get("upsample_mode", "bilinear")),
			downsample_stages=int(section.get("downsample_stages", 2)),
			patch_merge_factor=int(section.get("patch_merge_factor", 2)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", 16)),
		)
	if architecture == "cawfe_latte_lite":
		section = config.get("cawfe_latte_lite", {})
		if not isinstance(section, Mapping):
			section = {}
		backbone_type = section.get("backbone_type")
		if backbone_type is None:
			use_hybrid = bool(section.get("use_hybrid_backbone", True))
			use_transformer = bool(section.get("use_transformer_backbone", True))
			use_mamba = bool(section.get("use_mamba_backbone", True))
			if use_hybrid or (use_transformer and use_mamba):
				backbone_type = "hybrid_transformer_mamba"
			elif use_transformer:
				backbone_type = "transformer_only"
			elif use_mamba:
				backbone_type = "mamba_only"
			else:
				backbone_type = "conv_only"
		return CAWFELatteLite(
			input_channels=int(input_channels),
			output_channels=int(model_config.get("output_channels", 4)),
			input_sequence_length=int(section.get("input_sequence_length", config.get("input_sequence_length", 1))),
			patch_size=int(section.get("patch_size", config.get("patch_size", 64))),
			embed_dim=int(section.get("embed_dim", 64)),
			atm_embed_dim=int(section.get("atm_embed_dim", 48)),
			fire_embed_dim=int(section.get("fire_embed_dim", 48)),
			fused_dim=int(section.get("fused_dim", 96)),
			backbone_dim=int(section.get("backbone_dim", 96)),
			atmosphere_start_channel=int(section.get("atmosphere_start_channel", 0)),
			atmosphere_num_levels=int(section.get("atmosphere_num_levels", 8)),
			atmosphere_vars_per_level=int(section.get("atmosphere_vars_per_level", 10)),
			atmosphere_num_channels=int(section.get("atmosphere_num_channels", 80)),
			flux_channels=section.get("flux_channels", [80, 81, 82, 83]),
			fuel_channels=section.get("fuel_channels", [84, 85]),
			engineered_start_channel=int(section.get("engineered_start_channel", 86)),
			engineered_end_channel=int(section.get("engineered_end_channel", int(input_channels) - 1)),
			use_vertical_atmosphere_encoder=bool(section.get("use_vertical_atmosphere_encoder", True)),
			vertical_encoder_type=str(section.get("vertical_encoder_type", "attention")),
			use_fire_fuel_encoder=bool(section.get("use_fire_fuel_encoder", True)),
			use_fire_front_gate=bool(section.get("use_fire_front_gate", True)),
			backbone_type=str(backbone_type),
			backbone_depths=section.get("backbone_depths", [2, 2]),
			num_heads=section.get("num_heads", [4, 6]),
			window_size=int(section.get("window_size", 8)),
			shifted_window=bool(section.get("shifted_window", True)),
			mamba_backend=str(section.get("mamba_backend", "auto")),
			mamba_d_state=int(section.get("mamba_d_state", 16)),
			mamba_d_conv=int(section.get("mamba_d_conv", 4)),
			mamba_expand=int(section.get("mamba_expand", 2)),
			mamba_scan_mode=str(section.get("mamba_scan_mode", "tri_axis")),
			fire_gate_hidden_dim=int(section.get("fire_gate_hidden_dim", 32)),
			fire_gate_strength=float(section.get("fire_gate_strength", 1.0)),
			fire_gate_mode=str(section.get("fire_gate_mode", "multiplicative")),
			fire_gate_channels=section.get("fire_gate_channels", {"flux": True, "fuel": True, "engineered": True}),
			temporal_readout=str(section.get("temporal_readout", "attention_pool")),
			decoder_channels=section.get("decoder_channels", [128, 64]),
			use_skip_connections=bool(section.get("use_skip_connections", True)),
			upsample_mode=str(section.get("upsample_mode", "bilinear")),
			decoder_task_heads=str(section.get("decoder_task_heads", "separate")),
			use_physical_output_constraints=bool(section.get("use_physical_output_constraints", True)),
			constrain_consumed_nonnegative=bool(section.get("constrain_consumed_nonnegative", True)),
			constrain_energy_nonnegative=bool(section.get("constrain_energy_nonnegative", True)),
			mask_output_is_logits=bool(section.get("mask_output_is_logits", True)),
			dropout=float(section.get("dropout", 0.0)),
			attention_dropout=float(section.get("attention_dropout", 0.0)),
			drop_path=float(section.get("drop_path", 0.1)),
			mlp_ratio=float(section.get("mlp_ratio", 4.0)),
			gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
			required_patch_divisibility=int(section.get("required_patch_divisibility", config.get("patching", {}).get("require_patch_divisible_by", 16))),
		)
	raise ValueError(f"Unsupported model architecture: {architecture!r}.")
