"""Factory for building wildfire forecasting models from config."""

from __future__ import annotations

from typing import Any, Mapping

from src.models.architecture_registry import get_architecture_spec, resolve_model_architecture
from src.models.convlstm_unet import ConvLSTMUNet, build_convlstm_unet_from_config
from src.models.earthformer_lite import EarthformerLite
from src.models.st_mamba_lite import STMamba


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
	raise ValueError(f"Unsupported model architecture: {architecture!r}.")
