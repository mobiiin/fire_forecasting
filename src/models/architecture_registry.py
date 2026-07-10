"""Architecture metadata for wildfire forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ArchitectureSpec:
	"""Describe one supported model architecture."""

	name: str
	requires_fixed_patch_size: bool
	patch_divisibility: int
	input_type: str
	supports_sequence: bool
	output_type: str
	expected_input_shape: str
	expected_output_shape: str
	supports_patch_cache: bool = True
	supports_tiled_inference: bool = True


ARCHITECTURE_REGISTRY: dict[str, ArchitectureSpec] = {
	"convlstm_unet": ArchitectureSpec(
		name="convlstm_unet",
		requires_fixed_patch_size=False,
		patch_divisibility=16,
		input_type="sequence",
		supports_sequence=True,
		output_type="final_step_map",
		expected_input_shape="(B, T, C, H, W)",
		expected_output_shape="(B, C_out, H, W)",
	),
	"earthformer_lite": ArchitectureSpec(
		name="earthformer_lite",
		requires_fixed_patch_size=True,
		patch_divisibility=16,
		input_type="sequence",
		supports_sequence=True,
		output_type="final_step_map",
		expected_input_shape="(B, T, C, H, W)",
		expected_output_shape="(B, 4, H, W)",
	),
	"st_mamba_lite": ArchitectureSpec(
		name="st_mamba_lite",
		requires_fixed_patch_size=True,
		patch_divisibility=16,
		input_type="sequence",
		supports_sequence=True,
		output_type="final_step_map",
		expected_input_shape="(B, T, C, H, W)",
		expected_output_shape="(B, 4, H, W)",
		supports_patch_cache=True,
		supports_tiled_inference=True,
	),
}


def resolve_model_architecture(config: Mapping[str, Any]) -> str:
	"""Resolve the configured model architecture name."""

	model_config = config.get("model", config)
	if isinstance(model_config, Mapping):
		architecture = model_config.get("architecture", model_config.get("name", "convlstm_unet"))
	else:
		architecture = "convlstm_unet"
	return str(architecture).lower()


def get_architecture_spec(name: str) -> ArchitectureSpec:
	"""Return the registry spec for a supported architecture."""

	key = str(name).lower()
	if key not in ARCHITECTURE_REGISTRY:
		raise KeyError(f"Unsupported model architecture: {name!r}.")
	return ARCHITECTURE_REGISTRY[key]
