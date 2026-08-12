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
	custom_architecture: bool = False
	ablation_ready: bool = False
	paper_main_model: bool = False


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
	"weatherformer_lite": ArchitectureSpec(
		name="weatherformer_lite",
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

ARCHITECTURE_ALIASES = {
	"cawfe_st_mamba": "st_mamba_lite",
}

REMOVED_ARCHITECTURES = {"cawfe_latte", "cawfe_latte_lite"}
REMOVED_ARCHITECTURE_MESSAGE = "The old CAWFE-Latte implementation has been removed. A new design will be added later."


def resolve_model_architecture(config: Mapping[str, Any]) -> str:
	"""Resolve the configured model architecture name."""

	model_config = config.get("model", config)
	if isinstance(model_config, Mapping):
		architecture = model_config.get("architecture", model_config.get("name", "convlstm_unet"))
	else:
		architecture = "convlstm_unet"
	key = str(architecture).lower()
	return ARCHITECTURE_ALIASES.get(key, key)


def get_architecture_spec(name: str) -> ArchitectureSpec:
	"""Return the registry spec for a supported architecture."""

	key = str(name).lower()
	key = ARCHITECTURE_ALIASES.get(key, key)
	if key in REMOVED_ARCHITECTURES:
		raise KeyError(REMOVED_ARCHITECTURE_MESSAGE)
	if key not in ARCHITECTURE_REGISTRY:
		raise KeyError(f"Unsupported model architecture: {name!r}.")
	return ARCHITECTURE_REGISTRY[key]
