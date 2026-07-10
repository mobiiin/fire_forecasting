"""Route-based flatten/unflatten helpers for ST-Mamba-style scans."""

from __future__ import annotations

from itertools import permutations

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None


_AXIS_NAME_TO_DIM = {
	"T": 1,
	"H": 3,
	"V": 4,
	"W": 4,
}

_CANONICAL_AXIS = {
	"T": "T",
	"H": "H",
	"V": "V",
	"W": "V",
}


def canonicalize_route(route: str) -> str:
	"""Normalize route strings and validate supported axis labels."""

	normalized = "".join(_CANONICAL_AXIS.get(character.upper(), "?") for character in str(route))
	if "?" in normalized or len(normalized) != 3 or set(normalized) != {"T", "H", "V"}:
		raise ValueError(
			"Scan routes must be permutations of T/H/V (or T/H/W for width). "
			f"Got {route!r}."
		)
	return normalized


def all_supported_routes() -> list[str]:
	"""Return the six canonical T/H/V route permutations."""

	return ["".join(items) for items in permutations("THV", 3)]


def route_to_permute_dims(route: str) -> tuple[int, int, int]:
	"""Map a route string to tensor dimensions in ``(B, T, C, H, W)`` layout."""

	canonical = canonicalize_route(route)
	return tuple(_AXIS_NAME_TO_DIM[character] for character in canonical)


def flatten_by_route(x: torch.Tensor, route: str) -> torch.Tensor:
	"""Flatten ``(B, T, C, H, W)`` into ``(B, L, C)`` using one scan route."""

	if x.ndim != 5:
		raise ValueError(f"flatten_by_route expects a 5D tensor, got {tuple(x.shape)}.")
	permute_dims = route_to_permute_dims(route)
	reordered = x.permute((0, *permute_dims, 2)).contiguous()
	batch_size = int(reordered.shape[0])
	channels = int(reordered.shape[-1])
	return reordered.reshape(batch_size, -1, channels)


def unflatten_by_route(sequence: torch.Tensor, route: str, target_shape: tuple[int, int, int, int, int]) -> torch.Tensor:
	"""Invert ``flatten_by_route`` back into ``(B, T, C, H, W)``."""

	if sequence.ndim != 3:
		raise ValueError(f"unflatten_by_route expects a 3D tensor, got {tuple(sequence.shape)}.")
	if len(target_shape) != 5:
		raise ValueError(f"target_shape must be a 5-tuple, got {target_shape!r}.")
	batch_size, time_steps, channels, height, width = tuple(int(value) for value in target_shape)
	permute_dims = route_to_permute_dims(route)
	axis_size_by_dim = {1: time_steps, 3: height, 4: width}
	reshape_dims = [batch_size, *(axis_size_by_dim[dimension] for dimension in permute_dims), channels]
	expected_length = reshape_dims[1] * reshape_dims[2] * reshape_dims[3]
	if int(sequence.shape[0]) != batch_size or int(sequence.shape[1]) != expected_length or int(sequence.shape[2]) != channels:
		raise ValueError(
			"Sequence shape does not match the provided target shape for route restoration. "
			f"sequence={tuple(sequence.shape)} target_shape={target_shape} route={route!r}."
		)
	reordered = sequence.reshape(*reshape_dims)
	inverse_permute = [0] * 5
	for destination_index, source_index in enumerate((0, *permute_dims, 2)):
		inverse_permute[source_index] = destination_index
	return reordered.permute(*inverse_permute).contiguous()


def reverse_sequence(sequence: torch.Tensor) -> torch.Tensor:
	"""Reverse the sequence length dimension of a ``(B, L, C)`` tensor."""

	if sequence.ndim != 3:
		raise ValueError(f"reverse_sequence expects a 3D tensor, got {tuple(sequence.shape)}.")
	return torch.flip(sequence, dims=[1])
