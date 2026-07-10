"""Fire/fuel encoders and fire-front gate for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

from src.models.cawfe_latte_blocks import SequenceConvBlock, apply_per_timestep


def _range_channels(start: int, end: int, input_channels: int) -> list[int]:
	if int(end) < int(start):
		return []
	return [channel for channel in range(int(start), int(end) + 1) if 0 <= channel < int(input_channels)]


class FireFuelStateEncoder(nn.Module):
	"""Encode flux, fuel, and engineered fire-weather channels."""

	def __init__(
		self,
		input_channels: int,
		fire_embed_dim: int,
		flux_channels: Sequence[int],
		fuel_channels: Sequence[int],
		engineered_start_channel: int,
		engineered_end_channel: int,
		use_fire_fuel_encoder: bool = True,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.fire_embed_dim = int(fire_embed_dim)
		self.use_fire_fuel_encoder = bool(use_fire_fuel_encoder)
		selected = [int(channel) for channel in list(flux_channels) + list(fuel_channels) if 0 <= int(channel) < self.input_channels]
		selected.extend(_range_channels(int(engineered_start_channel), int(engineered_end_channel), self.input_channels))
		self.selected_channels = sorted(dict.fromkeys(selected))
		if not self.selected_channels:
			raise ValueError("FireFuelStateEncoder selected no input channels.")
		kernel_size = 3 if self.use_fire_fuel_encoder else 1
		self.encoder = SequenceConvBlock(len(self.selected_channels), self.fire_embed_dim, kernel_size=kernel_size)

	def extract_fire_channels(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"FireFuelStateEncoder expects (B, T, C, H, W), got {tuple(x.shape)}.")
		if int(x.shape[2]) < max(self.selected_channels) + 1:
			raise ValueError(f"Input has {int(x.shape[2])} channels, but fire encoder needs channel {max(self.selected_channels)}.")
		return x[:, :, self.selected_channels]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.encoder(self.extract_fire_channels(x))


class FireFrontAttentionGate(nn.Module):
	"""Learn a fire-front attention map and apply it to fused features."""

	def __init__(
		self,
		gate_input_channels: int,
		fused_channels: int,
		hidden_dim: int,
		gate_strength: float = 1.0,
		gate_mode: str = "multiplicative",
	) -> None:
		super().__init__()
		self.gate_mode = str(gate_mode).lower()
		if self.gate_mode not in {"multiplicative", "additive", "both"}:
			raise ValueError(f"Unsupported fire_gate_mode: {gate_mode!r}.")
		self.gate_strength = float(gate_strength)
		self.gate_net = nn.Sequential(
			nn.Conv2d(int(gate_input_channels), int(hidden_dim), kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(int(hidden_dim), 1, kernel_size=1),
		)
		self.gate_projection = nn.Conv2d(1, int(fused_channels), kernel_size=1) if self.gate_mode in {"additive", "both"} else None

	def forward(self, fused: torch.Tensor, gate_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if fused.ndim != 5 or gate_input.ndim != 5:
			raise ValueError("FireFrontAttentionGate expects 5D fused and gate_input tensors.")
		if fused.shape[:2] != gate_input.shape[:2] or fused.shape[-2:] != gate_input.shape[-2:]:
			raise ValueError(f"Gate input shape {tuple(gate_input.shape)} is incompatible with fused shape {tuple(fused.shape)}.")
		logits = apply_per_timestep(gate_input, self.gate_net)
		a_fire = torch.sigmoid(logits)
		gated = fused
		if self.gate_mode in {"multiplicative", "both"}:
			gated = gated * (1.0 + self.gate_strength * a_fire)
		if self.gate_mode in {"additive", "both"}:
			projected = apply_per_timestep(a_fire, self.gate_projection)  # type: ignore[arg-type]
			gated = gated + self.gate_strength * projected
		return gated, a_fire


__all__ = ["FireFrontAttentionGate", "FireFuelStateEncoder"]
