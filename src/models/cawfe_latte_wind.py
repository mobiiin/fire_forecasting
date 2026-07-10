"""Wind-guided modules for full CAWFE-Latte."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)

from src.models.cawfe_latte_blocks import apply_per_timestep


def _clean_channels(values: Sequence[int] | None, input_channels: int) -> list[int]:
	if not values:
		return []
	return [int(value) for value in values if 0 <= int(value) < int(input_channels)]


class WindGuidedDirectionalModule(nn.Module):
	"""Use low-level wind information to modulate fire-spread features."""

	def __init__(
		self,
		input_channels: int,
		feature_channels: int,
		hidden_dim: int,
		guidance_strength: float = 1.0,
		mode: str = "feature_modulation",
		atmosphere_start_channel: int = 0,
		atmosphere_vars_per_level: int = 10,
		low_level_indices: Sequence[int] = (0, 1, 2),
		wind_direction_cos_channels: Sequence[int] | None = None,
		wind_direction_sin_channels: Sequence[int] | None = None,
		wind_speed_channels: Sequence[int] | None = None,
		epsilon: float = 1.0e-6,
	) -> None:
		super().__init__()
		self.input_channels = int(input_channels)
		self.feature_channels = int(feature_channels)
		self.hidden_dim = int(hidden_dim)
		self.guidance_strength = float(guidance_strength)
		self.mode = str(mode).lower()
		self.atmosphere_start_channel = int(atmosphere_start_channel)
		self.atmosphere_vars_per_level = int(atmosphere_vars_per_level)
		self.low_level_indices = [int(value) for value in low_level_indices]
		self.wind_direction_cos_channels = _clean_channels(wind_direction_cos_channels, self.input_channels)
		self.wind_direction_sin_channels = _clean_channels(wind_direction_sin_channels, self.input_channels)
		self.wind_speed_channels = _clean_channels(wind_speed_channels, self.input_channels)
		self.epsilon = float(epsilon)
		if self.mode not in {"feature_modulation", "directional_convolution", "attention_bias", "combined"}:
			raise ValueError(f"Unsupported wind_guidance_mode: {mode!r}.")
		if self.mode == "attention_bias":
			raise NotImplementedError("wind_guidance_mode='attention_bias' is reserved for a future attention-bias integration.")

		self.modulation_net = nn.Sequential(
			nn.Conv2d(5, self.hidden_dim, kernel_size=3, padding=1),
			nn.GELU(),
			nn.Conv2d(self.hidden_dim, self.feature_channels, kernel_size=1),
		)
		self.horizontal_conv = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=(1, 3), padding=(0, 1), groups=self.feature_channels)
		self.vertical_conv = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=(3, 1), padding=(1, 0), groups=self.feature_channels)
		self.diagonal_conv = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1, groups=self.feature_channels)
		self.directional_project = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=1)

	def _mean_channels(self, raw_input: torch.Tensor, channels: list[int]) -> torch.Tensor | None:
		if not channels:
			return None
		return raw_input[:, :, channels].mean(dim=2, keepdim=True)

	def _fallback_low_level_wind(self, raw_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		u_channels: list[int] = []
		v_channels: list[int] = []
		for level in self.low_level_indices:
			base = self.atmosphere_start_channel + level * self.atmosphere_vars_per_level
			u_channel = base
			v_channel = base + 1
			if 0 <= u_channel < self.input_channels:
				u_channels.append(u_channel)
			if 0 <= v_channel < self.input_channels:
				v_channels.append(v_channel)
		if not u_channels or not v_channels:
			zeros = raw_input[:, :, 0:1].new_zeros(raw_input.shape[0], raw_input.shape[1], 1, raw_input.shape[-2], raw_input.shape[-1])
			return zeros, zeros
		return raw_input[:, :, u_channels].mean(dim=2, keepdim=True), raw_input[:, :, v_channels].mean(dim=2, keepdim=True)

	def _wind_summary(self, raw_input: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		u_low, v_low = self._fallback_low_level_wind(raw_input)
		speed_from_channels = self._mean_channels(raw_input, self.wind_speed_channels)
		cos_from_channels = self._mean_channels(raw_input, self.wind_direction_cos_channels)
		sin_from_channels = self._mean_channels(raw_input, self.wind_direction_sin_channels)
		speed = torch.sqrt(u_low.square() + v_low.square() + self.epsilon)
		wind_speed = speed_from_channels if speed_from_channels is not None else speed
		wind_cos = cos_from_channels if cos_from_channels is not None else u_low / (speed + self.epsilon)
		wind_sin = sin_from_channels if sin_from_channels is not None else v_low / (speed + self.epsilon)
		wind_tensor = torch.cat([u_low, v_low, wind_speed, wind_cos, wind_sin], dim=2)
		aux = {
			"wind_speed": wind_speed,
			"wind_cos": wind_cos,
			"wind_sin": wind_sin,
		}
		return wind_tensor, aux

	def _feature_modulation(self, fused_features: torch.Tensor, wind_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		modulation = torch.tanh(apply_per_timestep(wind_tensor, self.modulation_net))
		return fused_features * (1.0 + self.guidance_strength * modulation), modulation

	def _directional_convolution(self, fused_features: torch.Tensor, wind_cos: torch.Tensor, wind_sin: torch.Tensor) -> torch.Tensor:
		horizontal = apply_per_timestep(fused_features, self.horizontal_conv)
		vertical = apply_per_timestep(fused_features, self.vertical_conv)
		diagonal = apply_per_timestep(fused_features, self.diagonal_conv)
		weight_x = torch.abs(wind_cos)
		weight_y = torch.abs(wind_sin)
		weight_diag = 0.5 * torch.abs(wind_cos * wind_sin)
		directional = horizontal * weight_x + vertical * weight_y + diagonal * weight_diag
		return fused_features + self.guidance_strength * apply_per_timestep(directional, self.directional_project)

	def forward(self, fused_features: torch.Tensor, raw_input: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
		if fused_features.ndim != 5 or raw_input.ndim != 5:
			raise ValueError("WindGuidedDirectionalModule expects 5D fused_features and raw_input tensors.")
		if fused_features.shape[:2] != raw_input.shape[:2] or fused_features.shape[-2:] != raw_input.shape[-2:]:
			raise ValueError(f"Wind input shape {tuple(raw_input.shape)} is incompatible with fused shape {tuple(fused_features.shape)}.")
		wind_tensor, aux = self._wind_summary(raw_input)
		guided = fused_features
		modulation = fused_features.new_zeros(fused_features.shape)
		if self.mode in {"feature_modulation", "combined"}:
			guided, modulation = self._feature_modulation(guided, wind_tensor)
		if self.mode in {"directional_convolution", "combined"}:
			guided = self._directional_convolution(guided, aux["wind_cos"], aux["wind_sin"])
		aux["wind_guidance_map"] = modulation.mean(dim=2, keepdim=True)
		aux["wind_modulation_mean"] = modulation.mean()
		return guided, aux


__all__ = ["WindGuidedDirectionalModule"]
