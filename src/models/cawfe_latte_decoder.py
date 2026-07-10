"""Decoder for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None

from src.models.cawfe_latte_blocks import _make_group_norm


class MultiTaskFireDecoder(nn.Module):
	"""U-Net-style dense decoder with shared or separate task heads."""

	def __init__(
		self,
		stage1_channels: int,
		stage2_channels: int,
		decoder_channels: list[int] | tuple[int, ...],
		output_channels: int = 4,
		use_skip_connections: bool = True,
		upsample_mode: str = "bilinear",
		decoder_task_heads: str = "separate",
	) -> None:
		super().__init__()
		self.stage1_channels = int(stage1_channels)
		self.stage2_channels = int(stage2_channels)
		self.decoder_channels = [int(value) for value in decoder_channels]
		if len(self.decoder_channels) < 2:
			raise ValueError(f"decoder_channels must contain at least 2 values, got {decoder_channels!r}.")
		self.output_channels = int(output_channels)
		self.use_skip_connections = bool(use_skip_connections)
		self.upsample_mode = str(upsample_mode).lower()
		self.decoder_task_heads = str(decoder_task_heads).lower()
		if self.upsample_mode not in {"nearest", "bilinear", "convtranspose"}:
			raise ValueError(f"Unsupported upsample_mode: {upsample_mode!r}.")
		if self.decoder_task_heads not in {"shared", "separate"}:
			raise ValueError(f"Unsupported decoder_task_heads: {decoder_task_heads!r}.")

		if self.upsample_mode == "convtranspose":
			self.upsample = nn.ConvTranspose2d(self.stage2_channels, self.decoder_channels[0], kernel_size=2, stride=2)
			self.low_res_project = None
		else:
			self.upsample = None
			self.low_res_project = nn.Conv2d(self.stage2_channels, self.decoder_channels[0], kernel_size=1)

		decoder_input_channels = self.decoder_channels[0] + (self.stage1_channels if self.use_skip_connections else 0)
		decoder_layers: list[nn.Module] = []
		in_channels = decoder_input_channels
		for out_channels in self.decoder_channels[1:]:
			decoder_layers.extend(
				[
					nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
					_make_group_norm(out_channels),
					nn.GELU(),
				]
			)
			in_channels = out_channels
		self.decoder = nn.Sequential(*decoder_layers)
		head_channels = self.decoder_channels[-1]
		self.shared = nn.Sequential(
			nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
			nn.GELU(),
		)
		if self.decoder_task_heads == "shared":
			self.output_head = nn.Conv2d(head_channels, self.output_channels, kernel_size=1)
		else:
			self.surface_head = nn.Conv2d(head_channels, 1, kernel_size=1)
			self.canopy_head = nn.Conv2d(head_channels, 1, kernel_size=1)
			self.mask_head = nn.Conv2d(head_channels, 1, kernel_size=1)
			self.energy_head = nn.Conv2d(head_channels, 1, kernel_size=1)

	def _upsample_bottleneck(self, z2: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
		if self.upsample is not None:
			return self.upsample(z2)
		projected = self.low_res_project(z2)  # type: ignore[operator]
		if self.upsample_mode == "bilinear":
			return F.interpolate(projected, size=target_size, mode=self.upsample_mode, align_corners=False)
		return F.interpolate(projected, size=target_size, mode=self.upsample_mode)

	def forward(self, z2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
		if z2.ndim != 4 or z1.ndim != 4:
			raise ValueError("MultiTaskFireDecoder expects 4D z2 and z1 feature maps.")
		upsampled = self._upsample_bottleneck(z2, target_size=tuple(int(value) for value in z1.shape[-2:]))
		if self.use_skip_connections:
			decoded = self.decoder(torch.cat([upsampled, z1], dim=1))
		else:
			decoded = self.decoder(upsampled)
		shared = self.shared(decoded)
		if self.decoder_task_heads == "shared":
			return self.output_head(shared)
		return torch.cat(
			[
				self.surface_head(shared),
				self.canopy_head(shared),
				self.mask_head(shared),
				self.energy_head(shared),
			],
			dim=1,
		)


__all__ = ["MultiTaskFireDecoder"]
