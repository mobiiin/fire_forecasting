"""Flexible 2D U-Net building blocks for spatial wildfire prediction.

Expected input shape: ``(B, C, H, W)``.
Expected output shape: ``(B, out_channels, H, W)``.
"""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, Conv2d=object, BatchNorm2d=object, ReLU=object, MaxPool2d=object, ConvTranspose2d=object, Upsample=object, ModuleList=list, Sequential=object, Dropout2d=object)


class DoubleConv(nn.Module):
	"""Two convolution blocks with padding that preserves spatial size.

	Args:
		in_channels: Number of input channels.
		out_channels: Number of output channels.
		dropout: Dropout probability applied after the activations.
	"""

	def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
		super().__init__()
		if in_channels <= 0:
			raise ValueError(f"in_channels must be positive, got {in_channels}.")
		if out_channels <= 0:
			raise ValueError(f"out_channels must be positive, got {out_channels}.")
		if not 0.0 <= dropout < 1.0:
			raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

		layers = [
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		]
		if dropout > 0.0:
			layers.append(nn.Dropout2d(p=dropout))
		self.block = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply two padded convolutions while preserving H and W."""

		if x.ndim != 4:
			raise ValueError(f"DoubleConv expects a 4D tensor, got shape {tuple(x.shape)}.")
		return self.block(x)


class Down(nn.Module):
	"""Downsampling block: max-pool followed by DoubleConv."""

	def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2),
			DoubleConv(in_channels, out_channels, dropout=dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Downsample the feature map by a factor of 2."""

		if x.ndim != 4:
			raise ValueError(f"Down expects a 4D tensor, got shape {tuple(x.shape)}.")
		return self.block(x)


class Up(nn.Module):
	"""Upsampling block with skip-connection alignment.

	When ``bilinear=True``, the block upsamples using interpolation and then
	compresses channels with a DoubleConv. Otherwise it uses a transpose
	convolution followed by DoubleConv.
	"""

	def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.0) -> None:
		super().__init__()
		if in_channels <= 0:
			raise ValueError(f"in_channels must be positive, got {in_channels}.")
		if out_channels <= 0:
			raise ValueError(f"out_channels must be positive, got {out_channels}.")

		self.bilinear = bool(bilinear)
		if self.bilinear:
			self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
			self.conv = DoubleConv(in_channels + out_channels, out_channels, dropout=dropout)
		else:
			self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
			self.conv = DoubleConv(out_channels * 2, out_channels, dropout=dropout)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		"""Upsample ``x`` and fuse it with ``skip`` using shape-safe padding."""

		if x.ndim != 4 or skip.ndim != 4:
			raise ValueError("Up expects 4D tensors shaped (B, C, H, W).")

		x = self.up(x)
		height_diff = skip.size(2) - x.size(2)
		width_diff = skip.size(3) - x.size(3)

		if height_diff != 0 or width_diff != 0:
			x = nn.functional.pad(
				x,
				[
					width_diff // 2,
					width_diff - width_diff // 2,
					height_diff // 2,
					height_diff - height_diff // 2,
				],
			)

		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	"""Final 1x1 projection to the requested output channels."""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		if in_channels <= 0:
			raise ValueError(f"in_channels must be positive, got {in_channels}.")
		if out_channels <= 0:
			raise ValueError(f"out_channels must be positive, got {out_channels}.")
		self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Project feature maps to the output channel dimension."""

		if x.ndim != 4:
			raise ValueError(f"OutConv expects a 4D tensor, got shape {tuple(x.shape)}.")
		return self.proj(x)


class UNet2D(nn.Module):
	"""Flexible 2D U-Net.

	Args:
		in_channels: Number of input channels.
		out_channels: Number of output channels.
		base_channels: Width of the first encoder stage.
		depth: Number of encoder/decoder levels.
		bilinear: If ``True``, use bilinear upsampling; otherwise use transpose conv.
		dropout: Dropout probability applied inside DoubleConv blocks.
	"""

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		base_channels: int = 64,
		depth: int = 4,
		bilinear: bool = True,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		if in_channels <= 0:
			raise ValueError(f"in_channels must be positive, got {in_channels}.")
		if out_channels <= 0:
			raise ValueError(f"out_channels must be positive, got {out_channels}.")
		if base_channels <= 0:
			raise ValueError(f"base_channels must be positive, got {base_channels}.")
		if depth < 1:
			raise ValueError(f"depth must be at least 1, got {depth}.")

		self.in_channels = int(in_channels)
		self.out_channels = int(out_channels)
		self.base_channels = int(base_channels)
		self.depth = int(depth)
		self.bilinear = bool(bilinear)
		self.dropout = float(dropout)

		encoder_channels = [self.base_channels * (2**level) for level in range(self.depth)]

		self.inc = DoubleConv(self.in_channels, encoder_channels[0], dropout=self.dropout)
		self.downs = nn.ModuleList()

		for level in range(1, self.depth):
			self.downs.append(Down(encoder_channels[level - 1], encoder_channels[level], dropout=self.dropout))

		self.bottleneck = DoubleConv(encoder_channels[-1], encoder_channels[-1] * 2, dropout=self.dropout)

		decoder_in_channels = encoder_channels[-1] * 2
		self.ups = nn.ModuleList()
		for level in reversed(range(self.depth)):
			skip_channels = encoder_channels[level]
			self.ups.append(Up(decoder_in_channels, skip_channels, bilinear=self.bilinear, dropout=self.dropout))
			decoder_in_channels = skip_channels

		self.outc = OutConv(decoder_in_channels, self.out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Run the U-Net on a 4D tensor shaped ``(B, C, H, W)``."""

		if x.ndim != 4:
			raise ValueError(f"UNet2D expects a 4D tensor, got shape {tuple(x.shape)}.")
		if x.shape[1] != self.in_channels:
			raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")

		skips = []
		x = self.inc(x)
		skips.append(x)

		for down in self.downs:
			x = down(x)
			skips.append(x)

		x = self.bottleneck(x)

		for up, skip in zip(self.ups, reversed(skips)):
			x = up(x, skip)

		return self.outc(x)


if __name__ == "__main__":
	if torch is None:
		print("UNet2D smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	x = torch.randn(2, 64, 144, 144)
	model = UNet2D(in_channels=64, out_channels=1)
	y = model(x)
	print(y.shape)
	assert y.shape == (2, 1, 144, 144)
