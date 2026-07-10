"""Neural-operator bottleneck for full CAWFE-Latte."""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, ModuleList=list)
	F = None

from src.models.cawfe_latte_blocks import _make_group_norm


def _largest_divisor_at_most(value: int, limit: int) -> int:
	for candidate in range(min(int(value), int(limit)), 0, -1):
		if int(value) % candidate == 0:
			return candidate
	return 1


class AFNO2DBlock(nn.Module):
	"""A compact AFNO-style spectral residual block."""

	def __init__(
		self,
		channels: int,
		num_blocks: int,
		sparsity_threshold: float,
		hard_thresholding_fraction: float,
		hidden_factor: int,
		force_float32_fft: bool = True,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.num_blocks = _largest_divisor_at_most(self.channels, int(num_blocks))
		self.block_size = self.channels // self.num_blocks
		self.sparsity_threshold = float(sparsity_threshold)
		self.hard_thresholding_fraction = float(hard_thresholding_fraction)
		self.force_float32_fft = bool(force_float32_fft)
		self.weight_real = nn.Parameter(torch.randn(self.num_blocks, self.block_size, self.block_size) * 0.02)
		self.weight_imag = nn.Parameter(torch.randn(self.num_blocks, self.block_size, self.block_size) * 0.02)
		hidden_channels = max(self.channels * int(hidden_factor), 1)
		self.mlp = nn.Sequential(
			nn.Conv2d(self.channels, hidden_channels, kernel_size=1),
			nn.GELU(),
			nn.Conv2d(hidden_channels, self.channels, kernel_size=1),
		)
		self.norm = _make_group_norm(self.channels)

	def _spectral_mix(self, x: torch.Tensor) -> torch.Tensor:
		original_dtype = x.dtype
		fft_input = x.float() if self.force_float32_fft else x
		height, width = tuple(int(value) for value in fft_input.shape[-2:])
		coeffs = torch.fft.rfft2(fft_input, norm="ortho")
		num_freq_w = int(coeffs.shape[-1])
		kept_w = max(1, int(num_freq_w * self.hard_thresholding_fraction))
		mixed = torch.zeros_like(coeffs)
		active = coeffs[:, :, :, :kept_w]
		active = active.reshape(active.shape[0], self.num_blocks, self.block_size, height, kept_w)
		real = torch.einsum("nbihw,boi->nbohw", active.real, self.weight_real) - torch.einsum("nbihw,boi->nbohw", active.imag, self.weight_imag)
		imag = torch.einsum("nbihw,boi->nbohw", active.real, self.weight_imag) + torch.einsum("nbihw,boi->nbohw", active.imag, self.weight_real)
		updated = torch.complex(real, imag).reshape(coeffs.shape[0], self.channels, height, kept_w)
		if self.sparsity_threshold > 0:
			updated = torch.complex(
				F.softshrink(updated.real, lambd=self.sparsity_threshold),
				F.softshrink(updated.imag, lambd=self.sparsity_threshold),
			)
		mixed[:, :, :, :kept_w] = updated
		out = torch.fft.irfft2(mixed, s=(height, width), norm="ortho")
		return out.to(dtype=original_dtype)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4:
			raise ValueError(f"AFNO2DBlock expects a 4D tensor, got {tuple(x.shape)}.")
		residual = x
		mixed = self._spectral_mix(x)
		x = residual + mixed
		if self.force_float32_fft and x.dtype != torch.float32:
			mlp_out = self.mlp(self.norm(x.float())).to(dtype=x.dtype)
		else:
			mlp_out = self.mlp(self.norm(x))
		x = x + mlp_out
		return x


class NeuralOperatorBottleneck(nn.Module):
	"""Apply global spectral field mixing to a sequence bottleneck."""

	def __init__(
		self,
		channels: int,
		operator_type: str = "afno",
		depth: int = 2,
		num_blocks: int = 8,
		sparsity_threshold: float = 0.01,
		hard_thresholding_fraction: float = 1.0,
		hidden_factor: int = 1,
		force_float32_fft: bool = True,
		enabled: bool = True,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.operator_type = str(operator_type).lower()
		self.enabled = bool(enabled) and self.operator_type != "none"
		if self.operator_type not in {"afno", "fno", "spectral_conv", "none"}:
			raise ValueError(f"Unsupported neural_operator_type: {operator_type!r}.")
		if self.operator_type in {"fno", "spectral_conv"}:
			raise NotImplementedError(f"neural_operator_type={self.operator_type!r} is reserved for a future implementation.")
		self.blocks = nn.ModuleList()
		if self.enabled:
			for _ in range(int(depth)):
				self.blocks.append(
					AFNO2DBlock(
						channels=self.channels,
						num_blocks=int(num_blocks),
						sparsity_threshold=float(sparsity_threshold),
						hard_thresholding_fraction=float(hard_thresholding_fraction),
						hidden_factor=int(hidden_factor),
						force_float32_fft=bool(force_float32_fft),
					)
				)
		self.last_stats: dict[str, torch.Tensor] = {}

	def forward(self, h: torch.Tensor, return_aux: bool = False):
		if h.ndim != 5:
			raise ValueError(f"NeuralOperatorBottleneck expects (B, T, C, H, W), got {tuple(h.shape)}.")
		if int(h.shape[2]) != self.channels:
			raise ValueError(f"NeuralOperatorBottleneck expected channel dim {self.channels}, got {int(h.shape[2])}.")
		if not self.enabled:
			aux = {
				"operator_input_mean": h.mean(),
				"operator_output_mean": h.mean(),
				"operator_residual_norm": h.new_tensor(0.0),
			}
			self.last_stats = aux
			return (h, aux) if return_aux else h
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in h.shape)
		x = h.reshape(batch_size * time_steps, channels, height, width)
		input_mean = x.mean()
		residual = x
		for block in self.blocks:
			x = block(x)
		output_mean = x.mean()
		residual_norm = torch.mean((x - residual).square()).sqrt()
		out = x.reshape(batch_size, time_steps, channels, height, width)
		aux = {
			"operator_input_mean": input_mean,
			"operator_output_mean": output_mean,
			"operator_residual_norm": residual_norm,
		}
		self.last_stats = aux
		return (out, aux) if return_aux else out


__all__ = ["NeuralOperatorBottleneck"]
