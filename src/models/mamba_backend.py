"""Backend helpers for ST-Mamba-style sequence mixing."""

from __future__ import annotations

import warnings

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)


class FallbackGatedSSM(nn.Module):
	"""Fallback gated sequence block used when ``mamba_ssm`` is unavailable.

	This is not official Mamba. It exists so the project can run smoke tests and
	debug architecture plumbing without requiring CUDA-specific extensions.
	"""

	def __init__(
		self,
		d_model: int,
		d_state: int,
		d_conv: int,
		expand: int,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		if d_model <= 0:
			raise ValueError(f"d_model must be positive, got {d_model}.")
		if d_conv <= 0:
			raise ValueError(f"d_conv must be positive, got {d_conv}.")
		if expand <= 0:
			raise ValueError(f"expand must be positive, got {expand}.")
		self.d_model = int(d_model)
		self.d_state = int(d_state)
		self.d_conv = int(d_conv)
		self.expand = int(expand)
		self.hidden_dim = self.d_model * self.expand

		self.norm = nn.LayerNorm(self.d_model)
		self.in_proj = nn.Linear(self.d_model, 2 * self.hidden_dim)
		self.depthwise_conv = nn.Conv1d(
			in_channels=self.hidden_dim,
			out_channels=self.hidden_dim,
			kernel_size=self.d_conv,
			padding=self.d_conv // 2,
			groups=self.hidden_dim,
		)
		self.out_proj = nn.Linear(self.hidden_dim, self.d_model)
		self.dropout = nn.Dropout(float(dropout))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 3:
			raise ValueError(f"FallbackGatedSSM expects a 3D tensor shaped (B, L, C), got {tuple(x.shape)}.")
		if int(x.shape[-1]) != self.d_model:
			raise ValueError(f"FallbackGatedSSM expected last dim {self.d_model}, got {int(x.shape[-1])}.")
		normalized = self.norm(x)
		value_gate = self.in_proj(normalized)
		value, gate = torch.chunk(value_gate, chunks=2, dim=-1)
		sequence_length = int(value.shape[1])
		value = value.transpose(1, 2)
		value = self.depthwise_conv(value).transpose(1, 2)
		if int(value.shape[1]) != sequence_length:
			value = value[:, :sequence_length, :]
		value = torch.nn.functional.silu(value)
		gated = value * torch.sigmoid(gate)
		return self.dropout(self.out_proj(gated))


class MambaLayerWrapper(nn.Module):
	"""Thin wrapper around ``mamba_ssm.Mamba`` with a consistent interface."""

	def __init__(
		self,
		d_model: int,
		d_state: int,
		d_conv: int,
		expand: int,
		mamba_cls,
	) -> None:
		super().__init__()
		self.d_model = int(d_model)
		self.layer = mamba_cls(
			d_model=int(d_model),
			d_state=int(d_state),
			d_conv=int(d_conv),
			expand=int(expand),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 3:
			raise ValueError(f"MambaLayerWrapper expects a 3D tensor shaped (B, L, C), got {tuple(x.shape)}.")
		if int(x.shape[-1]) != self.d_model:
			raise ValueError(f"MambaLayerWrapper expected last dim {self.d_model}, got {int(x.shape[-1])}.")
		return self.layer(x)


def build_mamba_layer(
	d_model: int,
	d_state: int,
	d_conv: int,
	expand: int,
	backend: str,
	dropout: float = 0.0,
) -> nn.Module:
	"""Build one resolved Mamba-compatible layer.

	Supported backends:
	- ``mamba_ssm``
	- ``fallback``
	- ``auto``: prefer ``mamba_ssm`` and fall back with a warning
	"""

	backend_name = str(backend).lower()
	if backend_name not in {"auto", "mamba_ssm", "fallback"}:
		raise ValueError(f"Unsupported mamba backend: {backend!r}.")

	if backend_name in {"auto", "mamba_ssm"}:
		try:
			from mamba_ssm import Mamba  # type: ignore[import-not-found]
		except ImportError as exc:
			if backend_name == "mamba_ssm":
				raise ImportError(
					"mamba-ssm is not installed. Install it with `pip install mamba-ssm` "
					"or set st_mamba_lite.mamba_backend: fallback for smoke testing only."
				) from exc
			warnings.warn(
				"mamba-ssm is not installed. Using fallback gated SSM block. "
				"This is not official Mamba.",
				RuntimeWarning,
				stacklevel=2,
			)
		else:
			layer = MambaLayerWrapper(
				d_model=int(d_model),
				d_state=int(d_state),
				d_conv=int(d_conv),
				expand=int(expand),
				mamba_cls=Mamba,
			)
			layer.backend_name = "mamba_ssm"  # type: ignore[attr-defined]
			return layer

	layer = FallbackGatedSSM(
		d_model=int(d_model),
		d_state=int(d_state),
		d_conv=int(d_conv),
		expand=int(expand),
		dropout=float(dropout),
	)
	layer.backend_name = "fallback"  # type: ignore[attr-defined]
	return layer
