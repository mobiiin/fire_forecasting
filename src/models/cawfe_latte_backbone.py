"""Hybrid Transformer + Mamba backbone for CAWFE-Latte-Lite."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	from torch.utils.checkpoint import checkpoint  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object, ModuleList=list)
	checkpoint = None

from src.models.cawfe_latte_blocks import _make_group_norm, apply_per_timestep
from src.models.earthformer_blocks import StochasticDepth
from src.models.mamba_backend import build_mamba_layer
from src.models.st_mamba_lite_blocks import ChannelLayerNorm5D, ChannelMLP3D
from src.models.window_attention import WindowSelfAttention


class TriAxisMamba(nn.Module):
	"""Apply Mamba-compatible sequence mixers over time, width, and height axes."""

	def __init__(
		self,
		channels: int,
		mamba_backend: str,
		d_state: int,
		d_conv: int,
		expand: int,
		dropout: float = 0.0,
		scan_mode: str = "tri_axis",
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.scan_mode = str(scan_mode).lower()
		if self.scan_mode not in {"temporal", "spatial", "tri_axis"}:
			raise ValueError(f"Unsupported mamba_scan_mode: {scan_mode!r}.")
		self.temporal_layer = build_mamba_layer(self.channels, int(d_state), int(d_conv), int(expand), str(mamba_backend), dropout=float(dropout))
		self.width_layer = build_mamba_layer(self.channels, int(d_state), int(d_conv), int(expand), str(mamba_backend), dropout=float(dropout))
		self.height_layer = build_mamba_layer(self.channels, int(d_state), int(d_conv), int(expand), str(mamba_backend), dropout=float(dropout))
		self.backend_name = str(getattr(self.temporal_layer, "backend_name", "unknown"))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"TriAxisMamba expects (B, T, C, H, W), got {tuple(x.shape)}.")
		batch_size, time_steps, channels, height, width = tuple(int(value) for value in x.shape)
		if channels != self.channels:
			raise ValueError(f"TriAxisMamba expected channel dim {self.channels}, got {channels}.")
		outputs: list[torch.Tensor] = []
		if self.scan_mode in {"temporal", "tri_axis"}:
			seq = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, time_steps, channels)
			mixed = self.temporal_layer(seq)
			outputs.append(mixed.reshape(batch_size, height, width, time_steps, channels).permute(0, 3, 4, 1, 2).contiguous())
		if self.scan_mode in {"spatial", "tri_axis"}:
			seq_w = x.permute(0, 1, 3, 4, 2).reshape(batch_size * time_steps * height, width, channels)
			mixed_w = self.width_layer(seq_w)
			outputs.append(mixed_w.reshape(batch_size, time_steps, height, width, channels).permute(0, 1, 4, 2, 3).contiguous())
			seq_h = x.permute(0, 1, 4, 3, 2).reshape(batch_size * time_steps * width, height, channels)
			mixed_h = self.height_layer(seq_h)
			outputs.append(mixed_h.reshape(batch_size, time_steps, width, height, channels).permute(0, 1, 4, 3, 2).contiguous())
		return torch.stack(outputs, dim=0).mean(dim=0)


class ConvOnlyBlock(nn.Module):
	"""ConvNeXt-like ablation block for sequence tensors."""

	def __init__(self, channels: int, mlp_ratio: float, dropout: float = 0.0) -> None:
		super().__init__()
		hidden_channels = max(int(round(float(channels) * float(mlp_ratio))), 1)
		self.net = nn.Sequential(
			nn.Conv3d(int(channels), int(channels), kernel_size=3, padding=1, groups=int(channels)),
			nn.GELU(),
			nn.Conv3d(int(channels), hidden_channels, kernel_size=1),
			nn.GELU(),
			nn.Dropout(float(dropout)),
			nn.Conv3d(hidden_channels, int(channels), kernel_size=1),
			nn.Dropout(float(dropout)),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = x.permute(0, 2, 1, 3, 4).contiguous()
		y = self.net(y)
		return y.permute(0, 2, 1, 3, 4).contiguous()


class LatteHybridBlock(nn.Module):
	"""Residual local-window-attention plus tri-axis-Mamba block."""

	def __init__(
		self,
		channels: int,
		num_heads: int,
		window_size: int,
		shifted_window: bool,
		backbone_type: str,
		mamba_backend: str,
		mamba_d_state: int,
		mamba_d_conv: int,
		mamba_expand: int,
		mamba_scan_mode: str,
		mlp_ratio: float,
		dropout: float,
		attention_dropout: float,
		drop_path: float,
		gradient_checkpointing: bool = False,
	) -> None:
		super().__init__()
		self.channels = int(channels)
		self.backbone_type = str(backbone_type).lower()
		self.gradient_checkpointing = bool(gradient_checkpointing)
		if self.backbone_type not in {"transformer_only", "mamba_only", "hybrid_transformer_mamba", "conv_only"}:
			raise ValueError(f"Unsupported backbone_type: {backbone_type!r}.")
		self.use_attention = self.backbone_type in {"transformer_only", "hybrid_transformer_mamba"}
		self.use_mamba = self.backbone_type in {"mamba_only", "hybrid_transformer_mamba"}
		self.use_conv = self.backbone_type == "conv_only"

		self.norm1 = ChannelLayerNorm5D(self.channels)
		self.window_attention = (
			WindowSelfAttention(
				dim=self.channels,
				num_heads=int(num_heads),
				window_size=int(window_size),
				attention_dropout=float(attention_dropout),
				shift_size=int(window_size) // 2 if bool(shifted_window) else 0,
				use_global_tokens=False,
				num_global_tokens=0,
			)
			if self.use_attention
			else None
		)
		self.norm2 = ChannelLayerNorm5D(self.channels)
		self.tri_axis_mamba = (
			TriAxisMamba(
				channels=self.channels,
				mamba_backend=str(mamba_backend),
				d_state=int(mamba_d_state),
				d_conv=int(mamba_d_conv),
				expand=int(mamba_expand),
				dropout=float(dropout),
				scan_mode=str(mamba_scan_mode),
			)
			if self.use_mamba
			else None
		)
		self.conv_only = ConvOnlyBlock(self.channels, mlp_ratio=float(mlp_ratio), dropout=float(dropout)) if self.use_conv else None
		self.norm3 = ChannelLayerNorm5D(self.channels)
		self.channel_mlp = ChannelMLP3D(self.channels, mlp_ratio=float(mlp_ratio), dropout=float(dropout))
		self.drop_path = StochasticDepth(float(drop_path))
		if self.tri_axis_mamba is not None:
			self.backend_name = self.tri_axis_mamba.backend_name
		else:
			self.backend_name = "none"

	def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
		if self.window_attention is not None:
			x = x + self.drop_path(self.window_attention(self.norm1(x)))
		if self.tri_axis_mamba is not None:
			x = x + self.drop_path(self.tri_axis_mamba(self.norm2(x)))
		if self.conv_only is not None:
			x = x + self.drop_path(self.conv_only(self.norm2(x)))
		x = x + self.drop_path(self.channel_mlp(self.norm3(x)))
		return x

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.ndim != 5:
			raise ValueError(f"LatteHybridBlock expects a 5D tensor, got {tuple(x.shape)}.")
		if int(x.shape[2]) != self.channels:
			raise ValueError(f"LatteHybridBlock expected channel dim {self.channels}, got {int(x.shape[2])}.")
		if self.gradient_checkpointing and self.training and checkpoint is not None:
			return checkpoint(self._forward_impl, x, use_reentrant=False)
		return self._forward_impl(x)


class HybridTransformerMambaBackbone(nn.Module):
	"""Two-stage CAWFE-Latte-Lite backbone."""

	def __init__(
		self,
		backbone_dim: int,
		backbone_depths: Sequence[int],
		num_heads: Sequence[int],
		window_size: int,
		shifted_window: bool,
		backbone_type: str,
		mamba_backend: str,
		mamba_d_state: int,
		mamba_d_conv: int,
		mamba_expand: int,
		mamba_scan_mode: str,
		mlp_ratio: float,
		dropout: float,
		attention_dropout: float,
		drop_path: float,
		gradient_checkpointing: bool = False,
	) -> None:
		super().__init__()
		self.backbone_dim = int(backbone_dim)
		self.stage2_dim = self.backbone_dim * 2
		self.backbone_depths = [int(value) for value in backbone_depths]
		self.num_heads = [int(value) for value in num_heads]
		self.window_size = int(window_size)
		if len(self.backbone_depths) != 2:
			raise ValueError(f"backbone_depths must contain exactly 2 values, got {self.backbone_depths}.")
		if len(self.num_heads) != 2:
			raise ValueError(f"num_heads must contain exactly 2 values, got {self.num_heads}.")

		total_blocks = sum(self.backbone_depths)
		block_index = 0
		self.stage1_blocks = nn.ModuleList()
		for local_index in range(self.backbone_depths[0]):
			self.stage1_blocks.append(
				LatteHybridBlock(
					channels=self.backbone_dim,
					num_heads=self.num_heads[0],
					window_size=self.window_size,
					shifted_window=bool(shifted_window and local_index % 2 == 1),
					backbone_type=str(backbone_type),
					mamba_backend=str(mamba_backend),
					mamba_d_state=int(mamba_d_state),
					mamba_d_conv=int(mamba_d_conv),
					mamba_expand=int(mamba_expand),
					mamba_scan_mode=str(mamba_scan_mode),
					mlp_ratio=float(mlp_ratio),
					dropout=float(dropout),
					attention_dropout=float(attention_dropout),
					drop_path=float(drop_path) * block_index / max(total_blocks - 1, 1),
					gradient_checkpointing=bool(gradient_checkpointing),
				)
			)
			block_index += 1
		self.downsample = nn.Sequential(
			nn.Conv2d(self.backbone_dim, self.stage2_dim, kernel_size=3, stride=2, padding=1),
			_make_group_norm(self.stage2_dim),
			nn.GELU(),
		)
		self.stage2_blocks = nn.ModuleList()
		for local_index in range(self.backbone_depths[1]):
			self.stage2_blocks.append(
				LatteHybridBlock(
					channels=self.stage2_dim,
					num_heads=self.num_heads[1],
					window_size=self.window_size,
					shifted_window=bool(shifted_window and local_index % 2 == 1),
					backbone_type=str(backbone_type),
					mamba_backend=str(mamba_backend),
					mamba_d_state=int(mamba_d_state),
					mamba_d_conv=int(mamba_d_conv),
					mamba_expand=int(mamba_expand),
					mamba_scan_mode=str(mamba_scan_mode),
					mlp_ratio=float(mlp_ratio),
					dropout=float(dropout),
					attention_dropout=float(attention_dropout),
					drop_path=float(drop_path) * block_index / max(total_blocks - 1, 1),
					gradient_checkpointing=bool(gradient_checkpointing),
				)
			)
			block_index += 1
		self.mamba_backend_used = self._infer_backend()

	def _infer_backend(self) -> str:
		for block in list(self.stage1_blocks) + list(self.stage2_blocks):
			backend = str(getattr(block, "backend_name", "none"))
			if backend != "none":
				return backend
		return "none"

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if x.ndim != 5:
			raise ValueError(f"HybridTransformerMambaBackbone expects a 5D tensor, got {tuple(x.shape)}.")
		stage1 = x
		for block in self.stage1_blocks:
			stage1 = block(stage1)
		stage2 = apply_per_timestep(stage1, self.downsample)
		for block in self.stage2_blocks:
			stage2 = block(stage2)
		return stage1, stage2


__all__ = ["HybridTransformerMambaBackbone", "LatteHybridBlock", "TriAxisMamba"]
