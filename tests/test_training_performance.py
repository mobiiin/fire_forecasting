from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.training import hardware


def test_hardware_detection_does_not_crash_without_cuda() -> None:
	info = hardware.get_cuda_device_info()
	assert "available" in info
	assert "device_count" in info


def test_choose_amp_dtype_prefers_bf16_on_mocked_ampere(monkeypatch: pytest.MonkeyPatch) -> None:
	bfloat16 = object()
	float16 = object()
	fake_torch = SimpleNamespace(
		bfloat16=bfloat16,
		float16=float16,
		cuda=SimpleNamespace(
			is_available=lambda: True,
			is_bf16_supported=lambda: True,
			current_device=lambda: 0,
			get_device_capability=lambda device=0: (8, 0),
		),
	)
	monkeypatch.setattr(hardware, "torch", fake_torch)

	assert hardware.choose_amp_dtype({"training": {"performance": {"precision": "auto"}}}, "cuda") is bfloat16


def test_dataloader_worker_capping_respects_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")

	assert hardware.cap_num_workers_by_slurm({"training": {"auto_cap_num_workers_to_slurm_cpus": True}}, 8) == 4
	assert hardware.cap_num_workers_by_slurm({"training": {"auto_cap_num_workers_to_slurm_cpus": True}}, "auto") == 4


def test_host_memory_cap_limits_auto_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
	from src.training.train import _cap_batch_size_for_host_memory

	monkeypatch.setenv("SLURM_MEM_PER_NODE", "16")
	config = {
		"input_sequence_length": 1,
		"patching": {"patch_size": 64, "patch_height": 64, "patch_width": 64},
		"model": {"input_channels": 1, "output_channels": 1},
		"training": {"num_workers": 1, "pin_memory": False, "prefetch_factor": 1},
	}

	capped = _cap_batch_size_for_host_memory(
		config,
		batch_size=1000,
		auto_config={"max_host_memory_fraction": 0.5, "host_memory_sample_multiplier": 1.0},
		logger=None,
	)

	assert capped < 1000


def test_input_normalization_matches_expected_tensor_math() -> None:
	torch = pytest.importorskip("torch")
	from src.training.train import _apply_input_normalizer

	x = torch.arange(2 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 2, 3, 2, 2)
	mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).reshape(1, 1, 3, 1, 1)
	std = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32).reshape(1, 1, 3, 1, 1)
	expected = (x.clone() - mean) / std

	actual = _apply_input_normalizer(x.clone(), {"mean": mean, "std": std})

	torch.testing.assert_close(actual, expected)


class _FakeCachedDataset:
	def __init__(self) -> None:
		self.shards = [
			{"num_samples": 5},
			{"num_samples": 4},
		]
		self._offsets = [0, 5, 9]

	def __len__(self) -> int:
		return 9


def test_shard_local_batch_sampler_traverses_every_sample_once() -> None:
	pytest.importorskip("torch")
	from src.data.cached_patch_dataset import CachedShardBatchSampler

	sampler = CachedShardBatchSampler(
		_FakeCachedDataset(),
		batch_size=3,
		drop_last=False,
		shuffle_shards=False,
		shuffle_within_shard=False,
	)

	indices = [index for batch in sampler for index in batch]

	assert indices == list(range(9))
	assert len(sampler) == 4


def test_shard_local_batch_sampler_groups_samples_by_shard() -> None:
	pytest.importorskip("torch")
	from src.data.cached_patch_dataset import CachedShardBatchSampler

	sampler = CachedShardBatchSampler(
		_FakeCachedDataset(),
		batch_size=3,
		drop_last=False,
		shuffle_shards=False,
		shuffle_within_shard=False,
	)

	for batch in sampler:
		assert all(index < 5 for index in batch) or all(index >= 5 for index in batch)


def test_run_epoch_respects_max_batches() -> None:
	torch = pytest.importorskip("torch")
	from src.training.train import _run_epoch

	class ToyModel(torch.nn.Module):
		def __init__(self) -> None:
			super().__init__()
			self.scale = torch.nn.Parameter(torch.tensor(1.0))

		def forward(self, x):
			return x[:, -1, :1] * self.scale

	loader = [
		(torch.ones(2, 2, 1, 4, 4), torch.zeros(2, 1, 4, 4))
		for _ in range(5)
	]
	model = ToyModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	results = _run_epoch(
		model=model,
		loader=loader,
		criterion=torch.nn.MSELoss(),
		config={"training": {"performance": {"log_timing": False}}},
		device=torch.device("cpu"),
		input_sequence_length=2,
		input_channels=1,
		output_channels=1,
		train=True,
		optimizer=optimizer,
		max_batches=2,
	)

	assert results["train_batches"] == 2.0
	assert results["train_samples"] == 4.0
