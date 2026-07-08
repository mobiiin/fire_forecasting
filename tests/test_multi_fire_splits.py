"""Tests for manual multi-fire splits and deterministic eval patch refs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.splits import build_eval_patch_refs, manual_fire_holdout_splits


class MultiFireSplitTests(unittest.TestCase):
	"""Coverage for fire-level split assignment and eval patch expansion."""

	def setUp(self) -> None:
		self.tmpdir = tempfile.TemporaryDirectory()
		self.root = Path(self.tmpdir.name)
		self.config_path = self.root / "config.yaml"
		self.config_path.write_text("test: true\n", encoding="utf-8")
		self.dataset_records = [
			{
				"dataset_id": 0,
				"dataset_name": "FIRE_A",
				"data_dir": self.root / "FIRE_A",
				"geom_path": None,
				"terrain_path": None,
				"num_files": 10,
				"raw_shape": (6, 6, 86),
			},
			{
				"dataset_id": 1,
				"dataset_name": "FIRE_B",
				"data_dir": self.root / "FIRE_B",
				"geom_path": None,
				"terrain_path": None,
				"num_files": 12,
				"raw_shape": (6, 6, 86),
			},
			{
				"dataset_id": 2,
				"dataset_name": "FIRE_C",
				"data_dir": self.root / "FIRE_C",
				"geom_path": None,
				"terrain_path": None,
				"num_files": 8,
				"raw_shape": (6, 6, 86),
			},
		]
		for record in self.dataset_records:
			Path(record["data_dir"]).mkdir(parents=True, exist_ok=True)

	def tearDown(self) -> None:
		self.tmpdir.cleanup()

	def test_manual_fire_holdout_assigns_full_fires_and_saves_json(self) -> None:
		config = {
			"config_path": str(self.config_path),
			"train_fraction": 0.7,
			"val_fraction": 0.15,
			"test_fraction": 0.15,
			"patching": {
				"enabled": True,
				"eval_mode": "sliding_window",
				"eval_patch_size": 4,
				"eval_stride": 2,
				"include_border_patches": True,
			},
			"manual_fire_split": {
				"train_fires": ["FIRE_A"],
				"val_fires": ["FIRE_B"],
				"test_fires": ["FIRE_C"],
				"use_full_fire_for_split": True,
				"require_all_listed_fires_exist": True,
				"disallow_overlap_between_splits": True,
				"require_nonempty_train": True,
				"require_nonempty_val": True,
				"require_nonempty_test": True,
				"save_resolved_split_json": True,
				"resolved_split_json": "artifacts/splits/manual_fire_split_resolved.json",
			},
		}
		splits = manual_fire_holdout_splits(
			dataset_records=self.dataset_records,
			train_fire_names=["FIRE_A"],
			val_fire_names=["FIRE_B"],
			test_fire_names=["FIRE_C"],
			input_sequence_length=3,
			prediction_horizon=1,
			config=config,
		)
		self.assertEqual(len(splits["train"]), 7)
		self.assertEqual(len(splits["val"]), 9)
		self.assertEqual(len(splits["test"]), 5)
		self.assertTrue(all(ref["dataset_name"] == "FIRE_A" for ref in splits["train"]))
		self.assertTrue(all(ref["dataset_name"] == "FIRE_B" for ref in splits["val"]))
		self.assertTrue(all(ref["dataset_name"] == "FIRE_C" for ref in splits["test"]))

		resolved_path = self.root / "artifacts/splits/manual_fire_split_resolved.json"
		self.assertTrue(resolved_path.exists())
		payload = json.loads(resolved_path.read_text(encoding="utf-8"))
		self.assertEqual(payload["split_mode"], "manual_fire_holdout")
		self.assertEqual(payload["splits"]["train"][0]["fire_name"], "FIRE_A")
		self.assertEqual(payload["splits"]["val"][0]["valid_temporal_samples"], 9)

	def test_manual_fire_holdout_rejects_overlap(self) -> None:
		config = {
			"config_path": str(self.config_path),
			"manual_fire_split": {
				"require_all_listed_fires_exist": True,
				"disallow_overlap_between_splits": True,
			},
		}
		with self.assertRaises(ValueError):
			manual_fire_holdout_splits(
				dataset_records=self.dataset_records,
				train_fire_names=["FIRE_A"],
				val_fire_names=["FIRE_A"],
				test_fire_names=["FIRE_C"],
				input_sequence_length=3,
				prediction_horizon=1,
				config=config,
			)

	def test_build_eval_patch_refs_expands_temporal_refs(self) -> None:
		config = {
			"patching": {
				"enabled": True,
				"eval_mode": "sliding_window",
				"eval_patch_size": 4,
				"eval_stride": 2,
				"include_border_patches": True,
			}
		}
		sample_refs = [
			{"dataset_id": 0, "dataset_name": "FIRE_A", "sample_index": 0, "fire_split_group": "val"},
			{"dataset_id": 0, "dataset_name": "FIRE_A", "sample_index": 1, "fire_split_group": "val"},
		]
		patch_refs = build_eval_patch_refs(
			dataset_records=self.dataset_records,
			sample_refs=sample_refs,
			config=config,
			split_name="val",
		)
		self.assertEqual(len(patch_refs), 8)
		first_patch = patch_refs[0]["patch"]
		self.assertEqual(first_patch, {"y0": 0, "y1": 4, "x0": 0, "x1": 4})
		self.assertTrue(all(ref["dataset_name"] == "FIRE_A" for ref in patch_refs))


if __name__ == "__main__":
	unittest.main()
