"""Unit tests for fire discovery, geometry parsing, and energy release."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.energy_release import compute_energy_release_maps
from src.data.fire_index import (
	discover_fire_datasets,
	load_fire_dataset_index,
	save_fire_dataset_index,
	update_fire_dataset_index,
)
from src.data.geometry import compute_cell_area_from_geom, parse_geom_file


def _write_fake_geom(path: Path, nx: int, ny: int, nz: int, lons: list[float], lats: list[float]) -> None:
	cos_lats = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
	values = [str(nx), str(ny), str(nz)]
	values.extend(str(value) for value in lons)
	values.extend(str(value) for value in lats)
	values.extend(str(float(value)) for value in cos_lats)
	path.write_text("\n".join(values) + "\n", encoding="utf-8")


class FireIndexGeometryTests(unittest.TestCase):
	"""Coverage for fire indexing and geometry-based energy release."""

	def setUp(self) -> None:
		self.tmpdir = tempfile.TemporaryDirectory()
		self.root = Path(self.tmpdir.name)
		self.config = {
			"geometry": {
				"earth_radius_m": 6_371_000.0,
				"use_geom_cos_latitude": False,
				"geom_cos_tolerance": 1.0e-5,
				"spacing_tolerance_relative": 1.0e-4,
				"validate_against_terrain_header": False,
				"allow_area_transpose_if_needed": False,
			},
			"energy_release": {
				"enabled": True,
				"surface_sensible_flux_channel": 80,
				"surface_latent_flux_channel": 81,
				"canopy_sensible_flux_channel": 82,
				"canopy_latent_flux_channel": 83,
				"flux_units": "W_per_m2",
				"clamp_negative_flux_to_zero": True,
				"target_transform": "log1p",
				"inverse_transform": "expm1",
				"predict_total": True,
				"predict_sensible": False,
				"predict_latent": False,
				"add_as_input_history": False,
			},
		}

	def tearDown(self) -> None:
		self.tmpdir.cleanup()

	def _make_fire(self, name: str, with_terrain: bool = True) -> Path:
		fire_dir = self.root / name
		fire_dir.mkdir(parents=True, exist_ok=True)
		frame = np.zeros((2, 3, 86), dtype=np.float32)
		np.save(fire_dir / "tensor0000.npy", frame)
		_write_fake_geom(
			fire_dir / f"{name}.geom",
			nx=3,
			ny=2,
			nz=1,
			lons=[-120.0, -119.99, -119.98],
			lats=[35.0, 35.01],
		)
		if with_terrain:
			(fire_dir / f"{name}.terrain").write_text("3 2 1 911.0 1111.0\n", encoding="utf-8")
		return fire_dir

	def test_discover_fire_datasets_and_update_index(self) -> None:
		self._make_fire("FIRE_A", with_terrain=True)
		self._make_fire("FIRE_B", with_terrain=False)
		index = discover_fire_datasets(self.root, require_geom=True, require_terrain=False)
		self.assertEqual(index["num_fires"], 2)
		self.assertIn("FIRE_A", index["fires"])
		self.assertTrue(index["fires"]["FIRE_A"]["valid_for_energy_release"])
		output_json = self.root / "fire_dataset_index.json"
		save_fire_dataset_index(index, output_json)
		reloaded = load_fire_dataset_index(output_json)
		self.assertEqual(reloaded["num_fires"], 2)

		self._make_fire("FIRE_C", with_terrain=True)
		updated = update_fire_dataset_index(reloaded, discover_fire_datasets(self.root, require_geom=True, require_terrain=False))
		self.assertEqual(updated["num_fires"], 3)
		self.assertIn("FIRE_C", updated["fires"])

	def test_missing_geom_raises(self) -> None:
		fire_dir = self.root / "FIRE_NO_GEOM"
		fire_dir.mkdir(parents=True, exist_ok=True)
		np.save(fire_dir / "tensor0000.npy", np.zeros((2, 3, 86), dtype=np.float32))
		with self.assertRaises(FileNotFoundError):
			discover_fire_datasets(self.root, require_geom=True)

	def test_discovery_prefers_keepz_08_and_allows_transposed_geom_match(self) -> None:
		fire_root = self.root / "FIRE_KEEPZ"
		other_dir = fire_root / "other_sim"
		keepz_dir = fire_root / "keepz_08"
		other_dir.mkdir(parents=True, exist_ok=True)
		keepz_dir.mkdir(parents=True, exist_ok=True)

		np.save(other_dir / "tensor0000.npy", np.zeros((2, 3, 86), dtype=np.float32))
		_write_fake_geom(
			other_dir / "other.geom",
			nx=3,
			ny=2,
			nz=1,
			lons=[-120.0, -119.99, -119.98],
			lats=[35.0, 35.01],
		)

		# This tensor is stored as (nx, ny, channels) instead of the usual (ny, nx, channels).
		np.save(keepz_dir / "tensor0000.npy", np.zeros((3, 2, 86), dtype=np.float32))
		_write_fake_geom(
			keepz_dir / "keepz.geom",
			nx=3,
			ny=2,
			nz=1,
			lons=[-120.0, -119.99, -119.98],
			lats=[35.0, 35.01],
		)

		index = discover_fire_datasets(self.root, require_geom=True, require_terrain=False)
		record = index["fires"]["FIRE_KEEPZ"]
		self.assertEqual(Path(record["data_dir"]).name, "keepz_08")
		self.assertEqual(Path(record["geom_path"]).name, "keepz.geom")
		self.assertEqual(record["geom_tensor_orientation"], "transposed")
		self.assertTrue(record["geom_requires_transpose"])

	def test_parse_fake_geom(self) -> None:
		fire_dir = self._make_fire("FIRE_GEOM", with_terrain=False)
		info = parse_geom_file(fire_dir / "FIRE_GEOM.geom")
		self.assertEqual(info["nx"], 3)
		self.assertEqual(info["ny"], 2)
		self.assertEqual(info["nz"], 1)
		np.testing.assert_allclose(info["lons"], [-120.0, -119.99, -119.98])
		np.testing.assert_allclose(info["lats"], [35.0, 35.01])

	def test_cell_area_shape_and_row_constant(self) -> None:
		fire_dir = self._make_fire("FIRE_AREA", with_terrain=False)
		geom_info = parse_geom_file(fire_dir / "FIRE_AREA.geom")
		area_info = compute_cell_area_from_geom(geom_info, self.config)
		area_2d = area_info["area_2d_m2"]
		self.assertEqual(area_2d.shape, (2, 3))
		np.testing.assert_allclose(area_2d[0, :], area_2d[0, 0])
		np.testing.assert_allclose(area_2d[1, :], area_2d[1, 0])

	def test_constant_flux_energy_release(self) -> None:
		fire_dir = self._make_fire("FIRE_ENERGY", with_terrain=False)
		geom_info = parse_geom_file(fire_dir / "FIRE_ENERGY.geom")
		area_info = compute_cell_area_from_geom(geom_info, self.config)
		area_2d = np.asarray(area_info["area_2d_m2"], dtype=np.float32)
		frame = np.zeros((2, 3, 86), dtype=np.float32)
		frame[:, :, 80] = 100.0
		frame[:, :, 81] = 20.0
		frame[:, :, 82] = 50.0
		frame[:, :, 83] = 30.0
		energy_maps = compute_energy_release_maps(frame, self.config, area_2d_m2=area_2d)
		expected = area_2d * 200.0 / 1.0e6
		np.testing.assert_allclose(energy_maps["energy_release_total_MW"], expected)
		np.testing.assert_allclose(
			energy_maps["energy_release_sensible_MW"] + energy_maps["energy_release_latent_MW"],
			energy_maps["energy_release_total_MW"],
		)


if __name__ == "__main__":
	unittest.main()
