import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.terrain import compute_terrain_features, parse_terrain_file, validate_terrain_features


class TerrainParsingTests(unittest.TestCase):
    def test_header_is_ignored_and_nx_ny_matches_frame_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.terrain"
            path.write_text("4 2 1 10.0 20.0\n1 2 3 4\n5 6 7 8\n", encoding="utf-8")
            height, metadata = parse_terrain_file(path, expected_shape=(4, 2))

        expected = np.asarray([[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.float32)
        np.testing.assert_array_equal(height, expected)
        self.assertEqual(height.shape, (4, 2))
        self.assertTrue(metadata["transposed"])
        self.assertEqual(metadata["layout"], "frame_spatial")
        self.assertEqual(metadata["reconstruction"], "ny,nx->nx,ny")
        self.assertEqual(metadata["x_axis"], 0)
        self.assertEqual(metadata["y_axis"], 1)
        self.assertEqual(metadata["reshape_order"], "C")
        self.assertEqual(metadata["header"]["nx"], 4)
        self.assertEqual(metadata["header"]["ny"], 2)

    def test_relative_elevation_uses_per_fire_p1_p99_and_clips_outliers(self):
        height = np.arange(100, dtype=np.float32).reshape(10, 10)
        height[0, 0] = -1000.0
        height[-1, -1] = 1000.0
        features, metadata = compute_terrain_features(height, dx=1.0, dy=1.0, x_axis=1, y_axis=0)

        p1, p99 = np.percentile(height, [1, 99])
        expected = np.clip((height - p1) / (p99 - p1 + 1e-6), 0.0, 1.0).astype(np.float32)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(features.shape, (4, 10, 10))
        np.testing.assert_allclose(features[0], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(float(features[0, 0, 0]), 0.0)
        self.assertEqual(float(features[0, -1, -1]), 1.0)
        self.assertGreaterEqual(float(features[1].min()), 0.0)
        self.assertLessEqual(float(features[1].max()), 1.0)
        self.assertGreaterEqual(float(features[2].min()), -1.0)
        self.assertLessEqual(float(features[2].max()), 1.0)
        self.assertGreaterEqual(float(features[3].min()), -1.0)
        self.assertLessEqual(float(features[3].max()), 1.0)
        validate_terrain_features(features, expected_shape=(10, 10))
        self.assertAlmostEqual(metadata["rel_elev_p1"], float(p1))
        self.assertAlmostEqual(metadata["rel_elev_p99"], float(p99))
        self.assertIn("final_channel_stats", metadata)
        self.assertIn("relative_elevation", metadata["final_channel_stats"])
        self.assertIn("slope_x_abs_p99", metadata)
        self.assertIn("slope_y_abs_p99", metadata)

    def test_features_are_four_channel_and_x_is_first_spatial_axis(self):
        height = np.tile(np.arange(4, dtype=np.float32).reshape(4, 1), (1, 5))
        features, metadata = compute_terrain_features(height, dx=1.0, dy=1.0, x_axis=0, y_axis=1)

        self.assertEqual(features.shape, (4, 4, 5))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[1] >= 0))
        self.assertTrue(np.all(features[2] >= -1) and np.all(features[2] <= 1))
        self.assertTrue(np.all(features[3] >= -1) and np.all(features[3] <= 1))
        self.assertGreater(float(features[2].mean()), 0.0)
        self.assertAlmostEqual(float(features[3].mean()), 0.0, places=5)
        self.assertEqual(metadata["x_axis"], 0)
        self.assertEqual(metadata["y_axis"], 1)
        self.assertEqual([item["name"] for item in metadata["feature_channels"]], ["relative_elevation", "slope_magnitude", "slope_x", "slope_y"])


if __name__ == "__main__":
    unittest.main()
