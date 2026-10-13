"""Numerical checks for physical-band integration and history selection."""
import unittest

import numpy as np

from extract_lining_band_histories import (
    band_loads, overlap_weights, peak_record, wall_band_means,
)


class FixedBandReadoutTests(unittest.TestCase):
    def test_nonuniform_native_widths_preserve_integral(self):
        weights = overlap_weights([[3.0, 3.08], [3.08, 3.2]], [[3.0, 3.2]])
        np.testing.assert_allclose(weights @ [10.0, 20.0], [16.0])

    def test_wall_mean_integrates_internal_nodes(self):
        means = wall_band_means([3.0, 3.1, 3.2], [0.0, 2.0, 0.0], [[3.0, 3.2]])
        np.testing.assert_allclose(means, [1.0])

    def test_cross_batch_band_waits_for_last_installation(self):
        weights = overlap_weights([[3.0, 3.1], [3.1, 3.2]], [[3.0, 3.2]])
        steps = np.array([2, 3])
        self.assertTrue(np.isnan(band_loads([10.0, 0.0], weights, steps, 2)[0]))
        # At the second installation the new half is stress-free; the old half is loaded.
        np.testing.assert_allclose(band_loads([12.0, 0.0], weights, steps, 3), [6.0])

    def test_envelope_keeps_earlier_peak_and_its_location(self):
        result = peak_record(np.array([[np.nan, np.nan], [4.0, 7.0], [6.0, 5.0]]),
                             np.array([0.0, 0.4, 0.8]),
                             np.array([[3.0, 3.2], [3.2, 3.4]]), "pressure_MPa")
        self.assertEqual(result["pressure_MPa"], 7.0)
        self.assertEqual(result["face_over_a"], 0.4)
        self.assertEqual(result["band_over_a"], [3.2, 3.4])

    def test_nonfinite_source_is_not_an_uninstalled_band(self):
        with self.assertRaises(ValueError):
            band_loads([np.nan], np.array([[1.0]]), np.array([1]), 2)
        with self.assertRaises(ValueError):
            wall_band_means([3.0, 3.2], [0.0, np.nan], [[3.0, 3.2]])


if __name__ == "__main__":
    unittest.main()
