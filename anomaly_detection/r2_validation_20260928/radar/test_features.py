"""Numerical invariants for frame-local extraction, independent of fault labels."""
import unittest
import numpy as np
from extract_r2_features import features_for_window, FREQUENCIES, WEIGHTS


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.arange(50)[:, None] * .04 + np.arange(192)[None, :] / 5000
        cls.z = (12 * np.exp(1j * .12 * np.sin(2 * np.pi * 250 * t)))[:, :, None, None]
        cls.z = np.tile(cls.z, (1, 1, 4, 5))
        cls.valid = np.ones((50, 192), bool)
        cls.ranges = np.linspace(.36, .44, 5)

    def compute(self, z, valid=None):
        return features_for_window(z, self.valid if valid is None else valid, self.ranges, 40)

    def test_pool_is_mean_preserving(self):
        np.testing.assert_allclose(WEIGHTS @ np.ones(len(FREQUENCIES)), 1)

    def test_shape_and_known_tone(self):
        features = self.compute(self.z)
        self.assertEqual(features[0].shape, (128,))
        self.assertEqual(features[1].shape, (128,))
        self.assertEqual(features[2].shape, (12,))
        peak = FREQUENCIES[np.argmax(features[4])]
        self.assertLess(abs(peak - 250), 5)

    def test_conjugation_does_not_change_power_shape(self):
        a, b = self.compute(self.z), self.compute(self.z.conj())
        np.testing.assert_allclose(a[0], b[0], atol=1e-6)
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)

    def test_global_amplitude_scaling_not_main_input(self):
        a, b = self.compute(self.z), self.compute(self.z * 10)
        np.testing.assert_allclose(a[0], b[0], atol=2e-5)
        np.testing.assert_allclose(a[1], b[1], atol=2e-5)
        self.assertAlmostEqual(float(b[2][0]-a[2][0]), 2, places=5)

    def test_no_cross_frame_phase_dependency(self):
        rotation = np.exp(1j * np.linspace(-3, 3, 50))[:, None, None, None]
        a, b = self.compute(self.z), self.compute(self.z * rotation)
        np.testing.assert_allclose(a[0], b[0], atol=2e-5)
        np.testing.assert_allclose(a[1], b[1], atol=2e-5)

    def test_invalid_frame_is_flagged_not_joined(self):
        valid = self.valid.copy()
        valid[0, 80] = False
        bad = self.z.copy()
        bad[0] *= 10000
        a, b = self.compute(self.z, valid), self.compute(bad, valid)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[3]["complete_frame_fraction"], .98)

    def test_unobservable_window_explicit_finite_fallback(self):
        result = self.compute(self.z, np.zeros_like(self.valid))
        self.assertFalse(result[3]["spectral_estimate_available"])
        self.assertTrue(np.isfinite(result[0]).all())
        np.testing.assert_array_equal(result[0], np.zeros(128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
