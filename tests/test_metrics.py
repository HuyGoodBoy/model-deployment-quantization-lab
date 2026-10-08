"""Kiem tra cong thuc bang vi du co dap an giai tich."""
import math
import unittest

import numpy as np

from lab.common import json_safe, numerical_metrics


class NumericalMetricsTests(unittest.TestCase):
    def test_known_signal_noise(self):
        result = numerical_metrics(np.array([1.0, 2.0]), np.array([1.1, 1.8]))
        self.assertAlmostEqual(result["mae"], 0.15)
        self.assertAlmostEqual(result["rmse"], math.sqrt(0.025))
        self.assertAlmostEqual(result["max_abs_error"], 0.2)
        self.assertAlmostEqual(result["relative_l2"], 0.1)
        self.assertAlmostEqual(result["snr_db"], 20.0)
        self.assertAlmostEqual(result["cosine_similarity"], 4.7 / math.sqrt(5 * 4.45))

    def test_identical_nonzero_signal(self):
        a = np.array([0.1, 0.9], dtype=np.float32)
        result = numerical_metrics(a, a.copy())
        self.assertEqual(result["mae"], 0)
        self.assertEqual(result["snr_db"], math.inf)
        self.assertAlmostEqual(result["cosine_similarity"], 1)

    def test_cosine_alone_does_not_measure_magnitude_error(self):
        a = np.array([1.0, 2.0])
        result = numerical_metrics(a, 2 * a)
        self.assertAlmostEqual(result["cosine_similarity"], 1)
        self.assertAlmostEqual(result["relative_l2"], 1)
        self.assertAlmostEqual(result["snr_db"], 0)
        self.assertGreater(result["rmse"], 0)

    def test_zero_reference_edge_cases(self):
        result = numerical_metrics(np.zeros(2), np.zeros(2))
        self.assertIsNone(result["snr_db"])
        self.assertIsNone(result["cosine_similarity"])
        self.assertEqual(result["relative_l2"], 0)
        result = numerical_metrics(np.zeros(2), np.ones(2))
        self.assertEqual(result["snr_db"], -math.inf)
        self.assertEqual(result["relative_l2"], math.inf)

    def test_reject_shape_mismatch_empty_and_nonfinite(self):
        for a, b in [(np.ones(2), np.ones(3)), (np.array([]), np.array([])),
                     (np.array([np.nan]), np.array([1.0])),
                     (np.array([1.0]), np.array([np.inf]))]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                numerical_metrics(a, b)

    def test_json_safe_infinity(self):
        self.assertEqual(json_safe({"snr": math.inf}), {"snr": "inf"})
        self.assertEqual(json_safe({"snr": -math.inf}), {"snr": "-inf"})
        self.assertEqual(json_safe({"snr": math.nan}), {"snr": None})


if __name__ == "__main__":
    unittest.main()
