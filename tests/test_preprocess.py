"""Bat loi RGB/BGR, scaling va batch shape truoc khi chay model."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lab.common import validate_input
from lab.prepare import preprocess


class PreprocessingTests(unittest.TestCase):
    def test_red_image_keeps_rgb_and_maps_to_expected_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "red.png"
            Image.new("RGB", (8, 16), color=(255, 0, 0)).save(path)
            x = preprocess(path)
        validate_input(x)
        self.assertEqual(x.shape, (1, 224, 224, 3))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.flags.c_contiguous)
        np.testing.assert_array_equal(x[0, 0, 0], [1, -1, -1])

    def test_grayscale_expands_to_three_equal_channels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gray.png"
            Image.new("L", (5, 5), color=128).save(path)
            x = preprocess(path)
        validate_input(x)
        np.testing.assert_allclose(x[0, 0, 0], [128 / 127.5 - 1] * 3, atol=1e-7)

    def test_wrong_dtype_shape_range_or_nonfinite_is_rejected(self):
        examples = [np.zeros((224, 224, 3), dtype=np.float32),
                    np.zeros((1, 224, 224, 3), dtype=np.float64),
                    np.full((1, 224, 224, 3), 255, dtype=np.float32),
                    np.full((1, 224, 224, 3), np.nan, dtype=np.float32)]
        for x in examples:
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                validate_input(x)


if __name__ == "__main__":
    unittest.main()
