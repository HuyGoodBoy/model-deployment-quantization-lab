"""Check Android result integrity and consistent percentile definitions, no device needed."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lab.android_results import load_output, timing_statistics


class AndroidResultsTests(unittest.TestCase):
    def test_known_latency_and_reject_inconsistent_summary(self):
        benchmark = {'latency_samples_ms': [1, 2, 3, 4], 'mean_ms': 2.5,
                     'median_ms': 2.5, 'p95_ms': 3.85, 'images_per_second': 400}
        self.assertAlmostEqual(timing_statistics(benchmark, 4)['p95_ms'], 3.85)
        benchmark['median_ms'] = 4
        with self.assertRaisesRegex(ValueError, 'summary mismatch'):
            timing_statistics(benchmark, 4)

    def test_reject_missing_and_nonpositive_latency(self):
        with self.assertRaises(ValueError):
            timing_statistics({'latency_samples_ms': [1, 2]}, 3)
        with self.assertRaises(ValueError):
            timing_statistics({'latency_samples_ms': [1, 0]}, 2)

    def test_output_roundtrip_and_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            probability = np.zeros(1000, dtype='<f4')
            probability[652] = 1
            path = directory / 'sample.f32'
            probability.tofile(path)
            record = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            np.testing.assert_array_equal(load_output(directory, record), probability)
            path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_output(directory, record)

    def test_reject_path_traversal_and_logits(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                load_output(directory, {'file': '../outside.f32', 'sha256': ''})
            path = directory / 'logits.f32'
            np.ones(1000, dtype='<f4').tofile(path)
            record = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'softmax'):
                load_output(directory, record)


if __name__ == '__main__':
    unittest.main()
