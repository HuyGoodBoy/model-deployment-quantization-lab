"""Prevent misleading GPU claims from copied tensors or repeated profiling runs."""
import unittest

from qlab.gpu_runtime import summarize_profile


def event(name, op, provider):
    return {'cat': 'Node', 'name': name+'_kernel_time',
            'args': {'op_name': op, 'provider': provider}}


class ProviderEvidenceTests(unittest.TestCase):
    def test_memcpy_is_not_proof_of_gpu_compute(self):
        result = summarize_profile([event('upload', 'MemcpyFromHost', 'CUDAExecutionProvider'),
                                    event('matmul', 'MatMulInteger', 'CPUExecutionProvider')])
        self.assertEqual(result['cuda_compute_nodes'], 0)
        self.assertEqual(result['unique_nodes_by_provider']['CPUExecutionProvider'], 1)

    def test_mixed_graph_counts_unique_nodes_not_inference_runs(self):
        events = [event('conv', 'Conv', 'CUDAExecutionProvider'),
                  event('classifier', 'MatMulInteger', 'CPUExecutionProvider')]
        result = summarize_profile(events*2)
        self.assertEqual(result['unique_nodes_by_provider'],
                         {'CUDAExecutionProvider': 1, 'CPUExecutionProvider': 1})
        self.assertEqual(result['kernel_events_by_provider']['CUDAExecutionProvider'], 2)
        self.assertEqual(result['cuda_compute_operators'], ['Conv'])
        self.assertEqual(result['operator_counts_by_provider']['CPUExecutionProvider'],
                         {'MatMulInteger': 1})


if __name__ == '__main__':
    unittest.main()
