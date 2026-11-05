import unittest
from unittest.mock import Mock, call

from cuda_probe import probe_devices


class CUDAProbeTests(unittest.TestCase):
    def setUp(self):
        self.torch = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.device_count.return_value = 2
        self.torch.cuda.get_device_name.side_effect = lambda index: f"GPU {index}"
        self.torch.ones.return_value.sum.return_value.item.return_value = 4

    def test_computes_and_synchronizes_every_allocated_device(self):
        self.assertEqual(probe_devices(self.torch, 2), ["GPU 0", "GPU 1"])
        self.assertEqual(
            self.torch.ones.call_args_list,
            [
                call(4, dtype=self.torch.float32, device="cuda:0"),
                call(4, dtype=self.torch.float32, device="cuda:1"),
            ],
        )
        self.assertEqual(self.torch.cuda.synchronize.call_args_list, [call(0), call(1)])

    def test_enumeration_success_cannot_hide_allocation_failure(self):
        self.torch.ones.side_effect = [self.torch.ones.return_value, RuntimeError("device access denied")]
        with self.assertRaisesRegex(RuntimeError, "cuda:1: device access denied"):
            probe_devices(self.torch, 2)
        self.torch.cuda.synchronize.assert_called_once_with(0)

    def test_asynchronous_cuda_failure_is_not_a_pass(self):
        self.torch.cuda.synchronize.side_effect = RuntimeError("CUDA device unavailable")
        with self.assertRaisesRegex(RuntimeError, "cuda:0: CUDA device unavailable"):
            probe_devices(self.torch, 2)

    def test_unavailable_or_wrong_device_count_fails_before_computation(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            probe_devices(self.torch, 2)
        self.torch.cuda.is_available.return_value = True
        with self.assertRaisesRegex(RuntimeError, "Expected 1 CUDA device"):
            probe_devices(self.torch, 1)
        with self.assertRaisesRegex(RuntimeError, "at least one"):
            probe_devices(self.torch, 0)
        self.torch.ones.assert_not_called()


if __name__ == "__main__":
    unittest.main()
