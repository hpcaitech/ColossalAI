import unittest

from run_gpu_batch import find_idle, parse_pool, select_idle

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
GPU_C = "GPU-00000000-0000-0000-0000-000000000003"


class GpuSelectionTests(unittest.TestCase):
    def test_selects_requested_idle_devices_in_index_order(self):
        rows = f"2,{GPU_C},4,0\n0,{GPU_A},4,0\n1,{GPU_B},4,0"
        self.assertEqual(select_idle(rows, "", 2), [(0, GPU_A), (1, GPU_B)])

    def test_lists_all_idle_devices(self):
        rows = f"2,{GPU_C},4,0\n0,{GPU_A},4,0\n1,{GPU_B},257,0"
        self.assertEqual(find_idle(rows, ""), [(0, GPU_A), (2, GPU_C)])

    def test_respects_authorized_pool(self):
        rows = f"0,{GPU_A},4,0\n1,{GPU_B},4,0\n2,{GPU_C},4,0"
        self.assertEqual(select_idle(rows, "", 1, {2}), [(2, GPU_C)])

    def test_memory_utilization_and_processes_all_disqualify(self):
        rows = f"0,{GPU_A},257,0\n1,{GPU_B},4,1\n2,{GPU_C},4,0"
        with self.assertRaises(RuntimeError):
            select_idle(rows, GPU_C, 1)

    def test_unknown_occupancy_fails_closed(self):
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},N/A,0", "", 1)
        with self.assertRaises(RuntimeError):
            select_idle(f"0,{GPU_A},4,0", "N/A", 1)

    def test_pool_validation(self):
        self.assertEqual(parse_pool("0,2"), {0, 2})
        for value in ("0,0", "0,a", "0, 1"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                parse_pool(value)


if __name__ == "__main__":
    unittest.main()
