import unittest

from check_idle_gpus import readiness


class ReadinessTests(unittest.TestCase):
    def test_no_layer_runs_without_complete_layer2_capacity(self):
        self.assertEqual(readiness(1), {"layer2_ready": False, "layer3_ready": False})

    def test_layer2_runs_from_two_idle_gpus(self):
        self.assertEqual(readiness(2), {"layer2_ready": True, "layer3_ready": False})

    def test_both_layers_run_from_eight_idle_gpus(self):
        self.assertEqual(readiness(8), {"layer2_ready": True, "layer3_ready": True})

    def test_invalid_thresholds_fail_closed(self):
        for values in ((-1, 2, 8), (0, 0, 8), (0, 4, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                readiness(*values)


if __name__ == "__main__":
    unittest.main()
