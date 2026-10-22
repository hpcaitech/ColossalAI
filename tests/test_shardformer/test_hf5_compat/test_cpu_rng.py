import torch

from colossalai.accelerator.cpu_accelerator import CpuAccelerator


def test_cpu_rng_device_and_restore():
    accelerator = CpuAccelerator()
    initial = torch.get_rng_state()
    try:
        state = accelerator.get_rng_state()
        torch.testing.assert_close(state, accelerator.get_rng_state(torch.device("cpu")), atol=0, rtol=0)
        expected = torch.rand(8)
        accelerator.set_rng_state(state)
        torch.testing.assert_close(expected, torch.rand(8), atol=0, rtol=0)
    finally:
        torch.set_rng_state(initial)
