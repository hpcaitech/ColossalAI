import gc

from colossalai.accelerator import get_accelerator


def pytest_runtest_setup(item):
    # called for running each test in 'a' directory
    accelerator = get_accelerator()
    # CpuAccelerator.empty_cache() is intentionally unsupported.  Keeping this
    # hook CUDA-only lets genuinely CPU-only CI jobs run without exposing a GPU.
    if accelerator.name != "cpu":
        accelerator.empty_cache()
    gc.collect()
