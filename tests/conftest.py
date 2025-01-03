import gc

from colossalai.accelerator import get_accelerator


def pytest_runtest_setup(item):
    # called for running each test in 'a' directory
    accelerator = get_accelerator()
    accelerator.empty_cache()
    gc.collect()
