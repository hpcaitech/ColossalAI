import pytest

from colossalai.device import AlphaBetaProfiler
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


def check_alpha_beta(rank, world_size, port, physical_devices):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    profiler = AlphaBetaProfiler(physical_devices)
    best_logical_mesh = profiler.search_best_logical_mesh()

    if physical_devices == [0, 1, 2, 3]:
        assert best_logical_mesh == [[0, 1], [2, 3]]
    elif physical_devices == [0, 3]:
        assert best_logical_mesh == [[0, 3]]


@pytest.mark.skip(reason="Skip because assertion may fail for CI devices")
@pytest.mark.dist
@parameterize("physical_devices", [[0, 1, 2, 3], [0, 3]])
@rerun_if_address_is_in_use()
def test_profile_alpha_beta(physical_devices):
    spawn(check_alpha_beta, 4, physical_devices=physical_devices)


if __name__ == "__main__":
    test_profile_alpha_beta()
