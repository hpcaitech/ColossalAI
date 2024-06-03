import pytest

from colossalai.device import AlphaBetaProfiler
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


def check_extract_alpha_beta(rank, world_size, port, physical_devices):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    profiler = AlphaBetaProfiler(physical_devices)

    mesh_alpha, mesh_beta = profiler.extract_alpha_beta_for_device_mesh()
    for alpha in mesh_alpha:
        assert alpha > 0 and alpha < 1e-3
    for beta in mesh_beta:
        assert beta > 0 and beta < 1e-10


@pytest.mark.skip(reason="Skip because assertion may fail for CI devices")
@pytest.mark.dist
@parameterize("physical_devices", [[0, 1, 2, 3], [0, 3]])
@rerun_if_address_is_in_use()
def test_profile_alpha_beta(physical_devices):
    spawn(check_extract_alpha_beta, 4, physical_devices=physical_devices)


if __name__ == "__main__":
    test_profile_alpha_beta()
