import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.context import MOE_CONTEXT
from colossalai.tensor import ColoParameter
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.zero import ColoInitContext
from tests.test_moe.test_moe_zero_init import MoeModel
from tests.test_tensor.common_utils import debug_print
from tests.test_zero.test_legacy.common import CONFIG


@parameterize("init_device_type", ['cpu', 'cuda'])
def exam_moe_colo_init(init_device_type):
    world_size = dist.get_world_size()

    if init_device_type == 'cuda':
        init_device = get_current_device()
    elif init_device_type == 'cpu':
        init_device = torch.device("cpu")
    else:
        raise NotImplementedError("Unknown device found.")

    with ColoInitContext(device=init_device):
        model = MoeModel(checkpoint=True)

    for name, param in model.named_parameters():
        assert isinstance(param, ColoParameter), "parameter `{}` has an init problem".format(name)

        if hasattr(param, "moe_info"):
            param.set_process_group(param.moe_info.pg)

        if hasattr(param, "moe_info"):
            assert param.process_group.dp_world_size() == param.moe_info.dp_size
        else:
            assert param.process_group.dp_world_size() == world_size


def _run_dist(rank, world_size, port):
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(seed=42)
    exam_moe_colo_init()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_colo_init(world_size):
    spawn(_run_dist, world_size)


if __name__ == '__main__':
    test_moe_colo_init(world_size=4)
