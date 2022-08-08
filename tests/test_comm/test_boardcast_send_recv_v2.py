from functools import partial
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.communication.p2p_v2 import _send_object, _recv_object, init_process_group
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.utils import free_port, get_current_device
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.logging import disable_existing_loggers

disable_existing_loggers()
world_size = 4
CONFIG = dict(parallel=dict(pipeline=world_size))
torch.manual_seed(123)


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl', verbose=False)
    rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    if rank == 0:
        obj = [torch.randn(3,)]
        _send_object(obj, 1)

    if rank == 1:
        _recv_object(0)

    if rank == 2:
        _recv_object(3)

    if rank == 3:
        obj = [torch.randn(3,)]
        _send_object(obj, 2)

    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_object_list_p2p():
    disable_existing_loggers()
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_object_list_p2p()
