from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from checks_2p5d.check_layer_2p5d import *
from checks_2p5d.check_operation_2p5d import check_AB, check_ABT, check_ATB

CONFIG = dict(parallel=dict(
    pipeline=dict(size=1),
    tensor=dict(size=4, mode='2.5d', depth=1),
),)


def check_operations():
    check_AB()
    check_ABT()
    check_ATB()


def check_layer():
    check_linear()
    check_layernorm()
    check_embed()
    check_patch_embed()
    check_vocab_parallel_embed()
    check_classifier_no_given_weight()
    check_vocab_parallel_classifier_no_given_weight()
    check_classifier_given_embed_weight()
    check_vocab_parallel_classifier_given_embed_weight()
    check_loss()
    check_vocab_parallel_loss()


def check_layer_and_operation(rank, world_size, port):
    disable_existing_loggers()
    launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    check_operations()
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_2p5d():
    world_size = 4
    run_func = partial(check_layer_and_operation, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_2p5d()
