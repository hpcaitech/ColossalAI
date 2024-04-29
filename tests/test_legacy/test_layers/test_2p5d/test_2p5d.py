import pytest
import torch
from checks_2p5d.check_layer_2p5d import *
from checks_2p5d.check_operation_2p5d import check_AB, check_ABT, check_ATB

from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=4, mode="2.5d", depth=1),
    ),
)


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
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

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
    spawn(check_layer_and_operation, 4)


if __name__ == "__main__":
    test_2p5d()
