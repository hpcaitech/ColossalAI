#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pytest
import torch
from checks_3d.check_layer_3d import (
    check_classifier_no_given_weight,
    check_embed,
    check_layernorm,
    check_linear,
    check_loss,
    check_patch_embed,
    check_vocab_parallel_classifier_given_embed_weight,
    check_vocab_parallel_classifier_no_given_weight,
    check_vocab_parallel_embed,
    check_vocab_parallel_loss,
)

from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, skip_if_not_enough_gpus, spawn

CONFIG = dict(
    parallel=dict(
        pipeline=1,
        tensor=dict(mode="3d", size=8),
    ),
    seed=42,
)


def check_layer():
    check_linear()
    check_layernorm()
    check_classifier_no_given_weight()
    check_vocab_parallel_classifier_no_given_weight()
    check_vocab_parallel_classifier_given_embed_weight()
    check_embed()
    check_patch_embed()
    check_vocab_parallel_embed()
    check_loss()
    check_vocab_parallel_loss()


def check_layer_and_operation(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    check_layer()
    gpc.destroy()
    torch.cuda.empty_cache()


@pytest.mark.dist
@skip_if_not_enough_gpus(min_gpus=8)
@rerun_if_address_is_in_use()
def test_3d():
    spawn(check_layer_and_operation, 8)


if __name__ == "__main__":
    test_3d()
