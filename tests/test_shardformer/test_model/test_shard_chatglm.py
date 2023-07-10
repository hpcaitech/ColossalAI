import os

import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.api import is_customized_distributed_tensor, is_distributed_tensor
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, run_forward

def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # check forward
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)
    assert_hf_output_close(org_output, shard_output, atol=1e-4, rtol=1e-4)
    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # unwrap model
    if org_model.__class__.__name__ == 'ChatGLMModel':
        vit_model = org_model
        shard_vit_model = sharded_model
    else:
        chatglm_model = org_model.transformer
        shard_chatglm_model = sharded_model.transformer

    # check attention grad
    org_grad = vit_model.encoder.layer[0].attention.attention.query.weight.grad
    shard_grad = shard_vit_model.encoder.layer[0].attention.attention.query.weight.grad
    shard_weight = shard_vit_model.encoder.layer[0].attention.attention.query.weight

    if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
        shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(2)]
        shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
        all_shard_grad = torch.cat(shard_grad_list, dim=0)
    else:
        all_shard_grad = shard_grad
    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{shard_grad}"