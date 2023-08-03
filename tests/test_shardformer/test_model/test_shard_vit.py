import os

import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import (
    assert_hf_output_close,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, check_grad, run_forward


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # check forward
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)
    assert_hf_output_close(org_output, shard_output, atol=1e-3, rtol=1e-3)
    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # unwrap model
    if org_model.__class__.__name__ == 'ViTModel':
        vit_model = org_model
        shard_vit_model = sharded_model
    else:
        vit_model = org_model.vit
        shard_vit_model = sharded_model.vit

    # check grad
    col_layer_for_check = ['encoder.layer[0].attention.attention.query']
    row_layer_for_check = ['encoder.layer[0].attention.output.dense']
    check_grad(vit_model, shard_vit_model, col_layer_for_check, atol=1e-5, rtol=1e-3, dim=0, verbose=False)
    check_grad(vit_model, shard_vit_model, row_layer_for_check, atol=1e-5, rtol=1e-3, dim=1, verbose=False)


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
def run_vit_test(enable_fused_normalization, enable_tensor_parallelism):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_vit')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism)
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)
    torch.cuda.empty_cache()


def check_vit(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_vit_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_vit():
    spawn(check_vit, 2)


if __name__ == "__main__":
    test_vit()
