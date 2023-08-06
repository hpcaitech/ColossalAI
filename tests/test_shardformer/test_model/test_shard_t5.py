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
from tests.test_shardformer.test_model._utils import build_model, check_grad, check_state_dict, run_forward


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    # check forward
    # the value "past_key_values" is sharded, so we ignore
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)
    assert_hf_output_close(org_output, shard_output, ignore_keys=['past_key_values'], atol=1e-5)

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # check grad
    col_layer_for_check = ['encoder.block[0].layer[0].SelfAttention.q', 'shared']
    row_layer_for_check = ['encoder.block[0].layer[0].SelfAttention.relative_attention_bias']
    check_grad(org_model, sharded_model, col_layer_for_check, atol=1e-6, rtol=1e-5, dim=0, verbose=False)
    check_grad(org_model, sharded_model, row_layer_for_check, atol=1e-6, rtol=1e-5, dim=1, verbose=False)

    # check weights are tied
    if hasattr(org_model, 'lm_head'):
        assert org_model.shared.weight.data.data_ptr() == org_model.lm_head.weight.data.data_ptr()
        assert sharded_model.shared.weight.data.data_ptr() == sharded_model.lm_head.weight.data.data_ptr()


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
@parameterize('use_lazy_init', [False, True])
@parameterize('enable_flash_attention', [True, False])
@parameterize('enable_jit_fused', [True, False])
def run_t5_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init, enable_flash_attention,
                enable_jit_fused):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_t5')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism,
                                               enable_flash_attention, enable_jit_fused, use_lazy_init)
        check_state_dict(org_model, sharded_model, name=name)
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)
    torch.cuda.empty_cache()


def check_t5(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_t5_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_t5():
    spawn(check_t5, 2)


if __name__ == "__main__":
    test_t5()
