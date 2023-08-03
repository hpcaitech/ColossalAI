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
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)
    assert_hf_output_close(org_output, shard_output, ignore_keys=['past_key_values'])

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-6), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # unwrap model
    if org_model.__class__.__name__ == 'BloomModel':
        bloom = org_model
        sharded_bloom = sharded_model
    else:
        bloom = org_model.transformer
        sharded_bloom = sharded_model.transformer

    # check grad
    col_layer_for_check = ['h[0].self_attention.query_key_value']
    row_layer_for_check = ['h[0].self_attention.dense']
    check_grad(bloom, sharded_bloom, col_layer_for_check, atol=1e-6, rtol=1e-5, dim=0, verbose=False)
    check_grad(bloom, sharded_bloom, row_layer_for_check, atol=1e-6, rtol=1e-5, dim=1, verbose=False)


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
@parameterize('use_lazy_init', [False, True])
def run_bloom_test(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_bloom')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism,
                                               use_lazy_init)
        check_state_dict(org_model, sharded_model, name=name)
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)
    torch.cuda.empty_cache()


def check_bloom(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_bloom_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bloom():
    spawn(check_bloom, 2)


if __name__ == "__main__":
    test_bloom()
