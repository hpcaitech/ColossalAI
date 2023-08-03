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

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn):
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)

    # forward check
    assert_hf_output_close(org_output, shard_output, ignore_keys=['past_key_values'], rtol=1e-5)

    # run backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # unwrap model
    if hasattr(org_model, 'model'):
        llama_model = org_model.model
        shard_llama_model = sharded_model.model
    else:
        llama_model = org_model
        shard_llama_model = sharded_model

    # check grad
    col_layer_for_check = ['layers[0].self_attn.q_proj', 'embed_tokens']
    row_layer_for_check = ['layers[0].self_attn.o_proj']
    check_grad(llama_model, shard_llama_model, col_layer_for_check, atol=1e-6, rtol=1e-4, dim=0, verbose=False)
    check_grad(llama_model, shard_llama_model, row_layer_for_check, atol=1e-6, rtol=1e-4, dim=1, verbose=False)


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
@parameterize('use_lazy_init', [False, True])
def run_gpt2_llama(enable_fused_normalization, enable_tensor_parallelism, use_lazy_init):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism,
                                               use_lazy_init)
        check_state_dict(org_model, sharded_model, name=name)
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)
    torch.cuda.empty_cache()


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt2_llama()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 4)


if __name__ == "__main__":
    test_llama()
