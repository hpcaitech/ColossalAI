import pytest
import torch

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.tensor.d_tensor.api import is_customized_distributed_tensor, is_distributed_tensor
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
    # unwarp model
    if org_model.__class__.__name__ == 'BertModel':
        bert = org_model
        sharded_bert = sharded_model
    else:
        bert = org_model.bert
        sharded_bert = sharded_model.bert

    # check forward
    org_output, org_loss, shard_output, shard_loss = run_forward(org_model, sharded_model, data_gen_fn,
                                                                 output_transform_fn, loss_fn)
    assert_hf_output_close(org_output, shard_output)

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # check grad
    col_layer_for_check = ['encoder.layer[0].attention.self.query', 'embeddings.word_embeddings']
    row_layer_for_check = ['encoder.layer[0].attention.output.dense']
    check_grad(bert, sharded_bert, col_layer_for_check, atol=1e-7, rtol=1e-3, dim=0, verbose=False)
    check_grad(bert, sharded_bert, row_layer_for_check, atol=1e-7, rtol=1e-3, dim=1, verbose=False)


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
@parameterize('enable_flash_attention', [True, False])
@parameterize('enable_jit_fused', [True, False])
@parameterize('use_lazy_init', [False, True])
def run_bert_test(enable_fused_normalization, enable_tensor_parallelism, enable_flash_attention, enable_jit_fused,
                  use_lazy_init):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_bert')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism,
                                               enable_flash_attention, enable_jit_fused, use_lazy_init)
        check_state_dict(org_model, sharded_model, name=name)
        check_forward_backward(org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn)

    torch.cuda.empty_cache()


def check_bert(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_bert_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_bert():
    spawn(check_bert, 2)


if __name__ == "__main__":
    test_bert()
