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
    assert_hf_output_close(org_output, shard_output)

    # do backward
    org_loss.backward()
    shard_loss.backward()

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

    # check grad

    if org_model.__class__.__name__ == 'BertModel':
        bert = org_model
        sharded_bert = sharded_model
    else:
        bert = org_model.bert
        sharded_bert = sharded_model.bert

    # compare self attention grad
    org_grad = bert.encoder.layer[0].attention.self.query.weight.grad
    shard_grad = sharded_bert.encoder.layer[0].attention.self.query.weight.grad
    shard_weight = sharded_bert.encoder.layer[0].attention.self.query.weight

    if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
        shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(2)]
        shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
        all_shard_grad = torch.cat(shard_grad_list, dim=0)
    else:
        all_shard_grad = shard_grad
    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{all_shard_grad}"

    # compare embedding grad
    org_grad = bert.embeddings.word_embeddings.weight.grad
    shard_grad = sharded_bert.embeddings.word_embeddings.weight.grad
    shard_weight = sharded_bert.embeddings.word_embeddings.weight

    if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
        shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(2)]
        shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
        all_shard_grad = torch.cat(shard_grad_list, dim=0)
    else:
        all_shard_grad = shard_grad

    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{all_shard_grad}"


@parameterize('enable_fused_normalization', [True, False])
@parameterize('enable_tensor_parallelism', [True, False])
def run_bert_test(enable_fused_normalization, enable_tensor_parallelism):
    sub_model_zoo = model_zoo.get_sub_registry('transformers_bert')
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        org_model, sharded_model = build_model(model_fn, enable_fused_normalization, enable_tensor_parallelism)
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
