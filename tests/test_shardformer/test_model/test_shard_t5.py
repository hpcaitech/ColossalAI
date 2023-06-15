import copy
import os
import random

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, T5Config, T5ForConditionalGeneration, T5Tokenizer

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.shard import ShardConfig, shard_model
from colossalai.testing import rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def build_model(rank, world_size):
    config = T5Config.from_pretrained("t5-small")
    config.dropout_rate = 0
    org_model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config).to('cuda')

    shardconfig = ShardConfig(
        rank=rank,
        world_size=world_size,
        gather_output=True,
    )

    org_model_for_shard = copy.deepcopy(org_model)

    sharded_model = shard_model(org_model_for_shard, shardconfig).to('cuda')

    return org_model, sharded_model


def check_forward(org_model, sharded_model):

    input_ids = tokenizer("translate English to German: The house is wonderful.",
                          return_tensors="pt").input_ids.to('cuda')
    #orgin model
    org_model.eval()
    org_output = org_model.generate(input_ids)

    #shard model
    sharded_model.eval()
    shard_output = sharded_model.generate(input_ids)
    assert torch.allclose(
        org_output[0], shard_output[0],
        atol=1e-5), f"shard model output is not equal to orgin model output\n{org_out[0]}\n{shard_out[0]}"


def check_backward(org_model, sharded_model):
    # prepare input
    input_ids = tokenizer("translate English to German: The house is wonderful.",
                          return_tensors="pt").input_ids.to('cuda')
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids.to('cuda')

    #orgin model
    org_model.train()
    org_loss = org_model(input_ids=input_ids, labels=labels).loss
    org_loss.backward()
    org_grad = org_model.encoder.block[0].layer[0].SelfAttention.q.weight.grad

    #shard model
    sharded_model.train()
    shard_loss = sharded_model(input_ids=input_ids, labels=labels).loss
    shard_loss.backward()
    shard_grad = sharded_model.encoder.block[0].layer[0].SelfAttention.q.weight.grad

    shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(2)]
    shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
    all_shard_grad = torch.cat(shard_grad_list, dim=0)

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"
    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{shard_grad}"


def check_t5(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    org_model, sharded_model = build_model(rank, world_size)
    check_forward(org_model, sharded_model)
    check_backward(org_model, sharded_model)

    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_t5():
    spawn(check_t5, 2)


if __name__ == "__main__":
    test_t5()
