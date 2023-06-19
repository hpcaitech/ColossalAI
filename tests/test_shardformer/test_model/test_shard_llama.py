import copy
import os
import random

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaTokenizerFast

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


def build_model(world_size, model_fn):
    # create new model
    config = LlamaConfig(num_hidden_layers=8)
    org_model = model_fn(config).cuda()

    # shard model
    shard_config = ShardConfig(tensor_parallel_size=world_size)
    model_copy = copy.deepcopy(org_model)
    shard_former = ShardFormer(shard_config=shard_config)
    shard_former.init_distributed()
    sharded_model = shard_former.shard_model(model_copy)

    return org_model, sharded_model


def check_forward(org_model, sharded_model):
    input = 'Hello, my dog is cute'
    inputs = tokenizer(input, return_tensors='pt').to('cuda')
    del inputs["token_type_ids"]
    del inputs["attention_mask"]

    #orgin model
    org_model.eval()
    org_out = org_model(**inputs)

    #shard model
    sharded_model.eval()
    shard_out = sharded_model(**inputs)

    assert torch.allclose(
        org_out[0], shard_out[0],
        atol=1e-4), f"shard model output is not equal to orgin model output\n{org_out[0]}\n{shard_out[0]}"


def check_backward(org_model, sharded_model):
    # prepare input
    input = 'Hello, my dog is cute'
    tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
    del tokenized_input["token_type_ids"]
    del tokenized_input["attention_mask"]
    labels = tokenized_input['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized_input['labels'] = labels

    #orgin model
    org_model.train()
    org_out = org_model(**tokenized_input)
    org_loss = org_out.loss
    org_loss.backward()
    org_grad = org_model.model.layers[0].self_attn.q_proj.weight.grad

    torch.cuda.empty_cache()
    #shard model
    sharded_model.train()
    shard_out = sharded_model(**tokenized_input)
    shard_loss = shard_out.loss
    shard_loss.backward()
    shard_grad = sharded_model.model.layers[0].self_attn.q_proj.weight.grad
    shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(4)]
    shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
    all_shard_grad = torch.cat(shard_grad_list, dim=0)

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"
    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{shard_grad}"


def check_llama(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    model_list = [
        LlamaForCausalLM,

    # TODO: do not work yet
    # LlamaModel,
    # LlamaForSequenceClassification
    ]

    for model_fn in model_list:
        org_model, sharded_model = build_model(world_size, model_fn)
        check_forward(org_model, sharded_model)
        check_backward(org_model, sharded_model)

    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_llama():
    spawn(check_llama, 4)


if __name__ == "__main__":
    test_llama()
