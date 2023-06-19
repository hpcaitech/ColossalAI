import copy
import os
import random

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaTokenizerFast

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import assert_hf_output_close, clear_cache_before_run, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


def build_model(world_size, model_fn):
    # create new model
    config = LlamaConfig(num_hidden_layers=4,
                         hidden_size=128,
                         intermediate_size=256,
                         num_attention_heads=4,
                         max_position_embeddings=128)
    org_model = model_fn(config).cuda()

    # shard model
    shard_config = ShardConfig(tensor_parallel_size=world_size)
    model_copy = copy.deepcopy(org_model)
    shard_former = ShardFormer(shard_config=shard_config)
    shard_former.init_distributed()
    sharded_model = shard_former.shard_model(model_copy)

    return org_model, sharded_model


def check_forward_backward(org_model, sharded_model):
    # prepare input
    input = 'Hello, my dog is cute'
    tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
    del tokenized_input["token_type_ids"]
    del tokenized_input["attention_mask"]

    # switch to train mode
    org_model.train()
    sharded_model.train()

    if isinstance(org_model, (LlamaModel, LlamaForSequenceClassification)):
        org_output = org_model(**tokenized_input)
        org_loss = org_output.last_hidden_state.mean()
        shard_output = sharded_model(**tokenized_input)
        shard_loss = shard_output.last_hidden_state.mean()
    elif isinstance(org_model, LlamaForCausalLM):
        labels = tokenized_input['input_ids'].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_input['labels'] = labels
        org_output = org_model(**tokenized_input)
        org_loss = org_output.loss
        shard_output = sharded_model(**tokenized_input)
        shard_loss = shard_output.loss

    assert_hf_output_close(org_output, shard_output, ignore_keys=['past_key_values'], rtol=1e-4)

    # run backward
    org_loss.backward()
    shard_loss.backward()

    # check grad
    if isinstance(org_model, LlamaModel):
        llama_model = org_model
        shard_llama_model = sharded_model
    else:
        llama_model = org_model.model
        shard_llama_model = sharded_model.model

    org_grad = llama_model.layers[0].self_attn.q_proj.weight.grad
    shard_grad = shard_llama_model.layers[0].self_attn.q_proj.weight.grad
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
        LlamaModel,
    # LlamaForCausalLM,

    # TODO: do not work yet
    # LlamaForSequenceClassification
    ]

    for model_fn in model_list:
        org_model, sharded_model = build_model(world_size, model_fn)
        check_forward_backward(org_model, sharded_model)

    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama():
    spawn(check_llama, 4)


if __name__ == "__main__":
    test_llama()
