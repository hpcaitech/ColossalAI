import copy
import os

import pytest
import torch
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration, T5Model, T5Tokenizer, T5TokenizerFast

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.shard import ShardConfig, ShardFormer
from colossalai.testing import assert_hf_output_close, clear_cache_before_run, rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)
tokenizer = T5Tokenizer.from_pretrained("t5-small")


def build_model(world_size, model_fn):
    config = T5Config(decoder_start_token_id=0)
    config.dropout_rate = 0
    org_model = model_fn(config=config).to('cuda')
    shard_config = ShardConfig(tensor_parallel_size=world_size)

    # shard model
    shard_config = ShardConfig(tensor_parallel_size=world_size)
    model_copy = copy.deepcopy(org_model)
    shard_former = ShardFormer(shard_config=shard_config)
    shard_former.init_distributed()
    sharded_model = shard_former.shard_model(model_copy)

    return org_model, sharded_model


def check_forward_backward(org_model, sharded_model):
    # prepare input
    input_ids = tokenizer("translate English to German: The house is wonderful.",
                          return_tensors="pt").input_ids.to('cuda')
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids.to('cuda')

    # switch to train mode
    org_model.train()
    sharded_model.train()

    if isinstance(org_model, T5ForConditionalGeneration):
        org_output = org_model(input_ids=input_ids, labels=labels)
        org_loss = org_output.loss
        shard_output = sharded_model(input_ids=input_ids, labels=labels)
        shard_loss = shard_output.loss
    elif isinstance(org_model, T5Model):
        decoder_input_ids = org_model._shift_right(input_ids)
        org_output = org_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        org_loss = org_output.last_hidden_state.mean()
        shard_output = sharded_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        shard_loss = shard_output.last_hidden_state.mean()
    elif isinstance(org_model, T5EncoderModel):
        org_output = org_model(input_ids=input_ids)
        org_loss = org_output.last_hidden_state.mean()
        shard_output = sharded_model(input_ids=input_ids)
        shard_loss = shard_output.last_hidden_state.mean()

    # key is sharded, so we ignore
    assert_hf_output_close(org_output, shard_output, ignore_keys=['past_key_values'])

    # do backward
    org_loss.backward()
    shard_loss.backward()

    # check grad equality
    org_grad = org_model.encoder.block[0].layer[0].SelfAttention.q.weight.grad
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

    model_fn_list = [
        T5Model,
        T5ForConditionalGeneration,
        T5EncoderModel,
    ]

    for model_fn in model_fn_list:
        org_model, sharded_model = build_model(world_size, model_fn)
        check_forward_backward(org_model, sharded_model)
        torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_t5():
    spawn(check_t5, 2)


if __name__ == "__main__":
    test_t5()
