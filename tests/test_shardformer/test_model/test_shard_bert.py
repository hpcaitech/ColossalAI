import copy
import os

import pytest
import torch
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForSequenceClassification,
    BertLMHeadModel,
    BertModel,
)

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import rerun_if_address_is_in_use, spawn

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def build_model(rank, world_size, model):
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0

    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
    org_model_forshard = copy.deepcopy(org_model)

    org_model.to('cuda')
    # TODO: no need to transfer to cuda
    org_model_forshard.to('cuda')
    shard_config = ShardConfig(
        tensor_parallel_size=2,
        tensor_parallel_mode='1d',
    )
    shard_former = ShardFormer(shard_config=shard_config)
    shard_former.init_distributed()
    sharded_model = shard_former.shard_model(org_model_forshard).to('cuda')

    return org_model, sharded_model


def check_forward(org_model, sharded_model):
    input = 'Hello, my dog is cute'
    tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')

    #orgin model
    org_model.eval()
    org_out = org_model(**tokenized_input)

    #shard model
    sharded_model.eval()
    shard_out = sharded_model(**tokenized_input)

    assert torch.allclose(
        org_out[0], shard_out[0],
        atol=1e-5), f"shard model output is not equal to orgin model output\n{org_out[0]}\n{shard_out[0]}"


def check_backward(org_model, sharded_model):
    # prepare input
    input = 'Hello, my dog is cute'
    tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
    labels = tokenized_input['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized_input['labels'] = labels

    #orgin model
    org_model.train()
    org_out = org_model(**tokenized_input)
    org_loss = org_out.loss
    org_loss.backward()
    org_grad = org_model.bert.encoder.layer[0].attention.self.query.weight.grad

    #shard model
    sharded_model.train()
    shard_out = sharded_model(**tokenized_input)
    shard_loss = shard_out.loss
    shard_loss.backward()
    shard_grad = sharded_model.bert.encoder.layer[0].attention.self.query.weight.grad

    shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(2)]
    shard_grad = torch.distributed.all_gather(shard_grad_list, shard_grad)
    all_shard_grad = torch.cat(shard_grad_list, dim=0)

    assert torch.allclose(org_loss, shard_loss,
                          atol=1e-5), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"
    assert torch.allclose(org_grad, all_shard_grad,
                          atol=1e-5), f"shard model grad is not equal to orgin model grad\n{org_grad}\n{shard_grad}"


def check_bert(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    forward_list = [
        BertModel, BertForPreTraining, BertForMaskedLM, BertLMHeadModel, BertForNextSentencePrediction,
        BertForSequenceClassification
    ]
    backward_lsit = [BertForMaskedLM, BertLMHeadModel]

    for model in forward_list:
        org_model, sharded_model = build_model(rank, world_size, model)
        check_forward(org_model, sharded_model)
        if model in backward_lsit:
            check_backward(org_model, sharded_model)

        torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert():
    spawn(check_bert, 2)


if __name__ == "__main__":
    test_bert()
