import os
import random

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    get_scheduler,
)

import colossalai
from colossalai.shardformer.shard import ShardConfig, shard_model
from colossalai.utils import get_current_device, print_rank_0

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def get_args():
    parser = colossalai.get_default_parser()
    parser.add_argument("--mode", type=str, default='inference')
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--model", type=str, default='bert-base-uncased')
    return parser.parse_args()


def forward_verify():
    # launch dist
    colossalai.launch_from_torch(config=get_args().config)
    input = "Hello, my dog is cute"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_input = tokenizer(input, return_tensors='pt').to('cuda')
    # forward
    # orgin model
    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to('cuda')
    org_model.eval()
    org_out = org_model(**tokenized_input)
    print_rank_0(org_out[0])
    # shard model
    shardconfig = ShardConfig(
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
    )
    sharded_model = shard_model(BertForMaskedLM.from_pretrained('bert-base-uncased'), shardconfig).to('cuda')
    sharded_model.eval()
    shard_out = sharded_model(**tokenized_input)
    print_rank_0(shard_out[0])

    assert torch.allclose(org_out[0], shard_out[0], atol=1e-5), "shard model output is not equal to orgin model output"
    print("[OK] shard model output is equal to orgin model output")


def backward_verify():
    # launch dist
    colossalai.launch_from_torch(config=get_args().config)
    input = "Hello, my dog is cute"
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_input = tokenizer(input, padding='max_length', return_tensors='pt').to('cuda')
    labels = tokenized_input['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized_input['labels'] = labels
    # disable dropout
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    # backward
    # orgin model
    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config).to('cuda')
    org_model.train()
    org_out = org_model(**tokenized_input)
    org_loss = org_out.loss
    print_rank_0(org_loss)
    org_loss.backward()

    # shard model
    shardconfig = ShardConfig(
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']),
    )
    sharded_model = shard_model(BertForMaskedLM.from_pretrained('bert-base-uncased', config=config),
                                shardconfig).to('cuda')
    sharded_model.train()
    shard_out = sharded_model(**tokenized_input)
    shard_loss = shard_out.loss
    print_rank_0(shard_loss)
    shard_loss.backward()

    assert torch.allclose(org_out[0], shard_out[0], atol=1e-5), "shard model output is not equal to orgin model output"
    print("[OK] shard model output is equal to orgin model output")


if __name__ == '__main__':
    backward_verify()
