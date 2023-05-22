from transformers import AutoTokenizer
from transformers import BertForMaskedLM
import colossalai
from colossalai.shardformer.shard.shardmodel import ShardModel
from colossalai.utils import get_current_device, print_rank_0
from colossalai.logging import get_dist_logger
from colossalai.shardformer.shard.shardconfig import ShardConfig
import inspect
import argparse
import torch.nn as nn
import os

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def get_args():
    parser = colossalai.get_default_parser()
    return parser.parse_args()

def inference(model: nn.Module):
    # print(model)
    token = "Hello, my dog is cute"
    inputs = tokenizer(token, return_tensors="pt")
    inputs.to("cuda")
    model.to("cuda")
    outputs = model(**inputs)
    print(outputs)

if __name__ == "__main__":
    args = get_args()
    colossalai.launch_from_torch(config=args.config)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    shard_config = ShardConfig(
        rank = int(str(get_current_device()).split(':')[-1]),
        world_size= int(os.environ['WORLD_SIZE']),
    )
    shardmodel = ShardModel(model, shard_config)
    inference(shardmodel.model)
