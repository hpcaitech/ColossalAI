from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model

import colossalai.shardformer.layer.layers as col_nn

from .basepolicy import Argument, Col_Layer, Layer, Policy, Row_Layer


class GPT2Policy(Policy):

    @staticmethod
    def argument_policy(config, world_size):
        return {
            GPT2Model:
                Argument(attr_dict={}, param_funcs=[
                    GPT2Policy.embedding,
                ]),
            GPT2Block:
                Argument(
                    attr_dict={
        # 1. reduce hidden size
                        "attn.embed_dim": config.hidden_size // world_size,
                        "attn.split_size": config.hidden_size // world_size,
                        "crossattention.embed_dim": config.hidden_size // world_size,
                        "crossattention.split_size": config.hidden_size // world_size,
        # 2. reduce number of heads
                        "attn.num_heads": config.num_attention_heads // world_size,
                        "crossattention.num_heads": config.num_attention_heads // world_size,
                    },
                    param_funcs=[
                        GPT2Policy.attn_in,
                        GPT2Policy.attn_out,
                        GPT2Policy.mlp_in,
                        GPT2Policy.mlp_out,
                    ]),
        }

    @staticmethod
    def attn_in() -> List:
        return [
            Col_Layer(suffix="attn.c_attn",
                      weight="weight",
                      bias="bias",
                      n_cast=3,
                      reversed=True,
                      replace_layer=col_nn.Linear1D_Col),
            Col_Layer(suffix="crossattention.c_attn",
                      weight="weight",
                      bias="bias",
                      n_cast=2,
                      reversed=True,
                      ignore=True,
                      replace_layer=col_nn.Linear1D_Col),
            Col_Layer(suffix="crossattention.q_attn",
                      weight="weight",
                      bias="bias",
                      reversed=True,
                      ignore=True,
                      replace_layer=col_nn.Linear1D_Col)
        ]

    @staticmethod
    def attn_out() -> List:
        return [
            Row_Layer(suffix="attn.c_proj",
                      weight="weight",
                      bias="bias",
                      reversed=True,
                      replace_layer=col_nn.Linear1D_Row),
            Row_Layer(suffix="crossattention.c_proj",
                      weight="weight",
                      bias="bias",
                      reversed=True,
                      ignore=True,
                      replace_layer=col_nn.Linear1D_Row)
        ]

    @staticmethod
    def mlp_in() -> List:
        return [
            Col_Layer(suffix="mlp.c_fc", weight="weight", bias="bias", reversed=True,
                      replace_layer=col_nn.Linear1D_Col),
        ]

    @staticmethod
    def mlp_out() -> List:
        return [
            Row_Layer(suffix="mlp.c_proj",
                      weight="weight",
                      bias="bias",
                      reversed=True,
                      replace_layer=col_nn.Linear1D_Row)
        ]

    @staticmethod
    def embedding() -> List:
        return [Col_Layer(suffix="wte", weight="weight", replace_layer=col_nn.VocabParallelEmbedding1D)]


from transformers import GPT2LMHeadModel


class GPT2LMHeadModelPolicy(GPT2Policy):

    @staticmethod
    def argument_policy(config, world_size):
        base_argument = GPT2Policy.argument_policy(config, world_size)
        argument = {
            GPT2LMHeadModel: Argument(attr_dict={}, param_funcs=[
                GPT2LMHeadModelPolicy.unembedding,
            ]),
        }
        argument.update(base_argument)
        return argument

    @staticmethod
    def unembedding() -> List:
        return [
            Col_Layer(suffix="lm_head",
                      weight="weight",
                      bias="bias",
                      replace_layer=col_nn.Linear1D_Col,
                      gather_output=True)
        ]
