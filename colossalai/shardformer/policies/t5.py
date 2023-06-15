from typing import Dict

import torch.nn as nn
from torch.nn import Embedding
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Model,
    T5Stack,
)

import colossalai.shardformer.layer.layers as col_nn

from .basepolicy import Argument, Col_Layer, Dropout_Layer, Embedding_Layer, Policy, Row_Layer


class T5ModelPolicy(Policy):

    @staticmethod
    def argument_policy(config, world_size: int) -> Dict[nn.Module, Argument]:
        print('config heads', config.num_heads)
        return {
            T5Stack:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout, T5ModelPolicy.embedding]),
            T5Block:
                Argument(attr_dict={}, param_funcs=[]),
            T5LayerSelfAttention:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout]),
            T5LayerCrossAttention:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout]),
            T5Attention:
                Argument(attr_dict={
                    "d_model": config.d_model // world_size,
                    "n_heads": config.num_heads // world_size,
                    "inner_dim": config.num_heads * config.d_kv // world_size,
                },
                         param_funcs=[T5ModelPolicy.attn_layer]),
            T5LayerFF:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout]),
            T5DenseGatedActDense:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout, T5ModelPolicy.dense_gated_layer]),
            T5DenseActDense:
                Argument(attr_dict={}, param_funcs=[T5ModelPolicy.dropout, T5ModelPolicy.dense_act_layer]),
        }

    @staticmethod
    def dense_gated_layer():
        return [
            Col_Layer(
                suffix="wi_0",
                weight="weight",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Row_Layer(
                suffix="wi_1",
                weight="weight",
                replace_layer=col_nn.Linear1D_Row,
            ),
            Col_Layer(suffix="wo", weight="weight", replace_layer=col_nn.Linear1D_Col, gather_output=True)
        ]

    @staticmethod
    def dense_act_layer():
        return [
            Col_Layer(
                suffix="wi",
                weight="weight",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Row_Layer(
                suffix="wo",
                weight="weight",
                replace_layer=col_nn.Linear1D_Row,
            )
        ]

    @staticmethod
    def attn_layer():
        return [
            Col_Layer(
                suffix="q",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="k",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="v",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Row_Layer(
                suffix="o",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
            ),
        ]

    @staticmethod
    def dropout():
        return [Dropout_Layer(
            suffix="dropout",
            p="p",
            replace_layer=col_nn.Dropout1D,
        )]

    @staticmethod
    def embedding():
        return [
            Embedding_Layer(
                suffix="block[0].layer[0].SelfAttention.relative_attention_bias",
                weight="weight",
                replace_layer=col_nn.Embedding1D,
                gather_output=False,
            )
        ]


from transformers import T5ForConditionalGeneration


class T5ForConditionalGenerationPolicy(T5ModelPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        base_argument = T5ModelPolicy.argument_policy(config, world_size)
        argument = {
            T5ForConditionalGeneration: Argument(attr_dict={}, param_funcs=[T5ForConditionalGenerationPolicy.lm_head])
        }
        argument.update(base_argument)
        return argument

    @staticmethod
    def lm_head():
        return [Col_Layer(
            suffix="lm_head",
            weight="weight",
            replace_layer=col_nn.Linear1D_Col,
            gather_output=True,
        )]


from transformers import T5EncoderModel


class T5EncoderModelPolicy(T5ModelPolicy):
    pass
