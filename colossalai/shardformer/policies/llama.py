from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

import colossalai.shardformer.layer.layers as col_nn

from .basepolicy import Argument, Col_Layer, Policy, Row_Layer


class LlamaPolicy(Policy):

    @staticmethod
    def argument_policy(config, world_size: int) -> Dict[nn.Module, Argument]:
        return {
            LlamaDecoderLayer:
                Argument(attr_dict={
                    "self_attn.hidden_size": config.hidden_size // world_size,
                    "self_attn.num_heads": config.num_attention_heads // world_size,
                },
                         param_funcs=[LlamaPolicy.attn_layer, LlamaPolicy.mlp_layer]),
            LlamaModel:
                Argument(attr_dict={}, param_funcs=[LlamaPolicy.embeddings])
        }

    @staticmethod
    def attn_layer() -> List:
        return [
            Col_Layer(
                suffix="self_attn.q_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="self_attn.k_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="self_attn.v_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Row_Layer(
                suffix="self_attn.o_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
            )
        ]

    @staticmethod
    def mlp_layer() -> List:
        return [
            Col_Layer(
                suffix="mlp.gate_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                gather_output=True,
            ),
            Col_Layer(
                suffix="mlp.up_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
                gather_output=True,
            ),
            Col_Layer(
                suffix="mlp.down_proj",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                gather_output=True,
            ),
        ]

    @staticmethod
    def embeddings() -> List:
        return [Col_Layer(
            suffix="embed_tokens",
            weight="weight",
            replace_layer=col_nn.VocabParallelEmbedding1D,
        )]

from transformers import LlamaForCausalLM


class LlamaForCausalLMPolicy(LlamaPolicy):

    @staticmethod
    def argument(config, world_size):
        llamapolicy = LlamaPolicy.argument_policy(config, world_size)
        argument = {LlamaForCausalLM: Argument(attr_dict={}, param_funcs=[LlamaForCausalLMPolicy.lm_head])}
        argument.update(llamapolicy)

    @staticmethod
    def lm_head() -> List:
        return [Col_Layer(suffix="lm_head", weight="weight", replace_layer=col_nn.Linear1D_Col, gather_output=True)]


from transformers import LlamaForSequenceClassification


class LlamaForSequenceClassificationPolicy(LlamaPolicy):

    @staticmethod
    def argument(config, world_size):
        llamapolicy = LlamaPolicy.argument_policy(config, world_size)
        argument = {
            LlamaForSequenceClassification:
                Argument(attr_dict={}, param_funcs=[LlamaForSequenceClassificationPolicy.score])
        }
        argument.update(llamapolicy)

    @staticmethod
    def score() -> List:
        return [Col_Layer(suffix="score", weight="weight", replace_layer=col_nn.Linear1D_Col, gather_output=True)]
