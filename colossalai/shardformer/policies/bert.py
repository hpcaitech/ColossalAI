from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertLMPredictionHead

import colossalai.shardformer.layer.layers as col_nn

from .basepolicy import Argument, Col_Layer, Layer, Policy, Row_Layer


class BertPolicy(Policy):

    @staticmethod
    def argument_policy(config, world_size: int) -> Dict[nn.Module, Argument]:
        return {
            BertLayer:
                Argument(
                    attr_dict={
        # 1. shard hidden size
                        "attention.self.all_head_size": config.hidden_size // world_size,
                        "crossattention.self.all_head_size": config.hidden_size // world_size,
        # 2. shard number of heads
                        "attention.self.num_attention_heads": config.num_attention_heads // world_size,
                        "crossattention.self.num_attention_heads": config.num_attention_heads // world_size,
                    },
                    param_funcs=[BertPolicy.attn_in, BertPolicy.attn_out, BertPolicy.mlp_in, BertPolicy.mlp_out]),
            BertEmbeddings:
                Argument(
                    attr_dict={
        # 1. shard vocab size
        # "word_embeddings.num_embeddings": config.vocab_size // world_size,
        # 2. add the size of the sliced embedding layer excluding the last slice
                        "word_embeddings.dim_size": (config.vocab_size + world_size - 1) // world_size,
                    },
                    param_funcs=[
                        BertPolicy.embedding,
                    ]),
            BertLMPredictionHead:
                Argument(
                    attr_dict={
        # 1. shard vocab size
        # "word_embeddings.num_embeddings": config.vocab_size // world_size,
        # 2. add the size of the sliced embedding layer excluding the last slice
                    },
                    param_funcs=[
                        BertPolicy.unembedding,
                    ])
        }

    @staticmethod
    def binding_policy() -> Dict:
        return {
            "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
        }

    @staticmethod
    def attn_in() -> List:
        return [
            Col_Layer(
                weight="attention.self.query.weight",
                bias="attention.self.query.bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                weight="attention.self.key.weight",
                bias="attention.self.key.bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                weight="attention.self.value.weight",
                bias="attention.self.value.bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                weight="crossattention.self.query.weight",
                bias="crossattention.self.query.bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
            Col_Layer(
                weight="crossattention.self.key.weight",
                bias="crossattention.self.key.bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
            Col_Layer(
                weight="crossattention.self.value.weight",
                bias="crossattention.self.value.bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
        ]

    @staticmethod
    def attn_out() -> List:
        return [
            Row_Layer(
                weight="attention.output.dense.weight",
                bias="attention.output.dense.bias",
                replace_layer=col_nn.Linear1D_Row,
            ),
            Row_Layer(
                weight="crossattention.output.dense.weight",
                bias="crossattention.output.dense.bias",
                replace_layer=col_nn.Linear1D_Row,
                ignore=True,
            ),
        ]

    @staticmethod
    def mlp_in() -> List:
        return [
            Col_Layer(
                weight="intermediate.dense.weight",
                bias="intermediate.dense.bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
        ]

    @staticmethod
    def mlp_out() -> List:
        return [
            Row_Layer(
                weight="output.dense.weight",
                bias="output.dense.bias",
                replace_layer=col_nn.Linear1D_Row,
            ),
        ]

    @staticmethod
    def embedding() -> List:
        return [Col_Layer(
            weight="word_embeddings.weight",
            replace_layer=col_nn.VocabParallelEmbedding1D,
        )]

    @staticmethod
    def unembedding() -> List:
        return [
            Col_Layer(
                weight="decoder.weight",
                bias="decoder.bias",
                replace_layer=col_nn.Linear1D_Col,
        # gather_output=True,
            )
        ]


from transformers import BertForMaskedLM

from colossalai.shardformer.model.modeling_bert import BertForMaskedLM_


class BertForMaskedLMPolicy(BertPolicy):

    @staticmethod
    def inject_policy() -> Tuple[nn.Module, nn.Module]:
        return (BertForMaskedLM, BertForMaskedLM_)


class BertForSequenceClassificationPolicy(BertPolicy):

    @staticmethod
    def inject_policy() -> Dict:
        return {}


# model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# _ = BertForMaskedLMPolicy(model)
# print(isinstance(model,list(_.inject_policy().keys())[0]))
