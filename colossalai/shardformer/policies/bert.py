from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertLMPredictionHead

import colossalai.shardformer.layer.layers as col_nn

from .basepolicy import Argument, Col_Layer, Dropout_Layer, Policy, Row_Layer


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
                        "word_embeddings.dim_size": (config.vocab_size + world_size - 1) // world_size,
                    },
                    param_funcs=[
                        BertPolicy.embedding,
                    ]),
        }

    @staticmethod
    def attn_in():
        return [
            Col_Layer(
                suffix="attention.self.query",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="attention.self.key",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Col_Layer(
                suffix="attention.self.value",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
            Dropout_Layer(
                suffix="attention.self.dropout",
                p="p",
                replace_layer=col_nn.Dropout1D,
            ),
            Col_Layer(
                suffix="crossattention.self.query",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
            Col_Layer(
                suffix="crossattention.self.key",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
            Col_Layer(
                suffix="crossattention.self.value",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                ignore=True,
            ),
        ]

    @staticmethod
    def attn_out():
        return [
            Row_Layer(
                suffix="attention.output.dense",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
            ),
            Dropout_Layer(
                suffix="attention.output.dropout",
                p="p",
                replace_layer=col_nn.Dropout1D,
            ),
            Row_Layer(
                suffix="crossattention.output.dense",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
                ignore=True,
            ),
        ]

    @staticmethod
    def mlp_in():
        return [
            Col_Layer(
                suffix="intermediate.dense",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
            ),
        ]

    @staticmethod
    def mlp_out():
        return [
            Row_Layer(
                suffix="output.dense",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Row,
            ),
            Dropout_Layer(
                suffix="output.dropout",
                p="p",
                replace_layer=col_nn.Dropout1D,
            )
        ]

    @staticmethod
    def embedding():
        return [Col_Layer(
            suffix="word_embeddings",
            weight="weight",
            replace_layer=col_nn.VocabParallelEmbedding1D,
        )]

    @staticmethod
    def unembedding():
        return [
            Col_Layer(
                suffix="decoder",
                weight="weight",
                bias="bias",
                replace_layer=col_nn.Linear1D_Col,
                gather_output=True,
            )
        ]


# BertModel
class BertModelPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        return BertPolicy.argument_policy(config, world_size)


# BertForPretraining
class BertForPretrainingPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        base_argument = BertPolicy.argument_policy(config, world_size)
        argument = {
            BertLMPredictionHead: Argument(attr_dict={}, param_funcs=[
                BertPolicy.unembedding,
            ]),
        }
        argument.update(base_argument)
        return argument

    @staticmethod
    def inject_policy():
        return None

    @staticmethod
    def binding_policy():
        return {
            "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
        }


# BertForMaskedLM
from colossalai.shardformer.model.modeling_bert import BertForMaskedLM_


class BertForMaskedLMPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        base_argument = BertPolicy.argument_policy(config, world_size)
        argument = {
            BertLMPredictionHead: Argument(attr_dict={}, param_funcs=[
                BertPolicy.unembedding,
            ]),
        }
        argument.update(base_argument)
        return argument

    @staticmethod
    def inject_policy():
        # return (BertForMaskedLM, BertForMaskedLM_)
        return None

    @staticmethod
    def binding_policy():
        return {
            "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
        }


# BertLMHeadModel
class BertLMHeadModelPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        base_argument = BertPolicy.argument_policy(config, world_size)
        argument = {
            BertLMPredictionHead: Argument(attr_dict={}, param_funcs=[
                BertPolicy.unembedding,
            ]),
        }
        argument.update(base_argument)
        return argument

    @staticmethod
    def inject_policy():
        return None

    @staticmethod
    def binding_policy():
        return {
            "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
        }


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        return BertPolicy.argument_policy(config, world_size)


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        return BertPolicy.argument_policy(config, world_size)


# BertForMultipleChoice
class BertForMultipleChoicePolicy(BertPolicy):

    @staticmethod
    def argument_policy(config, world_size):
        return BertPolicy.argument_policy(config, world_size)
