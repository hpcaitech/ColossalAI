import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertLMPredictionHead

import colossalai.shardformer.layer.layers as col_nn

from ..shard.shard_config import ShardConfig
from ..utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class ParallelModule():

    def __init__(self):
        pass


class BertPolicy(Policy):

    def preprocess(self, shard_config: ShardConfig = None):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        # TODO:
        vocab_size = self.model.config.vocab_size
        world_size = shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self, shard_config: ShardConfig = None):
        return {
            BertLayer:
                ModulePolicyDescription(
                    attribute_replacement={
        # 1. shard hidden size
                        "attention.self.all_head_size":
                            self.model.config.hidden_size // shard_config.tensor_parallel_size,
                        "crossattention.self.all_head_size":
                            self.model.config.hidden_size // shard_config.tensor_parallel_size,
        # 2. shard number of heads
                        "attention.self.num_attention_heads":
                            self.model.config.num_attention_heads // shard_config.tensor_parallel_size,
                        "crossattention.self.num_attention_heads":
                            self.model.config.num_attention_heads // shard_config.tensor_parallel_size,
                    },
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="attention.self.query",
                            target_module=ParallelModule,
                        ),
                    ])
        }

    def new_model_class(self):
        # do nothing
        return None

    def postprocess(self):
        binding_map = {"bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            param = nn.Parameter(param)
            setattr_(self.model, k, param)
            setattr_(self.model, v, param)
        return self.model


class BertForMaskedLMPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()


class BertForSequenceClassificationPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()
