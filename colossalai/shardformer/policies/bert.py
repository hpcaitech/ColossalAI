from typing import Dict, List, Type

import torch.nn as nn
from basepolicy import Policy, Layer
import colossalai.nn as col_nn

class BertPolicy(Policy):
    @staticmethod
    def attn_in() -> List:
        return [
            Layer(
                weight="attention.self.query.weight",
                bias="attention.self.query.bias",
                replace_layer=col_nn.Linear,
            ),
            Layer(
                weight="attention.self.key.weight",
                bias="attention.self.key.bias",
                replace_layer=col_nn.Linear,
            ),
            Layer(
                weight="attention.self.value.weight",
                bias="attention.self.value.bias",
                replace_layer=col_nn.Linear,
            ),
            Layer(
                weight="crossattention.self.query.weight",
                bias="crossattention.self.query.bias",
                replace_layer=col_nn.Linear,
                ignore=True,
            ),
            Layer(
                weight="crossattention.self.key.weight",
                bias="crossattention.self.key.bias",
                replace_layer=col_nn.Linear,
                ignore=True,
            ),
            Layer(
                weight="crossattention.self.value.weight",
                bias="crossattention.self.value.bias",
                replace_layer=col_nn.Linear,
                ignore=True,
            ),

        ]
    
    @staticmethod
    def attn_out() -> List:
        return [
            Layer(
                weight="attention.output.dense.weight",
                bias="attention.output.dense.bias",
                replace_layer=col_nn.Linear,
            ),
            Layer(
                weight="crossattention.output.dense.weight",
                bias="crossattention.output.dense.bias",
                replace=col_nn.Linear,
                ignore=True,
            ),
        ]
    
    @staticmethod
    def mlp_in() -> List:
        return [
             Layer(
                weight="intermediate.dense.weight",
                bias="intermediate.dense.bias",
                replace_layer=col_nn.Linear,
            ),
        ]
    
    @staticmethod
    def mlp_out() -> List:
        return [
             Layer(
                weight="output.dense.weight",
                bias="output.dense.bias",
                replace_layer=col_nn.Linear,
            ),
        ]

    @staticmethod
    def embedding() -> List:
        return [

        ]
    
    @staticmethod
    def unembedding() -> List:
        return [

        ]

from transformers import BertForMaskedLM
from colossalai.shardformer.shardmodel.modeling_bert import BertForMaskedLM_
class BertForMaskedLMPolicy(BertPolicy):
    @staticmethod
    def inject_policy() -> Dict:
        return {BertForMaskedLM: BertForMaskedLM_}
    

    
class BertForSequenceClassificationPolicy(BertPolicy):
    @staticmethod
    def inject_policy() -> Dict:
        return {}


# model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# _ = BertForMaskedLMPolicy(model)
# print(isinstance(model,list(_.inject_policy().keys())[0]))