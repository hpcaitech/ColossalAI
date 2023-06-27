rom typing import Dict, Union

import torch.nn as nn

from transformers.models.vit.modeling_vit import ViTModel, ViTLayer

from colossalai.shardformer.layer.layers import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

class ViTPolicy(Policy):
    
    def preprocess(self):
        # Resize embedding
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size

        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)

        return self.model
    
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        return  {
            ViTLayer:ModulePolicyDescription(
                
            ),
            ViTModel:
                ModulePolicyDescription(
                    attribute_replacement{
                        
                    }
                ),
            
        }

    @staticmethod
    def embedding() -> List:
        return[
            Embedding_Layer(
                suffix="",
                weight="weight",
                replace_layer=col_nn.Embedding1D,
            )
        ]

    @staticmethod
    def dropout():
        return [Dropout_Layer(
            suffix="dropout",
            p="p",
            replace_layer=col_nn.Dropout1D,
        )]




