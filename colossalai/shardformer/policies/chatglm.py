from typing import Dict, Union

import torch.nn as nn
from ....tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import ChatGLMModel, GLMBlock

import colossalai.shardformer.layer as col_nn

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ['ChatGLMModelPolicy', 'ChatGLMForConditionalGenerationPolicy']

class ChatGLMModelPolicy(Policy):

    def config_sanity_check(self):
        pass
    
    def preprocess(self):
        # Resize embedding
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size

        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)

        return self.model
    
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from ....tests.kit.model_zoo.transformers.chatglm2_6b.modeling_chatglm import ChatGLMModel, GLMBlock

        policy = {}

        if self.shard_config.enable_tensor_parallelism:

            policy[GLMBlock] = ModulePolicyDescription(
                attribute_replacement = {},
                sub_module_replacement = [
                        # SubModuleReplacementDescription(
                        #     suffix = "self_attention.query_key_value",
                        #     target_module = col_nn.Linear1D_Col,
                        # ),
                        # SubModuleReplacementDescription(
                        #     suffix = "self_attention.dense",
                        #     target_module = col_nn.Linear1D_Row,
                        # )
                        # SubModuleReplacementDescription(
                        #     suffix = "self_attention.core_attention.attention_dropout",
                        #     target_module = col_nn.DropoutForParallelInput,
                        # )
                    ],)


    def postprocess(self):
        return self.model

