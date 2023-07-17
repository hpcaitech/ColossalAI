import torch.nn as nn

import colossalai.shardformer.layer as col_nn

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    'WhisperPolicy', 'WhisperModelPolicy', 'WhisperForConditionalGenerationPolicy', 'WhisperForAudioClassification'
]


class WhisperPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        # TODO:
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.whisper.modeling_whisper import (
            WhisperDecoder,
            WhisperDecoderLayer,
            WhisperEncoder,
            WhisperEncoderLayer,
        )

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[WhisperEncoderLayer] = ModulePolicyDescription(attribute_replacement={
                "self_attn.embed_dim":
                    self.model.config.d_model // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads":
                    self.model.config.encoder_attention_heads // self.shard_config.tensor_parallel_size,
            },
                                                                  sub_module_replacement=[
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.q_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.k_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.v_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.out_proj",
                                                                          target_module=col_nn.Linear1D_Row,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="fc1",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="fc2",
                                                                          target_module=col_nn.Linear1D_Row,
                                                                      ),
                                                                  ])

            policy[WhisperDecoderLayer] = ModulePolicyDescription(attribute_replacement={
                "self_attn.embed_dim":
                    self.model.config.d_model // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads":
                    self.model.config.decoder_attention_heads // self.shard_config.tensor_parallel_size,
                "encoder_attn.embed_dim":
                    self.model.config.d_model // self.shard_config.tensor_parallel_size,
                "encoder_attn.num_heads":
                    self.model.config.encoder_attention_heads // self.shard_config.tensor_parallel_size,
            },
                                                                  sub_module_replacement=[
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.q_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.k_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.v_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="self_attn.out_proj",
                                                                          target_module=col_nn.Linear1D_Row,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="encoder_attn.q_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="encoder_attn.k_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="encoder_attn.v_proj",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="encoder_attn.out_proj",
                                                                          target_module=col_nn.Linear1D_Row,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="fc1",
                                                                          target_module=col_nn.Linear1D_Col,
                                                                      ),
                                                                      SubModuleReplacementDescription(
                                                                          suffix="fc2",
                                                                          target_module=col_nn.Linear1D_Row,
                                                                      ),
                                                                  ])

            policy[WhisperDecoder] = ModulePolicyDescription(sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=col_nn.VocabParallelEmbedding1D,
                ),
            ])

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # Handle encoder layer
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=WhisperEncoderLayer)

            # Handle decoder layer
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=WhisperDecoderLayer)

            # handle encoder layer
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=WhisperEncoder)

            # handle decoder layer
            self.append_or_create_submodule_replacement(description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=col_nn.FusedLayerNorm,
                )
            ],
                                                        policy=policy,
                                                        target_key=WhisperDecoder)
        return policy

    def add_lm_head_policy(self, base_policy):
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

        # optimize for tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="proj_out", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}),
                                                        policy=base_policy,
                                                        target_key=WhisperForConditionalGeneration)

        return base_policy

    def postprocess(self):
        return self.model


# WhisperModel
class WhisperModelPolicy(WhisperPolicy):

    def __init__(self) -> None:
        super().__init__()


# WhisperForConditionalGeneration
class WhisperForConditionalGenerationPolicy(WhisperPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        module_policy = super().module_policy()
        module_policy = self.add_lm_head_policy(module_policy)
        return module_policy

    def postprocess(self):
        binding_map = {"model.decoder.embed_tokens.weight": "proj_out.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            setattr_(self.model, v, param)
        return self.model


# WhisperForAudioClassification
class WhisperForAudioClassificationPolicy(WhisperPolicy):

    def __init__(self) -> None:
        super().__init__()
