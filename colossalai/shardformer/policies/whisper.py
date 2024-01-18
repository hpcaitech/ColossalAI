import warnings
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor

import colossalai.shardformer.layer as col_nn

from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.whisper import (
    WhisperPipelineForwards,
    get_jit_fused_whisper_decoder_layer_forward,
    get_jit_fused_whisper_encoder_layer_forward,
    get_whisper_flash_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "WhisperPolicy",
    "WhisperModelPolicy",
    "WhisperForConditionalGenerationPolicy",
    "WhisperForAudioClassificationPolicy",
]


class WhisperPolicy(Policy):
    def __init__(self) -> None:
        super().__init__()
        import transformers
        from packaging.version import Version

        assert Version(transformers.__version__) <= Version(
            "4.33.0"
        ), "The Whisper model should run on a transformers version not greater than 4.33.0."

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.whisper.modeling_whisper import (
            WhisperAttention,
            WhisperDecoder,
            WhisperDecoderLayer,
            WhisperEncoder,
            WhisperEncoderLayer,
        )

        policy = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Whisper dosen't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        # TODO using the jit fused add_and_dropout affect the accuracy
        if self.shard_config.enable_jit_fused:
            self.shard_config.enable_jit_fused = False
            warnings.warn("Whisper dosen't support jit fused operator now, will ignore the jit fused operator flag.")

        if self.shard_config.enable_tensor_parallelism:
            policy[WhisperEncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.embed_dim": self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads": self.model.config.encoder_attention_heads
                    // self.shard_config.tensor_parallel_size,
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
                ],
            )

            policy[WhisperDecoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.embed_dim": self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads": self.model.config.decoder_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "encoder_attn.embed_dim": self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "encoder_attn.num_heads": self.model.config.encoder_attention_heads
                    // self.shard_config.tensor_parallel_size,
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
                ],
            )

            policy[WhisperDecoder] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="embed_tokens",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                ]
            )

        # optimization configuration
        # Handle encoder layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=WhisperEncoderLayer,
        )

        # Handle decoder layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="self_attn_layer_norm",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="final_layer_norm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=WhisperDecoderLayer,
        )

        # handle encoder layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=WhisperEncoder,
        )

        # handle decoder layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=WhisperDecoder,
        )

        # enable flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_whisper_flash_attention_forward(),
                },
                policy=policy,
                target_key=WhisperAttention,
            )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_whisper_decoder_layer_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=WhisperDecoderLayer,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_whisper_encoder_layer_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=WhisperEncoderLayer,
            )

        return policy

    def add_lm_head_policy(self, base_policy):
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

        # optimize for tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="proj_out", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}
                ),
                policy=base_policy,
                target_key=WhisperForConditionalGeneration,
            )

        return base_policy

    def postprocess(self):
        return self.model

    @staticmethod
    def distribute_whisper_layers(
        num_encoder_layers: int, num_decoder_layers: int, num_stages: int
    ) -> Tuple[List[int], int]:
        """
        Distribute whisper layers into stages when pipeline parallel is used.
        Return the layer distribution as a list and the starting stage of decoder.
        If decoder doesn't exist, returned decoder starting stage is set to num_encoder_layers.
        """

        # number of encoder layers must be a positive integer
        if num_encoder_layers <= 0:
            raise ValueError("The number of encoder layers for whisper must be a positive integer.")

        # number of layers should be large enough to fill in every stage
        if num_encoder_layers + num_decoder_layers < num_stages:
            raise ValueError("The total number of layers can't be smaller than number of stages.")

        # in the case of whisperEncoderModel, set decoder starting stage to num_stages since it doesn't exist
        if num_decoder_layers == 0:
            return Policy.distribute_layers(num_encoder_layers, num_stages), num_stages

        # the number of stages distributed between encoder and decoder is optmized in this way:
        # num_encoder_stages = argmin(abs(num_encoder_layers / encoder_stages - num_decoder_layers / decoder_stages))
        #                   s.t. num_encoder_stages + num_decoder_stages = num_stages, num_encoder_stages >= 1, num_decoder_stages >= 1
        def objective(num_encoder_stages):
            return abs(num_encoder_layers / num_encoder_stages - num_decoder_layers / (num_stages - num_encoder_stages))

        num_encoder_stages = np.argmin([objective(i) for i in range(1, num_stages)]) + 1
        num_decoder_stages = num_stages - num_encoder_stages

        encoder_distribution = Policy.distribute_layers(num_encoder_layers, num_encoder_stages)
        decoder_distribution = Policy.distribute_layers(num_decoder_layers, num_decoder_stages)
        return encoder_distribution + decoder_distribution, num_encoder_stages

    @staticmethod
    def get_whisper_stage_index(
        layers_per_stage: List[int], stage: int, decoder_starting_stage: int
    ) -> Tuple[bool, int, int]:
        """
        Input the distribution of layers among stages, the current stage and the first stage of decoder.
        Return the starting/ending idx of layers in encoder/decoder
        """
        if stage < decoder_starting_stage:
            return Policy.get_stage_index(layers_per_stage[:decoder_starting_stage], stage)
        else:
            return Policy.get_stage_index(layers_per_stage[decoder_starting_stage:], stage - decoder_starting_stage)

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"
        stage_manager = self.pipeline_stage_manager

        if self.model.__class__.__name__ == "WhisperModel":
            model = self.model
        elif self.model.__class__.__name__ == "WhisperForConditionalGeneration":
            model = self.model.model
        else:
            model = None

        if model:
            encoder = self.model.get_encoder()
            decoder = self.model.get_decoder()
        else:
            # whisper for audio classification holds encoder only
            encoder = self.model.encoder
            decoder = None

        num_encoder_layers = len(encoder.layers)
        if decoder:
            num_decoder_layers = len(decoder.layers)
        else:
            num_decoder_layers = 0

        held_layers = []
        layers_per_stage, decoder_starting_stage = WhisperPolicy.distribute_whisper_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages
        )
        start_idx, end_idx = WhisperPolicy.get_whisper_stage_index(
            layers_per_stage, stage_manager.stage, decoder_starting_stage
        )

        if stage_manager.stage < decoder_starting_stage:
            # current stage is in whisper's encoder
            if stage_manager.is_first_stage():
                held_layers.append(encoder.embed_positions)
                held_layers.append(encoder.conv1)
                held_layers.append(encoder.conv2)
            if stage_manager.stage == decoder_starting_stage - 1:
                held_layers.append(encoder.layer_norm)
            held_layers.extend(encoder.layers[start_idx:end_idx])
        else:
            # current stage is in whisper's decoder
            # TODO:(Jianghai) We divide encoder and decoder layers into different parts here,
            # the case encoder and decoder put in same stage should be add in the future.
            if stage_manager.stage == decoder_starting_stage:
                held_layers.append(decoder.embed_tokens)
                held_layers.append(decoder.embed_positions)
            if stage_manager.is_last_stage():
                held_layers.append(decoder.layer_norm)
            held_layers.extend(decoder.layers[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager

        if self.model.__class__.__name__ == "WhisperModel":
            model = self.model
        elif self.model.__class__.__name__ == "WhisperForConditionalGeneration":
            model = self.model.model
        else:
            model = None

        if model:
            encoder = self.model.get_encoder()
            decoder = self.model.get_decoder()
        else:
            encoder = self.model.encoder
            decoder = None

        num_encoder_layers = len(encoder.layers)
        if decoder:
            num_decoder_layers = len(decoder.layers)
        else:
            num_decoder_layers = 0

        layers_per_stage, decoder_starting_stage = WhisperPolicy.distribute_whisper_layers(
            num_encoder_layers, num_decoder_layers, stage_manager.num_stages
        )
        stage_index = WhisperPolicy.get_whisper_stage_index(
            layers_per_stage, stage_manager.stage, decoder_starting_stage
        )

        method_replacement = {
            "forward": partial(
                new_forward,
                stage_manager=stage_manager,
                stage_index=stage_index,
                decoder_starting_stage=decoder_starting_stage,
            )
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)


# WhisperModel
class WhisperModelPolicy(WhisperPolicy):
    def module_policy(self):
        from transformers import WhisperModel

        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=WhisperModel, new_forward=WhisperPipelineForwards.whisper_model_forward, policy=policy
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        "no shared params in whisper model"
        return []


# WhisperForConditionalGeneration
class WhisperForConditionalGenerationPolicy(WhisperPolicy):
    def module_policy(self):
        from transformers import WhisperForConditionalGeneration

        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=WhisperForConditionalGeneration,
                new_forward=WhisperPipelineForwards.whisper_for_conditional_generation_forward,
                policy=policy,
            )
        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.proj_out)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        module = self.model
        model = module.model

        if model:
            encoder = self.model.get_encoder()
            decoder = self.model.get_decoder()
        else:
            encoder = self.model.encoder
            decoder = None

        num_encoder_layers = len(encoder.layers)
        if decoder:
            num_decoder_layers = len(decoder.layers)
        else:
            num_decoder_layers = 0

        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None and stage_manager.num_stages > 1:
            _, decoder_starting_stage = WhisperPolicy.distribute_whisper_layers(
                num_encoder_layers, num_decoder_layers, stage_manager.num_stages
            )
            shared_params = []
            shared_embedding = {}
            if id(module.proj_out) == id(model.decoder.embed_tokens):
                shared_embedding[decoder_starting_stage] = model.decoder.embed_tokens
                shared_embedding[stage_manager.num_stages - 1] = module.proj_out
            if len(shared_embedding) > 0:
                shared_params.append(shared_embedding)
            return shared_params
        return []


# WhisperForAudioClassification
class WhisperForAudioClassificationPolicy(WhisperPolicy):
    def preprocess(self):
        return self.model

    def module_policy(self):
        from transformers import WhisperForAudioClassification

        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=WhisperForAudioClassification,
                new_forward=WhisperPipelineForwards.whisper_for_audio_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.projector)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []
