from functools import partial
from typing import Callable, Dict, List

from torch import Tensor, nn

import colossalai.shardformer.layer as col_nn

from ..modeling.gpt2 import GPT2PipelineForwards, get_gpt2_flash_attention_forward, gpt2_sequence_parallel_forward_fn
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "GPT2Policy",
    "GPT2ModelPolicy",
    "GPT2LMHeadModelPolicy",
    "GPT2DoubleHeadsModelPolicy",
    "GPT2ForTokenClassificationPolicy",
    "GPT2ForSequenceClassificationPolicy",
]


class GPT2Policy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model

        policy = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        sp_mode = self.shard_config.sequence_parallelism_mode if self.shard_config.enable_sequence_parallelism else None
        overlap = self.shard_config.enable_sequence_overlap
        sp_partial_derived = sp_mode in ["1"]

        if self.shard_config.enable_tensor_parallelism:
            policy[GPT2Model] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="wte",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="drop",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ]
            )

            policy[GPT2Block] = ModulePolicyDescription(
                attribute_replacement={
                    "attn.embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "attn.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.c_attn",
                        target_module=col_nn.GPT2FusedLinearConv1D_Col,
                        kwargs={"n_fused": 3, "seq_parallel_mode": sp_mode, "overlap": overlap},
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.c_proj",
                        target_module=col_nn.GPT2FusedLinearConv1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.c_fc",
                        target_module=col_nn.GPT2FusedLinearConv1D_Col,
                        kwargs={"n_fused": 1, "seq_parallel_mode": sp_mode, "overlap": overlap},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.c_proj",
                        target_module=col_nn.GPT2FusedLinearConv1D_Row,
                        kwargs={
                            "seq_parallel_mode": sp_mode,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.attn_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.resid_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="ln_f",
                target_module=norm_cls,
            ),
            policy=policy,
            target_key=GPT2Model,
        )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="ln_1",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                SubModuleReplacementDescription(
                    suffix="ln_2",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
                SubModuleReplacementDescription(
                    suffix="ln_cross_attn",
                    target_module=norm_cls,
                    ignore_if_not_exist=True,
                    kwargs={"sp_partial_derived": sp_partial_derived},
                ),
            ],
            policy=policy,
            target_key=GPT2Block,
        )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_gpt2_flash_attention_forward(),
                },
                policy=policy,
                target_key=GPT2Attention,
            )

        if sp_mode == "1":
            policy[GPT2Model].method_replacement = {"forward": gpt2_sequence_parallel_forward_fn(self.shard_config)}

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.h), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.wte)
            held_layers.append(module.wpe)
            held_layers.append(module.drop)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError("set_pipeline_forward method can only be called when pipeline parallel is enabled.")
        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer

        layers_per_stage = Policy.distribute_layers(len(module.h), stage_manager.num_stages)
        stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
        method_replacement = {
            "forward": partial(
                new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
            )
        }
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=model_cls)


# GPT2Model
class GPT2ModelPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model

        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2Model, new_forward=GPT2PipelineForwards.gpt2_model_forward, policy=policy
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2Model."""
        return []


# GPT2LMHeadModel
class GPT2LMHeadModelPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2LMHeadModel: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2LMHeadModel,
                new_forward=GPT2PipelineForwards.gpt2_lmhead_model_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """The weights of wte and lm_head are shared."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None:
            if stage_manager.num_stages > 1 and id(module.transformer.wte.weight) == id(module.lm_head.weight):
                first_stage, last_stage = 0, stage_manager.num_stages - 1
                return [{first_stage: module.transformer.wte.weight, last_stage: module.lm_head.weight}]
        return []


# GPT2DoubleHeadsModel
class GPT2DoubleHeadsModelPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2DoubleHeadsModel: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2DoubleHeadsModel,
                new_forward=GPT2PipelineForwards.gpt2_double_heads_model_forward,
                policy=module_policy,
            )

        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            multiple_choice_head = self.model.multiple_choice_head
            held_layers.append(self.model.lm_head)
            held_layers.append(multiple_choice_head.summary)
            held_layers.append(multiple_choice_head.activation)
            held_layers.append(multiple_choice_head.first_dropout)
            held_layers.append(multiple_choice_head.last_dropout)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """The weights of wte and lm_head are shared."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None:
            if stage_manager.num_stages > 1 and id(module.transformer.wte.weight) == id(module.lm_head.weight):
                first_stage, last_stage = 0, stage_manager.num_stages - 1
                return [{first_stage: module.transformer.wte.weight, last_stage: module.lm_head.weight}]
        return []


# GPT2ForQuestionAnswering
class GPT2ForQuestionAnsweringPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForQuestionAnswering

        module_policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForQuestionAnswering,
                new_forward=GPT2PipelineForwards.gpt2_for_question_answering_forward,
                policy=module_policy,
            )

        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared_params in gpt2 for QA."""
        return []


# GPT2ForTokenClassification
class GPT2ForTokenClassificationPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForTokenClassification

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2ForTokenClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(suffix="dropout", target_module=col_nn.DropoutForParallelInput)
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForTokenClassification,
                new_forward=GPT2PipelineForwards.gpt2_for_token_classification_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []


# GPT2ForSequenceClassification
class GPT2ForSequenceClassificationPolicy(GPT2Policy):
    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification

        module_policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForSequenceClassification,
                new_forward=GPT2PipelineForwards.gpt2_for_sequence_classification_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []
