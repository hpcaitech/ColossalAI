import colossalai.shardformer.layer as col_nn

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    "GPTJPolicy",
    "GPTJModelPolicy",
    "GPTJForCausalLMPolicy",
    "GPTJForSequenceClassificationPolicy",
    "GPTJForQuestionAnsweringPolicy",
    "FlaxGPTJPolicy",
    "FlaxGPTJForCausalLMPolicy",
]


class GPTJPolicy(Policy):
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
        from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJModel

        policy = {}
        use_sequence_parallel = self.shard_config.enable_sequence_parallelism
        overlap = self.shard_config.enable_sequence_overlap
        if self.shard_config.enable_tensor_parallelism:
            policy[GPTJModel] = ModulePolicyDescription(
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

        policy[GPTJBlock] = ModulePolicyDescription(
            attribute_replacement={
                "attn.embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attn.rotary_dim": self.model.config.rotary_dim // self.shard_config.tensor_parallel_size,
                "attn.num_attention_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
            },
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="attn.k_proj",
                    target_module=col_nn.GPT2FusedLinearConv1D_Col,
                    kwargs={"n_fused": 3, "seq_parallel": use_sequence_parallel, "overlap": overlap},
                ),
                SubModuleReplacementDescription(
                    suffix="attn.q_proj",
                    target_module=col_nn.GPT2FusedLinearConv1D_Col,
                    kwargs={"n_fused": 3, "seq_parallel": use_sequence_parallel, "overlap": overlap},
                ),
                SubModuleReplacementDescription(
                    suffix="attn.v_proj",
                    target_module=col_nn.GPT2FusedLinearConv1D_Col,
                    kwargs={"n_fused": 3, "seq_parallel": use_sequence_parallel, "overlap": overlap},
                ),
                SubModuleReplacementDescription(
                    suffix="attn.out_proj",
                    target_module=col_nn.GPT2FusedLinearConv1D_Row,
                    kwargs={
                        "seq_parallel": use_sequence_parallel,
                    },
                ),
                SubModuleReplacementDescription(
                    suffix="mlp.fc_in",
                    target_module=col_nn.GPT2FusedLinearConv1D_Col,
                    kwargs={"n_fused": 1, "seq_parallel": use_sequence_parallel, "overlap": overlap},
                ),
                SubModuleReplacementDescription(
                    suffix="mlp.fc_out",
                    target_module=col_nn.GPT2FusedLinearConv1D_Row,
                    kwargs={
                        "seq_parallel": use_sequence_parallel,
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
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="ln_f",
                    target_module=col_nn.FusedLayerNorm,
                ),
                policy=policy,
                target_key=GPTJModel,
            )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="ln_1",
                        target_module=col_nn.FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=GPTJBlock,
            )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_gptj_flash_attention_forward(),
                },
                policy=policy,
                target_key=GPTJAttention,
            )

        if self.shard_config.enable_sequence_parallelism:
            policy[GPTJModel].method_replacement = {"forward": gpt2_sequence_parallel_forward_fn(self.shard_config)}

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


"""
# GPTJModel
class GPTJModelPolicy(GPTJPolicy):

# GPTJForCausalLM
class GPTJForCausalLMPolicy(GPTJPolicy):

# GPTJForSequenceClassification
class GPTJForSequenceClassificationPolicy(GPTJPolicy):

# GPTJForQuestionAnswering
class GPTJForQuestionAnsweringPolicy(GPTJPolicy):

# FlaxGPTJModel
class FlaxGPTJPolicy(GPTJPolicy):

# FlaxGPTJForCausalLMModel
class FlaxGPTJForCausalLMPolicy(GPTJPolicy):
"""
