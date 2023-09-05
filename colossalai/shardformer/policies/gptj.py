import colossalai.shardformer.layer as col_nn

from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = []


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
        from transformers.models.gptj.modeling_gptj import GPTJBlock, GPTJModel

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
                "attn.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            },
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="attn.k_attn",
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


"""
# GPTJModel
class GPTJModelPolicy(GPTJPolicy):

# GPTJForCausalLM
class GPTJForCausalLMPolicy(GPTJPolicy):

# GPTJForSequenceClassification
class GPTJForSequenceClassificationPolicy(GPTJPolicy):

# GPTJForQuestionAnswering
class GPTJForQuestionAnsweringPolicy(GPTJPolicy):

# TFGPTJForQuestionAnswering
class TFGPTJPolicy(GPTJPolicy):

# TFGPTJForCausalLM
class TFGPTJCausalLMPolicy(GPTJPolicy):

# TFGPTJForSequenceClassification
class TFGPTJForSequenceClassificationPolicy(GPTJPolicy):

# TFGPTJForQuestionAnswering
class TFGPTJForQuestionAnsweringPolicy(GPTJPolicy):

# FlaxGPTJModel
class FlaxGPTJPolicy(GPTJPolicy):

# FlaxGPTJForCausalLMModel
class FlaxGPTJForCausalLMPolicy(GPTJPolicy):
"""
