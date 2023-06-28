import torch
import torch.distributed as dist

import colossalai.shardformer.layer as col_nn

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


def build_bloom_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    import math

    if dist.is_initialized():
        world_size = dist.get_world_size()
        num_heads = num_heads * world_size

    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2**math.floor(math.log2(num_heads))
    base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                        device=attention_mask.device,
                        dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                                  device=attention_mask.device,
                                  dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    if dist.is_initialized():
        num_heads_per_rank = int(num_heads / dist.get_world_size())
        offset = dist.get_rank() * num_heads_per_rank
        alibi = alibi.view(batch_size, num_heads, 1, seq_length)
        alibi = alibi[:, offset:num_heads_per_rank + offset, :, :]
        return alibi.reshape(batch_size * num_heads_per_rank, 1, seq_length).to(dtype)
    else:
        return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


class BloomPolicy(Policy):

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
        from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel

        return {
            BloomBlock:
                ModulePolicyDescription(
                    attribute_replacement={
        # 1. shard hidden size
                        "self_attention.hidden_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                        "self_attention.split_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
        # 2. shard number of heads
                        "self_attention.num_heads":
                            self.model.config.n_head // self.shard_config.tensor_parallel_size,
                    },
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="self_attention.query_key_value",
                            target_module=col_nn.Linear1D_Col,
        # kwargs={'n_fused': 3}
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attention.dense",
                            target_module=col_nn.Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attention.attention_dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.dense_h_to_4h",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.dense_4h_to_h",
                            target_module=col_nn.Linear1D_Row,
                        ),
                    ]),
            BloomModel:
                ModulePolicyDescription(attribute_replacement={
                    "num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                },
                                        param_replacement=[],
                                        method_replacement={"build_alibi_tensor": build_bloom_alibi_tensor},
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="word_embeddings",
                                                target_module=col_nn.VocabParallelEmbedding1D,
                                            )
                                        ])
        }

    def new_model_class(self):
        # do nothing
        return self.model

    def postprocess(self):
        return self.model


# BertModel
class BloomModelPolicy(BloomPolicy):
    pass


class BloomForCausalLMPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForCausalLM
        policy = super().module_policy()
        # add a new item for casual lm
        new_item = {
            BloomForCausalLM:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(suffix="lm_head",
                                                                            target_module=col_nn.Linear1D_Col,
                                                                            kwargs=dict(gather_output=True))
                                        ])
        }
        policy.update(new_item)
        return policy


class BloomForSequenceClassificationPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForSequenceClassification
        policy = super().module_policy()
        # add a new item for casual lm
        new_item = {
            BloomForSequenceClassification:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(suffix="score",
                                                                            target_module=col_nn.Linear1D_Col,
                                                                            kwargs=dict(gather_output=True))
                                        ])
        }
        policy.update(new_item)
        return policy


class BloomForTokenClassificationPolicy(BloomPolicy):

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomForTokenClassification
        policy = super().module_policy()
        # add a new item for casual lm
        new_item = {
            BloomForTokenClassification:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(suffix="classifier",
                                                                            target_module=col_nn.Linear1D_Col,
                                                                            kwargs=dict(gather_output=True)),
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=col_nn.DropoutForReplicatedInput,
                                            ),
                                        ])
        }
        policy.update(new_item)
        return policy


class BloomForQuestionAnsweringPolicy(BloomPolicy):
    # No head sharding as the output features is only 2
    pass
