from functools import partial

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaRMSNorm

from colossalai.gptq.cai_gptq import ColCaiQuantLinear, RowCaiQuantLinear
from colossalai.shardformer.layer import VocabParallelEmbedding1D
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription
# import colossalai
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

from ..modeling.llama import LlamaInferenceForwards, get_llama_vllm_rmsnorm_forward

try:
    from colossalai.kernel.triton.rms_norm import rmsnorm_forward
    HAS_TRITON_RMSNORM = True
except:
    print("you should install triton from https://github.com/openai/triton")
    HAS_TRITON_RMSNORM = False


def get_triton_rmsnorm_forward():
    if HAS_TRITON_RMSNORM:

        def _triton_rmsnorm_forward(self: LlamaRMSNorm, hidden_states: torch.Tensor):
            return rmsnorm_forward(hidden_states, self.weight.data, self.variance_epsilon)

        return _triton_rmsnorm_forward
    else:
        return None


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = {}
        if not self.shard_config.inference_gptq:
            policy = super().module_policy()
        else:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = \
                    self.model.config.num_key_value_heads // self.shard_config.tensor_parallel_size

            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=ColCaiQuantLinear,
                        kwargs={'split_num': 1},
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=RowCaiQuantLinear,
                        kwargs={'split_num': 1},
                    )
                ],
            )

            self.append_or_create_submodule_replacement(description=SubModuleReplacementDescription(
                suffix="embed_tokens",
                target_module=VocabParallelEmbedding1D,
            ),
                                                        policy=policy,
                                                        target_key=LlamaModel)

        self.shard_config._infer()

        infer_forward = LlamaInferenceForwards.llama_model_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)

        infer_forward = LlamaInferenceForwards.llama_decoder_layer_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=LlamaDecoderLayer)

        infer_forward = LlamaInferenceForwards.llama_flash_attn_kvcache_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=LlamaAttention)

        infer_forward = None
        if HAS_TRITON_RMSNORM:
            infer_forward = get_triton_rmsnorm_forward()
        else:
            # NOTE: adding rms_norm from cuda kernels caused precision issue, fix @tiandiao123
            infer_forward = get_llama_vllm_rmsnorm_forward()

        if infer_forward is not None:
            method_replacement = {'forward': partial(infer_forward)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=LlamaRMSNorm)

        return policy
