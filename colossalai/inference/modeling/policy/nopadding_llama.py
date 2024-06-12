from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel, LlamaRMSNorm

from colossalai.inference.config import RPC_PARAM
from colossalai.inference.modeling.models.nopadding_llama import (
    NopadLlamaAttention,
    NopadLlamaMLP,
    llama_causal_lm_forward,
    llama_decoder_layer_forward,
    llama_model_forward,
    llama_rmsnorm_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class NoPaddingLlamaModelInferPolicy(LlamaForCausalLMPolicy, RPC_PARAM):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    self.model.config.num_key_value_heads // self.shard_config.tensor_parallel_size
                )
        else:
            decoder_attribute_replacement = None

        policy[LlamaDecoderLayer] = ModulePolicyDescription(
            attribute_replacement=decoder_attribute_replacement,
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="mlp.gate_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="mlp.up_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="mlp.down_proj",
                    target_module=Linear1D_Row,
                ),
                SubModuleReplacementDescription(
                    suffix="mlp",
                    target_module=NopadLlamaMLP,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn.q_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn.k_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn.v_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn.o_proj",
                    target_module=Linear1D_Row,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn",
                    target_module=NopadLlamaAttention,
                    kwargs={
                        "model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"],
                    },
                ),
            ],
        )

        policy[LlamaForCausalLM] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="lm_head", target_module=Linear1D_Col, kwargs={"gather_output": True}
                )
            ],
        )

        # self.shard_config._infer()
        self.append_or_create_method_replacement(
            description={"forward": llama_causal_lm_forward}, policy=policy, target_key=LlamaForCausalLM
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_model_forward}, policy=policy, target_key=LlamaModel
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_decoder_layer_forward}, policy=policy, target_key=LlamaDecoderLayer
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_rmsnorm_forward}, policy=policy, target_key=LlamaRMSNorm
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model, self.model.config.rope_theta)
        return self.model

    def to_rpc_param(self) -> str:
        return __class__.__name__

    @staticmethod
    def from_rpc_param() -> "NoPaddingLlamaModelInferPolicy":
        return NoPaddingLlamaModelInferPolicy()
