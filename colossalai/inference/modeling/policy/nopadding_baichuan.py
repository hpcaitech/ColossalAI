from colossalai.inference.config import RPC_PARAM
from colossalai.inference.modeling.layers.baichuan_tp_linear import BaichuanLMHeadLinear1D_Col
from colossalai.inference.modeling.models.nopadding_baichuan import (
    NopadBaichuanAttention,
    NopadBaichuanMLP,
    baichuan_rmsnorm_forward,
)
from colossalai.inference.modeling.models.nopadding_llama import (
    llama_causal_lm_forward,
    llama_decoder_layer_forward,
    llama_model_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.layer import FusedLinear1D_Col, Linear1D_Col, Linear1D_Row
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class NoPaddingBaichuanModelInferPolicy(LlamaForCausalLMPolicy, RPC_PARAM):
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

        # used for Baichuan 7B and 13B for baichuan DecoderLayer
        for DecoderLayer in ["DecoderLayer", "BaichuanLayer"]:
            policy[DecoderLayer] = ModulePolicyDescription(
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
                        target_module=NopadBaichuanMLP,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.W_pack", target_module=FusedLinear1D_Col, kwargs={"n_fused": 3}
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn",
                        target_module=NopadBaichuanAttention,
                        kwargs={
                            "model_shard_infer_config": self.shard_config.extra_kwargs["model_shard_infer_config"],
                        },
                    ),
                ],
            )

            self.append_or_create_method_replacement(
                description={"forward": llama_decoder_layer_forward}, policy=policy, target_key=DecoderLayer
            )

        policy["BaichuanForCausalLM"] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="lm_head", target_module=BaichuanLMHeadLinear1D_Col, kwargs={"gather_output": True}
                )
            ],
        )

        self.append_or_create_method_replacement(
            description={"forward": llama_causal_lm_forward}, policy=policy, target_key="BaichuanForCausalLM"
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_model_forward}, policy=policy, target_key="BaichuanModel"
        )
        self.append_or_create_method_replacement(
            description={"forward": baichuan_rmsnorm_forward}, policy=policy, target_key="RMSNorm"
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model

    def to_rpc_param(self) -> str:
        return __class__.__name__

    @staticmethod
    def from_rpc_param() -> "NoPaddingBaichuanModelInferPolicy":
        return NoPaddingBaichuanModelInferPolicy()
