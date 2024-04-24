from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

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
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class BaichuanLinear1D_Col(Linear1D_Col):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        weight: Optional[Parameter] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=weight,
            *args,
            **kwargs,
        )
        self.weight = nn.Parameter(nn.functional.normalize(self.weight))

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch linear layer to a parallelized linear layer.
        """
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.weight.size(1)
        out_features = module.weight.size(0)
        bias = None
        device = module.weight.device
        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )

        linear_1d = Linear1D_Col(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=bias,
            *args,
            **kwargs,
        )

        return linear_1d


class NoPaddingBaichuanModelInferPolicy(LlamaForCausalLMPolicy):
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
                        suffix="self_attn.W_pack",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn",
                        target_module=NopadBaichuanAttention,
                    ),
                ],
            )

        policy["BaichuanForCausalLM"] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="lm_head", target_module=BaichuanLinear1D_Col, kwargs={"gather_output": True}
                )
            ],
        )

        self.append_or_create_method_replacement(
            description={"forward": llama_causal_lm_forward}, policy=policy, target_key="BaichuanForCausalLM"
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_model_forward}, policy=policy, target_key="BaichuanModel"
        )

        # used for Baichuan 7B
        self.append_or_create_method_replacement(
            description={"forward": llama_decoder_layer_forward}, policy=policy, target_key="DecoderLayer"
        )
        # used for Baichuan 13B
        self.append_or_create_method_replacement(
            description={"forward": llama_decoder_layer_forward}, policy=policy, target_key="BaichuanLayer"
        )

        self.append_or_create_method_replacement(
            description={"forward": baichuan_rmsnorm_forward}, policy=policy, target_key="RMSNorm"
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model
