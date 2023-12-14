import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from colossalai.lazy import LazyInitContext
from colossalai.moe import SparseMLP


class MixtralSparseMLP:
    r"""
    This is a wrapper around the apex fused layernorm implementation. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedLayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface convert a native pytorch layer norm module to FusedLayerNorm module provided by apex."
        )

    @staticmethod
    def from_native_module(module: MixtralSparseMoeBlock, *args, **kwargs) -> nn.Module:
        r"""
        Convert a native pytorch layer norm module to FusedLayerNorm module provided by apex,
        and optionally marking parameters for gradient aggregation.

        Args:
            module (nn.LayerNorm): The native PyTorch LayerNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: Union[FastLayerNorm, FusedLayerNorm].

        Raises:
            AssertionError: If the provided module is not an instance of nn.LayerNorm.
        """

        LazyInitContext.materialize(module)
        # get the attributes of the module
        moe_kwargs = dict(
            num_experts=module.num_experts,
            hidden_size=module.hidden_dim,
            intermediate_size=module.ffn_dim,
            router_top_k=module.top_k,
            # router_capacity_factor_train = module.
            # router_capacity_factor_eval = module.
            # router_min_capacity = module.
            # router_noisy_policy = module.
            # router_drop_tks = module.
            mlp_activation="silu",
            mlp_gated=True,
            # enable_load_balance = module.
            # load_balance_tolerance = module.
            # load_balance_beam_width = module.
            # load_balance_group_swap_factor = module.
            # enable_kernel = module.
            # enable_comm_overlap = module.
            # enable_hierarchical_comm = module.
            return_gate_logits=True,
        )
        dtype = module.gate.weight.dtype
        device = module.gate.weight.device

        sparse_mlp = SparseMLP(**moe_kwargs).to(dtype).to(device)
        w1 = None
        w2 = None
        w3 = None
        for i in module.experts:
            wi_1 = i.w1.weight.data.transpose(0, 1).unsqueeze(0)
            wi_2 = i.w2.weight.data.transpose(0, 1).unsqueeze(0)
            wi_3 = i.w3.weight.data.transpose(0, 1).unsqueeze(0)
            if w1 is None:
                w1 = wi_1
            else:
                w1 = torch.cat([w1, wi_1], dim=0)
            if w2 is None:
                w2 = wi_2
            else:
                w2 = torch.cat([w2, wi_2], dim=0)
            if w3 is None:
                w3 = wi_3
            else:
                w3 = torch.cat([w3, wi_3], dim=0)

        sparse_mlp.experts.wi_gate.data = w1[:2]
        sparse_mlp.experts.wi_up.data = w3[:2]
        sparse_mlp.experts.wo.data = w2[:2]
        sparse_mlp.gate_weight = module.gate.weight

        return sparse_mlp.to(dtype).to(device)
