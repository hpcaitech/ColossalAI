import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from colossalai.lazy import LazyInitContext
from colossalai.moe import SparseMLP
from colossalai.tensor.moe_tensor.api import get_ep_rank, is_moe_tensor


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
        with torch.no_grad():
            LazyInitContext.materialize(module)

            # get the attributes of the module
            moe_kwargs = dict(
                num_experts=module.num_experts,
                hidden_size=module.hidden_dim,
                intermediate_size=module.ffn_dim,
                router_top_k=module.top_k,
                router_norm=True,
                router_loss=False,
                # router_capacity_factor_train = .
                # router_capacity_factor_eval = .
                mlp_activation="silu",
                mlp_gated=True,
                # enable_load_balance = .
                # load_balance_tolerance = .
                # load_balance_beam_width = .
                # load_balance_group_swap_factor = .
                # enable_kernel = .
                # enable_comm_overlap = .
                # enable_hierarchical_comm = .
                return_gate_logits=True,
            )
            dtype = module.gate.weight.dtype
            device = module.gate.weight.device
            sparse_mlp = SparseMLP(**moe_kwargs).to(dtype).to(device)

            # cat all experts
            w1 = None
            w2 = None
            w3 = None
            for i in module.experts:
                # origin
                wi_1 = i.w1.weight.data.clone().transpose(0, 1).unsqueeze(0)
                wi_2 = i.w2.weight.data.clone().transpose(0, 1).unsqueeze(0)
                wi_3 = i.w3.weight.data.clone().transpose(0, 1).unsqueeze(0)
                # cat
                w1 = wi_1 if w1 is None else torch.cat([w1, wi_1], dim=0)
                w2 = wi_2 if w2 is None else torch.cat([w2, wi_2], dim=0)
                w3 = wi_3 if w3 is None else torch.cat([w3, wi_3], dim=0)

            # get local experts
            if is_moe_tensor(sparse_mlp.experts.wi_gate):
                ep_rank = get_ep_rank(sparse_mlp.experts.wi_gate)
                expert_num = sparse_mlp.experts.wi_gate.shape[0]
                expert_slice = slice(ep_rank * expert_num, (ep_rank + 1) * expert_num)
            else:
                expert_slice = slice(None)
            w1 = w1[expert_slice].clone().detach()
            w2 = w2[expert_slice].clone().detach()
            w3 = w3[expert_slice].clone().detach()
            assert (
                w1.shape == sparse_mlp.experts.wi_gate.shape
            ), f"current shape: {w1.shape}, target shape:{sparse_mlp.experts.wi_gate.shape}"
            assert (
                w2.shape == sparse_mlp.experts.wo.shape
            ), f"current shape: {w2.shape}, target shape:{sparse_mlp.experts.wo.shape}"
            assert (
                w3.shape == sparse_mlp.experts.wi_up.shape
            ), f"current shape: {w3.shape}, target shape:{sparse_mlp.experts.wi_up.shape}"

            # assign new param to colossal moe moudle
            sparse_mlp.experts.wi_gate.data = w1
            sparse_mlp.experts.wi_up.data = w3
            sparse_mlp.experts.wo.data = w2
            sparse_mlp.gate_weight = module.gate.weight

            # TODO: fix
            # the old weight is referenced somewhere so we can not del it.
            # Change data pointer of old weight to release memory.
            # The pointer will not be used and can be any pointer.
            for i in module.experts:
                i.w1.weight.data = w1
                i.w2.weight.data = w2
                i.w3.weight.data = w3

        return sparse_mlp
