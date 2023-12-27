import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralSparseMoeBlock

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
    def from_native_module(module: MixtralSparseMoeBlock, enable_kernel: bool) -> nn.Module:
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
                num_experts=8,
                hidden_size=module.hidden_dim,
                intermediate_size=module.ffn_dim,
                router_top_k=module.top_k,
                router_norm=True,
                router_loss=False,
                # router_capacity_factor_train=
                # router_capacity_factor_eval=
                mlp_activation="silu",
                mlp_gated=True,
                # enable_load_balance=
                # load_balance_tolerance=
                # load_balance_beam_width=
                # load_balance_group_swap_factor=
                enable_kernel=enable_kernel,
                # enable_comm_overlap=
                # enable_hierarchical_comm=
                return_gate_logits=True,
            )
            dtype = module.gate.weight.dtype
            device = module.gate.weight.device
            sparse_mlp = SparseMLP(**moe_kwargs).to(dtype).to(device)

        return sparse_mlp


def replace_moe_layer(model: nn.Module, enable_kernel: bool = False) -> nn.Module:
    """
    Reverse the replace layer operation

    Args:
        module (torch.nn.Module): The object of layer to shard
    """
    if isinstance(model, MixtralDecoderLayer):
        model.block_sparse_moe = MixtralSparseMLP.from_native_module(
            model.block_sparse_moe, enable_kernel=enable_kernel
        )
    else:
        for _, child in model.named_children():
            replace_moe_layer(child, enable_kernel)
