"""Gemma 4 replicated experts: shared dispatch/GEMM, native router and decoder semantics.

This adapter is DP-only. Expert weights retain their HF names, layout and Parameter identities; they have no EP
tags or gradient scaling. CPU uses the explicit reference backend; CUDA requires the shared grouped GEMM backend.
"""

from dataclasses import replace

import torch
import torch.nn as nn

from colossalai.moe import GroupedMMExperts, NCCLAllToAllDispatcher, ReferenceExperts


class Gemma4GroupedExperts(nn.Module):
    @staticmethod
    def from_native_module(module, process_group=None, **kwargs):
        # process_group belongs to TP. The policy checks TP=1 and rejects EP; do not infer EP from WORLD.
        layer = Gemma4GroupedExperts()
        layer.num_experts = module.num_experts
        layer.hidden_dim = module.hidden_dim
        layer.intermediate_dim = module.intermediate_dim
        layer.act_fn = module.act_fn
        layer.gate_up_proj = module.gate_up_proj
        layer.down_proj = module.down_proj
        layer.dispatcher = NCCLAllToAllDispatcher(module.num_experts, local_only=True)
        layer.train(module.training)
        return layer

    def forward(self, hidden_states, top_k_index, top_k_weights):
        dispatched = self.dispatcher.dispatch(hidden_states, top_k_index, top_k_weights)
        gate_up, down = self.gate_up_proj, self.down_proj
        if hidden_states.device.type == "cpu":
            backend = ReferenceExperts()
        elif hidden_states.device.type == "cuda":
            backend = GroupedMMExperts()
            # torch._grouped_mm is not registered for CUDA autocast like HF's F.linear. Cast explicitly so DDP/FSDP
            # autocast uses the same GEMM/activation precision while gradients still reach the original parameters.
            if torch.is_autocast_enabled("cuda") and hidden_states.dtype != torch.float64:
                dtype = torch.get_autocast_dtype("cuda")
                dispatched = replace(dispatched, tokens=dispatched.tokens.to(dtype))
                gate_up, down = gate_up.to(dtype), down.to(dtype)
        else:
            raise NotImplementedError("Gemma 4 shared experts currently support CPU reference and CUDA grouped GEMM")
        output = backend(dispatched, gate_up, down, self.act_fn)
        # Under autocast GEMM returns BF16/FP16 even when the residual is FP32. Native HF combines into
        # zeros_like(hidden_states), so promote before combining rather than round the combined result early.
        return self.dispatcher.combine(output.to(hidden_states.dtype), dispatched)


def replace_gemma4_experts(model):
    """Adapt a Gemma text/multimodal model in place, before constructing a DP plugin, without materializing weights."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    for parent in list(model.modules()):
        for name, module in list(parent.named_children()):
            if isinstance(module, Gemma4TextExperts):
                setattr(parent, name, Gemma4GroupedExperts.from_native_module(module))
    return model
