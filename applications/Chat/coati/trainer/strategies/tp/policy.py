from functools import partial
from types import MethodType
from typing import Dict, Type

import torch.nn as nn
from coati.models.bloom.triton_attention_forward import TritonBloomAttention
from coati.models.lora import LoraLinear
from torch.nn import Module
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomForCausalLM, BloomMLP
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder, OPTDecoderLayer

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.lazy import LazyTensor
from colossalai.nn.layer.utils import divide

from .parallel import (
    linear_1d_col_fn,
    linear_1d_row_fn,
    lora_linear_1d_col_fn,
    lora_linear_1d_row_fn,
    vocab_parallel_embedding_fn,
    vocab_parallel_lm_head_fn,
)


class Policy:

    def replace(self, module: nn.Module) -> bool:
        """Modfiy the module in place

        Args:
            module (nn.Module): Module to be modified

        Returns:
            bool: Whether to recurse into the module's children
        """
        pass


class Linear1DColPolicy(Policy):

    def __init__(self, gather_output: bool = False) -> None:
        super().__init__()
        self.gather_output = gather_output
        self.tp_size = gpc.tensor_parallel_size
        self.tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    def replace(self, module: nn.Module) -> bool:
        assert isinstance(module, (nn.Linear, LoraLinear))
        # shard params
        # TODO(ver217): this should be done via DTensor
        divide(module.out_features, self.tp_size)
        if isinstance(module.weight, LazyTensor):
            module.weight.materialize()
        module.weight.data = module.weight.chunk(self.tp_size, dim=0)[self.tp_rank].data.clone()
        if module.bias is not None:
            if isinstance(module.bias, LazyTensor):
                module.bias.materialize()
            module.bias.data = module.bias.chunk(self.tp_size, dim=0)[self.tp_rank].data.clone()
        if isinstance(module, LoraLinear):
            if isinstance(module.lora_B, LazyTensor):
                module.lora_B.materialize()
            module.lora_B.data = module.lora_B.chunk(self.tp_size, dim=0)[self.tp_rank].data.clone()
        # replace forward
        fwd = lora_linear_1d_col_fn if isinstance(module, LoraLinear) else linear_1d_col_fn
        module.forward = MethodType(partial(fwd, gather_output=self.gather_output), module)
        return False


class Linear1DRowPolicy(Policy):

    def __init__(self, parallel_input: bool = True) -> None:
        super().__init__()
        self.parallel_input = parallel_input
        self.tp_size = gpc.tensor_parallel_size
        self.tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    def replace(self, module: nn.Module) -> bool:
        assert isinstance(module, (nn.Linear, LoraLinear))
        divide(module.in_features, gpc.tensor_parallel_size)
        if isinstance(module.weight, LazyTensor):
            module.weight.materialize()
        module.weight.data = module.weight.chunk(self.tp_size, dim=1)[self.tp_rank].data.clone()
        if module.bias is not None and isinstance(module.bias, LazyTensor):
            module.bias.materialize()
        if isinstance(module, LoraLinear):
            if isinstance(module.lora_A, LazyTensor):
                module.lora_A.materialize()
            module.lora_A.data = module.lora_A.chunk(self.tp_size, dim=1)[self.tp_rank].data.clone()
        fwd = lora_linear_1d_row_fn if isinstance(module, LoraLinear) else linear_1d_row_fn
        module.forward = MethodType(partial(fwd, parallel_input=self.parallel_input), module)
        return False


class Embedding1DPolicy(Policy):

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = gpc.tensor_parallel_size
        self.tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)

    def replace(self, module: Module) -> bool:
        assert isinstance(module, nn.Embedding)
        divide(module.num_embeddings, self.tp_size)
        if isinstance(module.weight, LazyTensor):
            module.weight.materialize()
        module.weight.data = module.weight.chunk(self.tp_size, dim=0)[self.tp_rank].data.clone()
        module.forward = MethodType(vocab_parallel_embedding_fn, module)
        return False


def bloom_attn_fwd(module: BloomAttention, *args, alibi=None, **kwargs):
    if alibi is not None:
        tp_size = gpc.tensor_parallel_size
        tp_rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        alibi = alibi.chunk(tp_size, dim=0)[tp_rank]
    if module.training:
        return BloomAttention.forward(module, *args, alibi=alibi, **kwargs)
    else:
        return TritonBloomAttention.forward(module, *args, alibi=alibi, **kwargs)


class BloomAttentionPolicy(Policy):

    def replace(self, module: Module) -> bool:
        assert isinstance(module, BloomAttention)
        assert module.num_heads % gpc.tensor_parallel_size == 0
        assert module.hidden_size % gpc.tensor_parallel_size == 0
        module.num_heads = module.num_heads // gpc.tensor_parallel_size
        module.hidden_size = module.hidden_size // gpc.tensor_parallel_size
        col_policy = Linear1DColPolicy(gather_output=False)
        row_policy = Linear1DRowPolicy(parallel_input=True)
        col_policy.replace(module.query_key_value)
        row_policy.replace(module.dense)
        module.forward = MethodType(bloom_attn_fwd, module)
        return False


class BloomMLPPolicy(Policy):

    def replace(self, module: Module) -> bool:
        assert isinstance(module, BloomMLP)
        col_policy = Linear1DColPolicy(gather_output=False)
        row_policy = Linear1DRowPolicy(parallel_input=True)
        col_policy.replace(module.dense_h_to_4h)
        row_policy.replace(module.dense_4h_to_h)
        return False


class BloomForCausalLMPolicy(Policy):

    def replace(self, module: Module) -> bool:
        assert isinstance(module, BloomForCausalLM)
        module.lm_head.gather_output = True
        module.lm_head.forward = MethodType(vocab_parallel_lm_head_fn, module.lm_head)
        return True


class OPTAttentionPolicy(Policy):

    def replace(self, module: nn.Module) -> bool:
        assert isinstance(module, OPTAttention)
        # reset attr
        assert module.num_heads % gpc.tensor_parallel_size == 0
        assert module.embed_dim % gpc.tensor_parallel_size == 0
        module.num_heads = module.num_heads // gpc.tensor_parallel_size
        module.embed_dim = module.embed_dim // gpc.tensor_parallel_size
        col_policy = Linear1DColPolicy(gather_output=False)
        row_policy = Linear1DRowPolicy(parallel_input=True)
        for layer in (module.k_proj, module.v_proj, module.q_proj):
            col_policy.replace(layer)
        row_policy.replace(module.out_proj)
        return False


class OPTDecoderLayerPolicy(Policy):

    def replace(self, module: Module) -> bool:
        assert isinstance(module, OPTDecoderLayer)
        attn_policy = OPTAttentionPolicy()
        attn_policy.replace(module.self_attn)
        col_policy = Linear1DColPolicy(gather_output=False)
        row_policy = Linear1DRowPolicy(parallel_input=True)
        col_policy.replace(module.fc1)
        row_policy.replace(module.fc2)
        return False


class OPTDecoderPolicy(Policy):

    def replace(self, module: Module) -> bool:
        assert isinstance(module, OPTDecoder)
        col_policy = Linear1DColPolicy(gather_output=True)
        if module.project_in is not None:
            col_policy.replace(module.project_in)
        if module.project_out is not None:
            col_policy.replace(module.project_out)
        decoder_layer_policy = OPTDecoderLayerPolicy()
        for decoder_layer in module.layers:
            decoder_layer_policy.replace(decoder_layer)
        return False


POLICY_MAP: Dict[Type[nn.Module], Type[Policy]] = {
    OPTDecoder: OPTDecoderPolicy,
    BloomForCausalLM: BloomForCausalLMPolicy,
    BloomMLP: BloomMLPPolicy,
    BloomAttention: BloomAttentionPolicy,
    nn.Embedding: Embedding1DPolicy,
}
