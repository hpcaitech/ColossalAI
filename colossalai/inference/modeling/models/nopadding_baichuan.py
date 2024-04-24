# This code is adapted from huggingface baichuan model: hhttps://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/modeling_baichuan.py
import itertools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import (
    context_attention_unpadded,
    copy_k_to_blocked_cache,
    decoding_fused_rotary_embedding,
    flash_decoding_attention,
    rms_layernorm,
    rotary_embedding,
)
from colossalai.logging import get_dist_logger
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.tensor.d_tensor import distribute_tensor, is_distributed_tensor

logger = get_dist_logger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    use_flash_attn2 = True
except ImportError:
    use_flash_attn2 = False
    logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")

inference_ops = InferenceOpsLoader().load()

logger = get_dist_logger(__name__)


def baichuan_rmsnorm_forward(
    self,
    hidden_states: torch.Tensor,
    norm_output: torch.Tensor,
    residual: torch.Tensor = None,
    use_cuda_kernel: bool = True,
):
    # Used to address the issue of inconsistent epsilon variable names in baichuan2 7b and 13b.
    if hasattr(self, "variance_epsilon"):
        eps = self.variance_epsilon
    elif hasattr(self, "epsilon"):
        eps = self.epsilon

    # TODO "Currently, CUDA RMS Norm does not support the hidden_size of 5120. Remove this code after CUDA RMS Norm is fixed."
    use_cuda_kernel = False

    if use_cuda_kernel:
        if residual is not None:
            inference_ops.fused_add_rms_layernorm(hidden_states, residual, self.weight.data, eps)
            return hidden_states, residual

        if norm_output is None:
            norm_output = torch.empty_like(hidden_states)
        inference_ops.rms_layernorm(norm_output, hidden_states, self.weight.data, eps)
        return norm_output, hidden_states
    else:
        return rms_layernorm(hidden_states, self.weight.data, eps, norm_output, residual)


class NopadBaichuanAttention(ParallelModule, nn.Module):
    def __init__(
        self,
        config,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj: ParallelModule = None,
        num_heads: int = None,
        hidden_size: int = None,
        process_group: ProcessGroup = None,
    ):
        """This layer will replace the BaichuanAttention.

        Args:
            config (BaichuanConfig): Holding the Baichuan model config.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj (Linear1D_Row, optional): The Linear1D_Row o_proj weight. Defaults to None.
        """
        ParallelModule.__init__(self)
        self.o_proj = attn_oproj

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.process_group = process_group

        qkv_weight_list = [attn_qproj_w.transpose(0, 1), attn_kproj_w.transpose(0, 1), attn_vproj_w.transpose(0, 1)]
        self.qkv_weight = nn.Parameter(torch.stack(qkv_weight_list, dim=0))

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> "NopadBaichuanAttention":
        """Used for initialize the weight of NopadBaichuanAttention by origin BaichuanAttention.

        Args:
            module (nn.Module): The origin BaichuanAttention layer.
        """

        config = module.config

        q_proj_w, k_proj_w, v_proj_w = module.W_pack.weight.view((3, module.hidden_size, -1))

        attn_qproj_w = q_proj_w
        attn_kproj_w = k_proj_w
        attn_vproj_w = v_proj_w
        attn_oproj = module.o_proj

        attn_layer = NopadBaichuanAttention(
            config=config,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj=attn_oproj,
            num_heads=module.num_heads,
            hidden_size=module.hidden_size,
            process_group=process_group,
        )

        return attn_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_tables: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        sequence_lengths: torch.Tensor,
        cos_sin: Tuple[torch.Tensor],
        fd_inter_tensor: FDIntermTensors,
        is_prompts: bool = True,
        is_verifier: bool = False,
        tokens_to_verify: int = None,
        kv_seq_len: int = 0,
        output_tensor: torch.Tensor = None,
        sm_scale: int = None,
        use_cuda_kernel: bool = True,
        cu_seqlens: torch.Tensor = None,
        high_precision: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
            block_tables (torch.Tensor): A 2D tensor of shape [batch_size, max_blocks_per_sequence],
                storing mapping of token_position_id -> block_id.
            k_cache (torch.Tensor): It holds the GPU memory for the key cache.
            v_cache (torch.Tensor): It holds the GPU memory for the key cache.
            sequence_lengths (torch.Tensor, optional): Holding the sequence length of each sequence.
            cos_sin (Tuple[torch.Tensor], optional): Holding cos and sin.
            fd_inter_tensor (FDIntermTensors, optional): Holding tensors used for
                storing intermediate values in flash-decoding.
            is_prompts (bool, optional): Whether the current inference process is in the context input phase. Defaults to True.
            kv_seq_len (int, optional): The max sequence length of input sequences. Defaults to 0.
            output_tensor (torch.Tensor, optional): The mid tensor holds the output of attention. Defaults to None.
            sm_scale (int, optional): Used for flash attention. Defaults to None.
            use_cuda_kernel: (bool, optional): Whether to use cuda kernel. Defaults to True.
            cu_seqlens(torch.Tensor, optional): Holding the cumulative sum of sequence length.
            high_precision(Optional[bool]): Whether to use float32 for underlying calculations of float16 data to achieve higher precision, defaults to False.
        """

        token_nums = hidden_states.size(0)
        # fused qkv
        hidden_states = hidden_states.expand(3, -1, -1)
        query_states, key_states, value_states = (
            torch.bmm(hidden_states, self.qkv_weight).view(3, token_nums, self.num_heads, self.head_dim).unbind(0)
        )

        use_alibi_attn = False

        if fd_inter_tensor.alibi_slopes is not None:
            use_alibi_attn = True

        block_size = k_cache.size(-2)

        if is_prompts:
            if (
                not is_verifier
                and use_cuda_kernel
                and query_states.dtype != torch.float32
                and use_flash_attn2
                and not use_alibi_attn
            ):
                # flash attn 2 currently only supports FP16/BF16.
                inference_ops.rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1], high_precision)
                inference_ops.context_kv_cache_memcpy(
                    key_states, value_states, k_cache, v_cache, sequence_lengths, cu_seqlens, block_tables, kv_seq_len
                )

                attn_output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=kv_seq_len,
                    max_seqlen_k=kv_seq_len,
                    dropout_p=0.0,
                    softmax_scale=sm_scale,
                    causal=True,
                )
                attn_output = attn_output.view(token_nums, -1)
            else:
                if not use_alibi_attn:
                    rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
                attn_output = context_attention_unpadded(
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    context_lengths=sequence_lengths,
                    block_tables=block_tables,
                    block_size=block_size,
                    output=output_tensor,
                    alibi_slopes=fd_inter_tensor.alibi_slopes,
                    max_seq_len=kv_seq_len,
                    sm_scale=sm_scale,
                )
        else:
            q_len = tokens_to_verify + 1 if is_verifier else 1

            if use_cuda_kernel:
                if not use_alibi_attn:
                    inference_ops.rotary_embedding_and_cache_copy(
                        query_states,
                        key_states,
                        value_states,
                        cos_sin[0],
                        cos_sin[1],
                        k_cache,
                        v_cache,
                        sequence_lengths,
                        block_tables,
                        high_precision,
                    )
                else:
                    inference_ops.decode_kv_cache_memcpy(
                        key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables
                    )
            else:
                if not is_verifier and not use_alibi_attn:
                    decoding_fused_rotary_embedding(
                        query_states,
                        key_states,
                        value_states,
                        cos_sin[0],
                        cos_sin[1],
                        k_cache,
                        v_cache,
                        block_tables,
                        sequence_lengths,
                    )
                else:
                    if not use_alibi_attn:
                        rotary_embedding(query_states, key_states, cos_sin[0], cos_sin[1])
                    copy_k_to_blocked_cache(
                        key_states, k_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                    )
                    copy_k_to_blocked_cache(
                        value_states, v_cache, kv_lengths=sequence_lengths, block_tables=block_tables, n=q_len
                    )

            attn_output = flash_decoding_attention(
                q=query_states,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_seq_len=sequence_lengths,
                block_tables=block_tables,
                block_size=block_size,
                max_seq_len_in_batch=kv_seq_len,
                output=output_tensor,
                mid_output=fd_inter_tensor.mid_output,
                mid_output_lse=fd_inter_tensor.mid_output_lse,
                alibi_slopes=fd_inter_tensor.alibi_slopes,
                sm_scale=sm_scale,
                q_len=q_len,
            )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE This is a hack to ensure we could load the right weight from LlamaAttention checkpoint due to the use of torch.stack(q_weight, k_weight, v_weight)
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        key = "qkv_weight"
        W_pack = state_dict[prefix + "W_pack.weight"]

        device_mesh = self.helper_layout.device_mesh
        sharding_spec = self.helper_layout.sharding_spec
        W_pack = distribute_tensor(W_pack, device_mesh, sharding_spec)

        q_proj_w, k_proj_w, v_proj_w = W_pack.weight.view((3, -1, W_pack.size(-1)))

        qkv_w = torch.stack([q_proj_w.T, k_proj_w.T, v_proj_w.T], dim=0)

        input_param = nn.Parameter(
            qkv_w
        )  # NOTE qkv_weight doesn't have to be a distensor, Like input_param = sharded_tensor_to_param(input_param)

        param = local_state[key]

        try:
            with torch.no_grad():
                param.copy_(input_param)
        except Exception as ex:
            error_msgs.append(
                'While copying the parameter named "{}", '
                "whose dimensions in the model are {} and "
                "whose dimensions in the checkpoint are {}, "
                "an exception occurred : {}.".format(key, param.size(), input_param.size(), ex.args)
            )

        strict = False  # to avoid unexpected_keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self) -> str:
        return f"qkv_weight_proj MergedLinear1D_Col: in_features={self.qkv_weight.shape[1]}x3, out_features={self.qkv_weight.shape[2]}, bias=False"


# NOTE This will cause difference as out length increases.
class NopadBaichuanMLP(ParallelModule, nn.Module):
    def __init__(
        self,
        mlp_gproj_w: torch.Tensor = None,
        mlp_uproj_w: torch.Tensor = None,
        mlp_dproj: ParallelModule = None,
        process_group: ProcessGroup = None,
    ):
        """This layer will replace the BaichuanAttention.

        Args:
            mlp_gproj_w (torch.Tensor, optional): The transposed gate_proj weight. Defaults to None.
            mlp_uproj_w (torch.Tensor, optional): The transposed up_proj weight. Defaults to None.
            mlp_dproj (Linear1D_Row, optional): The Linear1D_Row mlp_dproj weight. Defaults to None.
        """
        ParallelModule.__init__(self)
        assert is_distributed_tensor(
            mlp_gproj_w
        ), "mlp_gproj_w must be dtensor so we could get the layout of the weight"
        self.helper_layout = (
            mlp_gproj_w.dist_layout
        )  # NOTE this is a hack for the right load/shard of gate_up_weight(used in _load_from_state_dict)
        self.gate_up_weight = nn.Parameter(
            torch.stack([mlp_gproj_w.transpose(0, 1), mlp_uproj_w.transpose(0, 1)], dim=0)
        )
        self.down_proj = mlp_dproj
        self.process_group = process_group

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        """Used for initialize the weight of NopadLlamaMLP by origin LlamaMLP.

        Args:
            module (LlamaMLP): The origin LlamaMLP layer.
        """
        mlp_gproj_w = module.gate_proj.weight
        assert is_distributed_tensor(
            module.gate_proj.weight
        ), "gate_proj.weight must be dtensor so we could get the layout of the weight"
        mlp_uproj_w = module.up_proj.weight
        mlp_dproj = module.down_proj

        mlp_layer = NopadBaichuanMLP(
            mlp_gproj_w=mlp_gproj_w,
            mlp_uproj_w=mlp_uproj_w,
            mlp_dproj=mlp_dproj,
            process_group=process_group,
        )

        return mlp_layer

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # NOTE This is a hack to ensure we could load the right weight from LlamaMLP checkpoint due to the use of torch.stack(gate_weight, up_weight)

        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        key = "gate_up_weight"
        k1 = "gate_proj.weight"
        k2 = "up_proj.weight"

        gate_w = state_dict[prefix + k1]
        up_w = state_dict[prefix + k2]

        device_mesh = self.helper_layout.device_mesh
        sharding_spec = self.helper_layout.sharding_spec
        gate_w = distribute_tensor(gate_w, device_mesh, sharding_spec)
        up_w = distribute_tensor(up_w, device_mesh, sharding_spec)

        gate_up_w = torch.stack([gate_w.T, up_w.T], dim=0)

        input_param = nn.Parameter(
            gate_up_w
        )  # NOTE gate_up_weight doesn't have to be a distensor, Like input_param = sharded_tensor_to_param(input_param)
        param = local_state[key]

        try:
            with torch.no_grad():
                param.copy_(input_param)
        except Exception as ex:
            error_msgs.append(
                'While copying the parameter named "{}", '
                "whose dimensions in the model are {} and "
                "whose dimensions in the checkpoint are {}, "
                "an exception occurred : {}.".format(key, param.size(), input_param.size(), ex.args)
            )

        strict = False  # to avoid unexpected_keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        """
        hidden_states = hidden_states.expand(2, -1, -1)
        gate_up_proj_out = torch.bmm(hidden_states, self.gate_up_weight)
        act_out = inference_ops.silu_and_mul(gate_up_proj_out)
        return self.down_proj(act_out)

    def extra_repr(self) -> str:
        return f"gate_up_proj MergedLinear1D_Col: in_features={self.gate_up_weight.shape[1]}x2, out_features={self.gate_up_weight.shape[2]}, bias=False"
