# This code is adapted from huggingface baichuan model: hhttps://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/modeling_baichuan.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

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

logger = get_dist_logger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    use_flash_attn2 = True
except ImportError:
    use_flash_attn2 = False
    logger.warning(f"flash_attn2 has not been installed yet, we will use triton flash attn instead.")

inference_ops = InferenceOpsLoader().load()

logger = get_dist_logger(__name__)


# alibi slopes calculation adapted from https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/bloom/modeling_bloom.py#L57
def get_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32, device=device)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32, device=device
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32, device=device)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


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
    else:
        TypeError(
            "Currently, the variable name for the epsilon of baichuan7B/13B should be 'variance_epsilon' or 'epsilon'."
        )

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


class NopadBaichuanAttention(nn.Module):
    def __init__(
        self,
        config,
        attn_qproj_w: torch.Tensor = None,
        attn_kproj_w: torch.Tensor = None,
        attn_vproj_w: torch.Tensor = None,
        attn_oproj_w: torch.Tensor = None,
    ):
        """This layer will replace the BaichuanAttention.

        Args:
            config (BaichuanConfig): Holding the Baichuan model config.
            attn_qproj_w (torch.Tensor, optional): The transposed q_proj weight. Defaults to None.
            attn_kproj_w (torch.Tensor, optional): The transposed k_proj weight. Defaults to None.
            attn_vproj_w (torch.Tensor, optional): The transposed v_proj weight. Defaults to None.
            attn_oproj_w (torch.Tensor, optional): The transposed o_proj weight. Defaults to None.
        """
        super().__init__()
        self.o_proj_weight = attn_oproj_w

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.alibi_slopes = None
        self.use_alibi_attn = False
        if self.hidden_size == 5120:
            self.use_alibi_attn = True
            self.alibi_slopes = get_alibi_slopes(self.num_heads, device=attn_qproj_w.device)

        qkv_weight_list = [attn_qproj_w, attn_kproj_w, attn_vproj_w]
        self.qkv_weight = torch.stack(qkv_weight_list, dim=0)

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> "NopadBaichuanAttention":
        """Used for initialize the weight of NopadBaichuanAttention by origin BaichuanAttention.

        Args:
            module (nn.Module): The origin BaichuanAttention layer.
        """

        config = module.config

        q_proj_w, k_proj_w, v_proj_w = module.W_pack.weight.view((3, module.hidden_size, module.hidden_size))

        attn_qproj_w = q_proj_w.transpose(0, 1)
        attn_kproj_w = k_proj_w.transpose(0, 1)
        attn_vproj_w = v_proj_w.transpose(0, 1)
        attn_oproj_w = module.o_proj.weight.transpose(0, 1)

        attn_layer = NopadBaichuanAttention(
            config=config,
            attn_qproj_w=attn_qproj_w,
            attn_kproj_w=attn_kproj_w,
            attn_vproj_w=attn_vproj_w,
            attn_oproj_w=attn_oproj_w,
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

        block_size = k_cache.size(-2)

        if is_prompts:
            if (
                not is_verifier
                and use_cuda_kernel
                and query_states.dtype != torch.float32
                and use_flash_attn2
                and not self.use_alibi_attn
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
                if not self.use_alibi_attn:
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
                    alibi_slopes=self.alibi_slopes,
                    max_seq_len=kv_seq_len,
                    sm_scale=sm_scale,
                )
        else:
            q_len = tokens_to_verify + 1 if is_verifier else 1

            if use_cuda_kernel:
                if not self.use_alibi_attn:
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
                if not is_verifier and not self.use_alibi_attn:
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
                    if not self.use_alibi_attn:
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
                alibi_slopes=self.alibi_slopes,
                sm_scale=sm_scale,
                q_len=q_len,
            )

        attn_output = attn_output.view(-1, self.hidden_size)
        attn_output = torch.mm(attn_output, self.o_proj_weight)

        return attn_output


# NOTE This will cause difference as out length increases.
class NopadBaichuanMLP(nn.Module):
    def __init__(
        self,
        mlp_gproj_w: torch.Tensor = None,
        mlp_uproj_w: torch.Tensor = None,
        mlp_dproj_w: torch.Tensor = None,
    ):
        """This layer will replace the BaichuanAttention.

        Args:
            mlp_gproj_w (torch.Tensor, optional): The transposed gate_proj weight. Defaults to None.
            mlp_uproj_w (torch.Tensor, optional): The transposed up_proj weight. Defaults to None.
            mlp_dproj_w (torch.Tensor, optional): The transposed down_proj weight. Defaults to None.
        """
        super().__init__()
        self.gate_up_weight = torch.stack([mlp_gproj_w, mlp_uproj_w], dim=0)
        self.down_proj_weight = mlp_dproj_w

    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        """Used for initialize the weight of NopadBaichuanMLP by origin MLP(Baichuan).

        Args:
            module (nn.Module): The origin MLP(Baichuan) layer.
        """

        mlp_gproj_w = module.gate_proj.weight.transpose(0, 1)
        mlp_uproj_w = module.up_proj.weight.transpose(0, 1)
        mlp_dproj_w = module.down_proj.weight.transpose(0, 1)

        mlp_layer = NopadBaichuanMLP(
            mlp_gproj_w=mlp_gproj_w,
            mlp_uproj_w=mlp_uproj_w,
            mlp_dproj_w=mlp_dproj_w,
        )

        return mlp_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input to the layer of shape [token_num, embed_dim].
        """
        hidden_states = hidden_states.expand(2, -1, -1)
        gate_up_proj_out = torch.bmm(hidden_states, self.gate_up_weight)
        act_out = inference_ops.silu_and_mul(gate_up_proj_out)
        return torch.mm(act_out, self.down_proj_weight)
