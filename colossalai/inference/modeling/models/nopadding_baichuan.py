# This code is adapted from huggingface baichuan model: hhttps://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/modeling_baichuan.py
from typing import Optional, Tuple

import torch
import torch.nn as nn

from colossalai.inference.flash_decoding_utils import FDIntermTensors
from colossalai.inference.modeling.models.nopadding_llama import NopadLlamaAttention
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.logging import get_dist_logger

inference_ops = InferenceOpsLoader().load()

logger = get_dist_logger(__name__)


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

        # Used to adapt llama_base_attn_forward
        self.num_key_value_heads = self.num_heads

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

        return NopadLlamaAttention.forward(
            self,
            hidden_states=hidden_states,
            block_tables=block_tables,
            k_cache=k_cache,
            v_cache=v_cache,
            sequence_lengths=sequence_lengths,
            cos_sin=cos_sin,
            fd_inter_tensor=fd_inter_tensor,
            is_prompts=is_prompts,
            is_verifier=is_verifier,
            tokens_to_verify=tokens_to_verify,
            kv_seq_len=kv_seq_len,
            output_tensor=output_tensor,
            sm_scale=sm_scale,
            use_cuda_kernel=use_cuda_kernel,
            cu_seqlens=cu_seqlens,
            high_precision=high_precision,
        )


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
