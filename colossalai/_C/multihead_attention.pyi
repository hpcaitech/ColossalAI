from typing import List

from torch import Tensor
from torch.distributed import ProcessGroup

def multihead_attention_fw_fp32(layer_id: int, input: Tensor, input_mask: Tensor,
                                in_proj_weight: Tensor, in_proj_bias: Tensor,
                                out_proj_weight: Tensor, out_proj_bias: Tensor,
                                norm_weight: Tensor, norm_bias: Tensor,
                                training_mode: bool, prelayernorm: bool) -> List[Tensor]:
    ...


def multihead_attention_fw_fp16(layer_id: int, input: Tensor, input_mask: Tensor,
                                in_proj_weight: Tensor, in_proj_bias: Tensor,
                                out_proj_weight: Tensor, out_proj_bias: Tensor,
                                norm_weight: Tensor, norm_bias: Tensor,
                                training_mode: bool, prelayernorm: bool) -> List[Tensor]:
    ...


def multihead_attention_bw_fp32(layer_id: int, grad_dec_output: Tensor,
                                output: Tensor, input: Tensor,
                                input_mask: Tensor, in_proj_weight: Tensor,
                                in_proj_bias: Tensor, out_proj_weight: Tensor,
                                out_proj_bias: Tensor, norm_weight: Tensor,
                                norm_bias: Tensor) -> List[Tensor]:
    ...


def multihead_attention_bw_fp16(layer_id: int, grad_dec_output: Tensor,
                                output: Tensor, input: Tensor,
                                input_mask: Tensor, in_proj_weight: Tensor,
                                in_proj_bias: Tensor, out_proj_weight: Tensor,
                                out_proj_bias: Tensor, norm_weight: Tensor,
                                norm_bias: Tensor) -> List[Tensor]:
    ...


def create_multihead_attention_fp32(layer_id: int, max_batch_tokens: int,
                                    max_seq_len: int, hidden_dim: int, num_heads: int,
                                    attn_prob_dropout_ratio: float,
                                    hidden_dropout_ratio: float,
                                    pre_or_postLayerNorm: bool,
                                    pg: ProcessGroup) -> int:
    ...


def create_multihead_attention_fp16(layer_id: int, max_batch_tokens: int,
                                    max_seq_len: int, hidden_dim: int, num_heads: int,
                                    attn_prob_dropout_ratio: float,
                                    hidden_dropout_ratio: float,
                                    pre_or_postLayerNorm: bool,
                                    pg: ProcessGroup) -> int:
    ...
