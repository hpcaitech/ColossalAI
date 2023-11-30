# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from einops import rearrange

HAS_NPU_TRIANGLE_ATTENTION = False
try:
    from torch_npu import npu_confusion_transpose, npu_scaled_masked_softmax

    HAS_NPU_TRIANGLE_ATTENTION = True
except ImportError:
    logging.warning("Import torch_npu Error.")


if HAS_NPU_TRIANGLE_ATTENTION:

    def npu_triangle_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        origin_attn_mask: torch.Tensor = None,
        scale: float = 1.0,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        block_size=512,
    ):
        """
        The triangle attention reduces the attention calculation of the mask
        part by dividing the q, k, and v matrices into blocks

        Arguments:
            block_size: The size of the inverted triangle block, the default is 512,
                        the smaller the block_size, the more calculations will be reduced,
                        but the number of small operators will be increased
            masked_softmax_func: mask function to be applied.
            dropout_func: dropout function to be applied.
        """

        def compute_attn(q_layer, k_layer, v_layer, mask_tmp):
            # [b, hn, q_size, hd] * [b, hn, hd, kv_size] -> [b, hn, q_size, kv_size]
            cur_sim = torch.matmul(q_layer, k_layer)
            attention_probs = npu_scaled_masked_softmax(cur_sim, mask_tmp)
            # attention dropout
            if dropout_p > 0:
                attention_probs = torch.nn.functional.dropout(
                    attention_probs, p=dropout_p, training=attention_probs.require_grad
                )
            # [b, hn, q_size, kv_size] * [b, hn, kv_size, hd] -> [b, hn, q_size, hd]
            context_layer_tmp = torch.matmul(attention_probs, v_layer)
            return context_layer_tmp

        q, k, v = [rearrange(x, "b s h d -> b h s d") for x in (q, k, v)]
        origin_attn_mask = origin_attn_mask.to(torch.bool)
        #  input shape: [b, hn, sq, hd]
        bsz, head_num, sequence_len, head_dim = k.shape
        sparse_groups = sequence_len // block_size
        # Determine whether blocks size can be divided by sequence_length
        divisible_flag = sequence_len == block_size * sparse_groups
        k = k.transpose(2, 3).contiguous()
        if divisible_flag:
            q_tmp_layers = torch.chunk(q, sparse_groups, 2)
            k_tmp_layers = torch.chunk(k, sparse_groups, 3)
            v_tmp_layers = torch.chunk(v, sparse_groups, 2)
        else:
            seq_tmp = block_size * sparse_groups
            q_last = q[:, :, seq_tmp:, :].contiguous()
            mask_last = origin_attn_mask[:, :, seq_tmp:, :].contiguous()
            q_tmp_layers = torch.chunk(q[:, :, :seq_tmp, :], sparse_groups, 2)
            k_tmp_layers = torch.chunk(k[:, :, :, :seq_tmp], sparse_groups, 3)
            v_tmp_layers = torch.chunk(v[:, :, :seq_tmp, :], sparse_groups, 2)
        context_list_tmp, k_tmp, v_tmp = [], (), ()
        for i in range(sparse_groups):
            # compute slice shape of q k v for each loop
            q_begin, q_end = i * block_size, (i + 1) * block_size
            kv_begin, kv_end = 0, (i + 1) * block_size
            q_tmp = q_tmp_layers[i]
            # slice k and v
            if i == 0:
                k_tmp = k_tmp_layers[i].contiguous()
                v_tmp = v_tmp_layers[i].contiguous()
            else:
                k_tmp = torch.cat((k_tmp, k_tmp_layers[i]), -1).contiguous()
                v_tmp = torch.cat((v_tmp, v_tmp_layers[i]), -2).contiguous()

            mask_tmp = origin_attn_mask[:, :, q_begin:q_end, kv_begin:kv_end].contiguous()
            context_layer_tmp = compute_attn(q_tmp, k_tmp, v_tmp, mask_tmp)
            context_list_tmp.append(context_layer_tmp)

        if not divisible_flag:
            # circumstances that cannot be divisible
            context_layer_tmp = compute_attn(q_last, k, v, mask_last)
            context_list_tmp.append(context_layer_tmp)
        context_layer = torch.cat(context_list_tmp, 2)
        new_context_layer_shape = (bsz, sequence_len, head_num * head_dim)
        context_layer = npu_confusion_transpose(context_layer, [0, 2, 1, 3], [*new_context_layer_shape], True)
        # =========================
        # Context layer. [b, sq, hp]
        # =========================
        return context_layer
