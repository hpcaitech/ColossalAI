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
import torch.nn as nn

HAS_NPU_TRIANGLE_ATTENTION = False
try:
    from torch_npu import npu_scaled_masked_softmax
    from torch_npu import npu_confusion_transpose
    HAS_NPU_TRIANGLE_ATTENTION = True
except ImportError:
    logging.warning("Import torch_npu Error.")


class TriangleAttention(nn.Module):
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

    def __init__(self, block_size=512, masked_softmax_func=None, dropout_func=None):
        super(TriangleAttention, self).__init__()
        self.block_size = block_size
        self.mask_tmp_initialed = False
        self.mask_tmp_groups = []
        if masked_softmax_func is not None:
            self.scaled_masked_softmax = masked_softmax_func
        else:
            self.scaled_masked_softmax = npu_scaled_masked_softmax
        if dropout_func:
            self.dropout = True
            self.attn_dropout = dropout_func
        else:
            self.dropout = False
            
    def compute_attn(self, q_layer, k_layer, v_layer, mask_tmp):
        # [b, hn, q_size, hd] * [b, hn, hd, kv_size] -> [b, hn, q_size, kv_size]
        cur_sim = torch.matmul(q_layer, k_layer)

        attention_probs = self.scaled_masked_softmax(cur_sim, mask_tmp)

        # attention dropout
        if self.dropout:
            attention_probs = self.attn_dropout(attention_probs)

        # [b, hn, q_size, kv_size] * [b, hn, kv_size, hd] -> [b, hn, q_size, hd]
        context_layer_tmp = torch.matmul(attention_probs, v_layer)
        return context_layer_tmp

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        #  input shape: [b, hn, sq, hd]
        bsz, head_num, sequence_len, head_dim = key_layer.shape
        sparse_groups = sequence_len // self.block_size
        # Determine whether blocks size can be divided by sequence_length
        flag = sequence_len == self.block_size * sparse_groups
        key_layer = key_layer.transpose(2, 3).contiguous()
        if flag:
            q_tmp_layers = torch.chunk(query_layer, sparse_groups, 2)
            k_tmp_layers = torch.chunk(key_layer, sparse_groups, 3)
            v_tmp_layers = torch.chunk(value_layer, sparse_groups, 2)
        else:
            seq_tmp = self.block_size * sparse_groups
            q_last = query_layer[:, :, seq_tmp:, :].contiguous()
            mask_last = attention_mask[:, :, seq_tmp:, :].contiguous()
            q_tmp_layers = torch.chunk(query_layer[:, :, :seq_tmp, :], sparse_groups, 2)
            k_tmp_layers = torch.chunk(key_layer[:, :, :, :seq_tmp], sparse_groups, 3)
            v_tmp_layers = torch.chunk(value_layer[:, :, :seq_tmp, :], sparse_groups, 2)
        context_list_tmp, k_tmp, v_tmp = [], (), ()
        for i in range(sparse_groups):
            # compute slice shape of q k v for each loop
            q_begin, q_end = i * self.block_size, (i + 1) * self.block_size 
            kv_begin, kv_end = 0, (i + 1) * self.block_size
            q_tmp = q_tmp_layers[i]
            # slice k and v
            if i == 0:
                k_tmp = k_tmp_layers[i].contiguous()
                v_tmp = v_tmp_layers[i].contiguous()
            else:
                k_tmp = torch.cat((k_tmp, k_tmp_layers[i]), -1).contiguous()
                v_tmp = torch.cat((v_tmp, v_tmp_layers[i]), -2).contiguous()

            if not self.mask_tmp_initialed:
                mask_tmp = attention_mask[:, :, q_begin:q_end, kv_begin:kv_end]
                self.mask_tmp_groups.append(mask_tmp.contiguous())
            else:
                mask_tmp = self.mask_tmp_groups[i]

            context_layer_tmp = self.compute_attn(q_tmp, k_tmp, v_tmp, mask_tmp)
            context_list_tmp.append(context_layer_tmp)

        if not flag:
            # circumstances that cannot be divisible
            context_layer_tmp = self.compute_attn(q_last, key_layer, value_layer, mask_last)
            context_list_tmp.append(context_layer_tmp)
        context_layer = torch.cat(context_list_tmp, 2)
        self.mask_tmp_initialed = True
        new_context_layer_shape = (bsz, sequence_len, head_num * head_dim)
        context_layer = npu_confusion_transpose(context_layer, [0, 2, 1, 3], [*new_context_layer_shape], True)
        # =========================
        # Context layer. [b, sq, hp]
        # =========================
        return context_layer


if __name__ == "__main__":
    attn = TriangleAttention()
    q, k, v = [torch.randn((2, 12, 1024, 64), requires_grad=True).npu().half() for _ in range(3)]
    mask = torch.ones(2, 12, 1024, 1024).npu().bool()
    out = attn(q, k, v, mask)
    loss = out.sum()
    loss.backward()

