# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np

import torch
import torch.nn as nn

from .checkpointing import get_checkpoint_fn
from .tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
    _chunk_slice,
)


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                normal_init_(self.weight)
            elif init == "relu":
                normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out


@torch.jit.ignore
def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    s = torch.nn.functional.softmax(t, dim=dim)

    return s


#@torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, Q, C_hidden]
    query = permute_final_dims(query, (1, 0, 2))

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 2, 0))

    # [*, H, V, C_hidden]
    value = permute_final_dims(value, (1, 0, 2))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a


@torch.jit.ignore
def _attention_chunked_trainable(
    query,
    key,
    value,
    biases,
    chunk_size,
    chunk_dim,
    checkpoint,
):
    if (checkpoint and len(biases) > 2):
        raise ValueError("Checkpointed version permits only permits two bias terms")

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        return _attention(q, k, v, bs)

    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = (slice(start, end) if b.shape[chunk_dim] != 1 else slice(None))
            return b[tuple(idx)]

        if (checkpoint):
            bias_1_chunk, bias_2_chunk = [
                _slice_bias(b) if b is not None else None for b in (biases + [None, None])[:2]
            ]

            o_chunk = checkpoint_fn(_checkpointable_attention, q_chunk, k_chunk, v_chunk,
                                    bias_1_chunk, bias_2_chunk)
        else:
            bias_chunks = [_slice_bias(b) for b in biases]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)

        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init="final")

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: torch.Tensor,
                  kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_lma: bool = False,
        q_chunk_size: Optional[int] = None,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_lma:
                Whether to use low-memory attention
            q_chunk_size:
                Query chunk size (for LMA)
            kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if (biases is None):
            biases = []
        if (use_lma and (q_chunk_size is None or kv_chunk_size is None)):
            raise ValueError("If use_lma is specified, q_chunk_size and kv_chunk_size must "
                             "be provided")

        q, k, v = self._prep_qkv(q_x, kv_x)

        if (use_lma):
            biases = [b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) for b in biases]

            o = _lma(q, k, v, biases, q_chunk_size, kv_chunk_size)
        else:
            o = _attention(q, k, v, biases)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):

    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(c_in, c_hidden * no_heads, bias=False, init="glorot")

        self.linear_k = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_v = Linear(
            c_in,
            c_hidden,
            bias=False,
            init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1),
                      dim=-2) / (torch.sum(mask, dim=-1)[..., None] + self.eps)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.c_hidden**(-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        # [*, N_res, H, N_seq]
        a = torch.matmul(
            q,
            k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
        )
        bias = (self.inf * (mask - 1))[..., :, None, :]
        a += bias
        a = softmax(a)

        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a,
            v,
        )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


def _lma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    q_chunk_size: int,
    kv_chunk_size: int,
):
    no_q, no_kv = q.shape[-3], k.shape[-3]

    # [*, Q, H, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s:q_s + q_chunk_size, :, :]
        large_bias_chunks = [b[..., q_s:q_s + q_chunk_size, :] for b in biases]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s:kv_s + kv_chunk_size, :, :]
            v_chunk = v[..., kv_s:kv_s + kv_chunk_size, :, :]
            small_bias_chunks = [b[..., kv_s:kv_s + kv_chunk_size] for b in large_bias_chunks]

            a = torch.einsum(
                "...qhd,...khd->...hqk",
                q_chunk,
                k_chunk,
            )

            for b in small_bias_chunks:
                a += b

            a = a.transpose(-2, -3)

            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...vhf,...qhv->...qhf", v_chunk, exp_a)

            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= max_diffs.unsqueeze(-1)
        chunk_weights *= max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s:q_s + q_chunk_size, :, :] = q_chunk_out

    return o
