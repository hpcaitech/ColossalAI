import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import all_to_all_comm
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


class SequenceParallelAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        heads_num: torch.Tensor,
        hidden_dim: torch.Tensor,
        enable_sequence_parallellism: bool = False,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super(SequenceParallelAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        assert hidden_dim % heads_num == 0
        self.head_dim = hidden_dim // heads_num
        self.enable_sequence_parallellism = enable_sequence_parallellism

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def attn(self, q, k, v):
        batch_size, seq_len = q.shape[0], q.shape[1]

        scale = self.head_dim**0.5
        qk = torch.matmul(q, k.transpose(-2, -1)) / scale
        weights = F.softmax(qk, dim=-1)

        attention_score = torch.matmul(weights, v)

        return attention_score

    def forward(self, x) -> Tensor:
        bsz, q_len, _ = x.size()

        seq_len = q_len * dist.get_world_size(self.spg) if self.enable_sequence_parallellism else q_len
        num_heads = (
            self.heads_num // dist.get_world_size(self.spg) if self.enable_sequence_parallellism else self.heads_num
        )

        # in shape : e.g.,  [s/p:h:]
        query_states = self.q(x)
        key_states = self.k(x)
        value_states = self.v(x)

        if self.enable_sequence_parallellism:
            query_states = all_to_all_comm(query_states, self.spg, self.scatter_idx, self.gather_idx)
            key_states = all_to_all_comm(key_states, self.spg, self.scatter_idx, self.gather_idx)
            value_states = all_to_all_comm(value_states, self.spg, self.scatter_idx, self.gather_idx)

        query_states = query_states.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
        # out shape : e.g., [s:h/p:]
        attn_score = self.attn(query_states, key_states, value_states)
        attn_score = attn_score.transpose(1, 2).contiguous()
        attn_score = attn_score.reshape(bsz, seq_len, num_heads * self.head_dim)
        if self.enable_sequence_parallellism:
            attn_score = all_to_all_comm(attn_score, self.spg, self.gather_idx, self.scatter_idx)

        # output e.g., [s/p::h]
        output = self.out(attn_score)

        return output


def seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_input = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_input.requires_grad_(True)

    # Multi-head Attention
    mha = SequenceParallelAttention(head_num, hidden_dim).cuda()
    # Multi-head Attention forward
    mha_out = mha(x_unshard)

    # Sequence parallel Attention
    sp_attn = SequenceParallelAttention(head_num, hidden_dim, True).cuda()
    sp_attn.load_state_dict(copy.deepcopy(mha.state_dict()))
    # Sequence parallel Attention forward
    dist_attn_out = sp_attn(x_input)

    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(dist_attn_out) for _ in range(world_size)]
    dist.all_gather(out_list, dist_attn_out)
    seq_out = torch.cat(out_list, dim=1)

    # forward result check
    assert_close(seq_out, mha_out)

    # Multi-head Attention backward
    mha_out.sum().backward()
    q_grad = mha.q.weight.grad
    k_grad = mha.k.weight.grad
    v_grad = mha.v.weight.grad
    o_grad = mha.out.weight.grad
    x_grad = x_unshard.grad

    # Sequence parallel Attention backward
    dist_attn_out.sum().backward()
    q_grad_seq = sp_attn.q.weight.grad
    k_grad_seq = sp_attn.k.weight.grad
    v_grad_seq = sp_attn.v.weight.grad
    o_grad_seq = sp_attn.out.weight.grad
    x_grad_seq = x_input.grad
    # all_reduce the grad of sequence parallel attention weight
    dist.all_reduce(q_grad_seq)
    dist.all_reduce(k_grad_seq)
    dist.all_reduce(v_grad_seq)
    dist.all_reduce(o_grad_seq)
    # gather the grad of sequence parallel attention input
    x_grad_seq_list = [torch.empty_like(x_grad_seq) for _ in range(world_size)]
    dist.all_gather(x_grad_seq_list, x_grad_seq)
    x_grad_seq_gather = torch.cat(x_grad_seq_list, dim=1)

    # backward result check
    assert_close(q_grad_seq, q_grad)
    assert_close(k_grad_seq, k_grad)
    assert_close(v_grad_seq, v_grad, atol=1e-4, rtol=1e-4)
    assert_close(o_grad_seq, o_grad)
    assert_close(x_grad_seq_gather, x_grad)


@parameterize("seq_len", [128])
@parameterize("hidden_dim", [64])
@parameterize("head_num", [4])
@parameterize("batch_size", [1])
def run_seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_attn()


@rerun_if_address_is_in_use()
def test_all_to_all_attention():
    spawn(check_all2all_attn, nprocs=4)


if __name__ == "__main__":
    test_all_to_all_attention()
