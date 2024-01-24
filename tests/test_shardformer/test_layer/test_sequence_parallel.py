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


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        heads_num,
        hidden_dim,
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        assert hidden_dim % heads_num == 0
        self.head_dim = hidden_dim // heads_num

        self.q = q_proj
        self.k = k_proj
        self.v = v_proj
        self.out = out_proj

    def attn(self, q, k, v):
        batch_size, seq_len = q.shape[0], q.shape[1]

        scale = self.head_dim**0.5
        qk = torch.matmul(q, k.transpose(-2, -1)) / scale

        # if attn_mask is not None:
        #     mask = attn_mask == 0
        #     qk[mask] = torch.tensor(float('-inf'))

        weights = F.softmax(qk, dim=-1)

        attention_score = torch.matmul(weights, v)

        return attention_score

    def forward(self, x) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # in shape : e.g.,  [s/p:h:]
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        # TODO Merge three alltoall calls into one
        query_layer = all_to_all_comm(query, self.spg, self.scatter_idx, self.gather_idx)
        key_layer = all_to_all_comm(key, self.spg, self.scatter_idx, self.gather_idx)
        value_layer = all_to_all_comm(value, self.spg, self.scatter_idx, self.gather_idx)

        # out shape : e.g., [s:h/p:]
        attn_score = self.attn(query_layer, key_layer, value_layer)

        output = all_to_all_comm(attn_score, self.spg, self.gather_idx, self.scatter_idx)

        # output e.g., [s/p::h]
        output = self.out(output)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, head_num, hidden_dim, q_proj, k_proj, v_proj, out_proj):
        super(MultiHeadAttn, self).__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        assert hidden_dim % head_num == 0
        self.head_dim = hidden_dim // head_num

        self.q = q_proj
        self.k = k_proj
        self.v = v_proj
        self.out = out_proj

    def attn(self, q, k, v):
        batch_size, seq_len = q.shape[0], q.shape[1]

        scale = self.head_dim**0.5
        qk = torch.matmul(q, k.transpose(-2, -1)) / scale

        # if attn_mask is not None:
        #     mask = attn_mask == 0
        #     qk[mask] = torch.tensor(float('-inf'))

        weights = F.softmax(qk, dim=-1)

        attention_score = torch.matmul(weights, v)

        return attention_score

    def split(self, x, batch_size, seq_len):
        res = x.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        return res

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, "hidden_dim should be equal to self.hidden_dim"
        query_mha = self.split(self.q(x), batch_size, seq_len)
        key_mha = self.split(self.k(x), batch_size, seq_len)
        value_mha = self.split(self.v(x), batch_size, seq_len)
        score_mha = self.attn(query_mha, key_mha, value_mha)
        score_mha_final = score_mha.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output_mha = self.out(score_mha_final)

        return output_mha


def seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()

    q_proj = nn.Linear(hidden_dim, hidden_dim)
    k_proj = nn.Linear(hidden_dim, hidden_dim)
    v_proj = nn.Linear(hidden_dim, hidden_dim)
    out_proj = nn.Linear(hidden_dim, hidden_dim)

    q_proj_copy = copy.deepcopy(q_proj)
    k_proj_copy = copy.deepcopy(k_proj)
    v_proj_copy = copy.deepcopy(v_proj)
    out_proj_copy = copy.deepcopy(out_proj)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_input = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_input.requires_grad_(True)

    # x_unshard = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    # x_unshard.requires_grad_(True)
    # x_input = torch.chunk(x_unshard.clone(), world_size, dim=1)[dist.get_rank()]
    # x_input.requires_grad_(True)

    # Multi-head Attention
    mhn = MultiHeadAttn(head_num, hidden_dim, q_proj, k_proj, v_proj, out_proj).cuda()
    # Multi-head Attention forward
    mhn_out = mhn(x_unshard)

    # Sequence parallel Attention
    dist_attn = DistributedAttention(head_num, hidden_dim, q_proj_copy, k_proj_copy, v_proj_copy, out_proj_copy).cuda()
    # Sequence parallel Attention forward
    dist_attn_out = dist_attn(x_input)
    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(dist_attn_out) for _ in range(world_size)]
    dist.all_gather(out_list, dist_attn_out)
    seq_out = torch.cat(out_list, dim=1)

    # forward result check
    assert_close(seq_out, mhn_out)

    # Multi-head Attention backward
    mhn_out.sum().backward()
    q_grad = mhn.q.weight.grad
    k_grad = mhn.k.weight.grad
    v_grad = mhn.v.weight.grad
    o_grad = mhn.out.weight.grad
    x_grad = x_unshard.grad

    # Sequence parallel Attention backward
    dist_attn_out.sum().backward()
    q_grad_seq = dist_attn.q.weight.grad
    k_grad_seq = dist_attn.k.weight.grad
    v_grad_seq = dist_attn.v.weight.grad
    o_grad_seq = dist_attn.out.weight.grad
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

    # print_rank('x_grad', x_grad_seq, 0)
    # print_rank('x_grad', x_grad_seq, 1)
    # print_rank('x_grad', x_grad_seq, 2)
    # print_rank('x_grad', x_grad_seq, 3)


@parameterize("seq_len", [128])
@parameterize("hidden_dim", [64])
@parameterize("head_num", [4])
@parameterize("batch_size", [1])
def run_seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_attn()


@rerun_if_address_is_in_use()
def test_all_to_all_attention():
    spawn(check_all2all_attn, nprocs=4)


if __name__ == "__main__":
    test_all_to_all_attention()
