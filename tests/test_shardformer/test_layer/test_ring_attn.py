import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer.attn import AttnMaskType, RingAttention
from colossalai.shardformer.layer.utils import zigzag_split_batch
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


@parameterize("seq_len", [4096])
@parameterize("batch_size", [1])
@parameterize("nheads", [5])
@parameterize("d", [128])
@parameterize("dtype", [torch.bfloat16])
def check_ring_attn(seq_len, batch_size, nheads, d, dtype):
    torch.cuda.manual_seed(2)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    sp_group = dist.group.WORLD
    sp_stream = torch.cuda.Stream()

    # Some outliers may seem large, but our errors are still lower than
    # than Megatron-LM's context parallel's
    # (https://github.com/NVIDIA/TransformerEngine/blob/33a3d02f81c56e6f7b542c09bfa86657078d57fb/tests/pytorch/fused_attn/run_fused_attn_with_cp.py#L215)
    # and the original zigzag implementation's (https://github.com/zhuzilin/ring-flash-attention/tree/main)
    atol = rtol = 7e-3

    # Setup inputs
    qkv = torch.randn(batch_size, seq_len, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    local_qkv = zigzag_split_batch(qkv, sp_group)
    q, k, v = local_qkv.unbind(dim=-3)
    q, k, v = [x.squeeze(2).detach().clone().transpose(1, 2) for x in (q, k, v)]  # (B, nHeads, Sq, D)
    q.requires_grad = k.requires_grad = v.requires_grad = True

    # Ring attention vs single GPU
    ring_out, ring_lse = RingAttention.attention(q, k, v, sp_group, sp_stream, AttnMaskType.CAUSAL, return_softmax=True)
    ring_lse = ring_lse.transpose(0, 1).view(batch_size, seq_len // world_size, nheads).transpose(1, 2).contiguous()
    out, lse, _ = flash_attn_qkvpacked_func(
        qkv, dropout_p=0.0, causal=True, window_size=(-1, -1), alibi_slopes=None, return_attn_probs=True
    )

    local_out = zigzag_split_batch(out, sp_group)
    local_lse = zigzag_split_batch(lse, sp_group, seq_dim=-1)
    assert_close(ring_out, local_out, atol=atol, rtol=rtol)
    assert_close(ring_lse, local_lse, atol=atol, rtol=rtol)

    ring_out.sum().backward()
    out.sum().backward()
    ring_dq, ring_dk, ring_dv = [x.transpose(1, 2) for x in (q.grad, k.grad, v.grad)]
    dqkv = qkv.grad
    local_dqkv = zigzag_split_batch(dqkv, sp_group)
    assert_close(ring_dq, local_dqkv[:, :, 0], atol=atol, rtol=rtol)
    assert_close(ring_dk, local_dqkv[:, :, 1], atol=atol, rtol=rtol)
    assert_close(ring_dv, local_dqkv[:, :, 2], atol=atol, rtol=rtol)


def launch(rank, world_size, port):
    colossalai.launch(rank, world_size, "localhost", port)
    check_ring_attn()


@rerun_if_address_is_in_use()
def test_ring_attn():
    spawn(launch, nprocs=8)


if __name__ == "__main__":
    test_ring_attn()
