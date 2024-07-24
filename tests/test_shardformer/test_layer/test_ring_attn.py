import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import AttnMaskType, ColoAttention
from colossalai.shardformer.layer.attn import AttnMaskType, RingAttention
from colossalai.shardformer.layer.utils import split_batch_zigzag
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


@parameterize("seq_len", [4096])
@parameterize("bs", [1])
@parameterize("nheads", [5])
@parameterize("d", [128])
@parameterize("dtype", [torch.bfloat16])
def check_ring_attn(seq_len, bs, nheads, d, dtype):
    torch.cuda.manual_seed(2)
    dist.get_rank()
    dist.get_world_size()
    device = get_current_device()
    sp_group = dist.group.WORLD
    sp_stream = torch.cuda.Stream()

    # Some outliers may seem large, but our errors are still lower than
    # than Megatron-LM's context parallel's
    # (https://github.com/NVIDIA/TransformerEngine/blob/33a3d02f81c56e6f7b542c09bfa86657078d57fb/tests/pytorch/fused_attn/run_fused_attn_with_cp.py#L215)
    # and the original zigzag implementation's (https://github.com/zhuzilin/ring-flash-attention/tree/main)
    atol = rtol = 7e-3

    # Setup inputs
    qkv = torch.randn(bs, seq_len, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
    local_qkv = split_batch_zigzag(qkv, sp_group)
    q, k, v = local_qkv.unbind(dim=-3)
    q, k, v = [x.squeeze(2).detach().clone().transpose(1, 2) for x in (q, k, v)]  # (B, nHeads, Sq, D)
    q.requires_grad = k.requires_grad = v.requires_grad = True

    # Ring attention vs single GPU
    ring_out, ring_lse = RingAttention.attention(q, k, v, sp_group, sp_stream, AttnMaskType.CAUSAL, return_softmax=True)
    out, lse, _ = flash_attn_qkvpacked_func(
        qkv, dropout_p=0.0, causal=True, window_size=(-1, -1), alibi_slopes=None, return_attn_probs=True
    )

    # Checkout out and softmax denominator
    local_out = split_batch_zigzag(out, sp_group)
    local_lse = split_batch_zigzag(lse, sp_group, seq_dim=-1)
    local_lse = local_lse.transpose(1, 2).contiguous().view(-1, ring_lse.shape[-1])  # (B, nHeads, Sq) -> (T, nHeads)
    assert_close(ring_out, local_out, atol=atol, rtol=rtol)
    assert_close(ring_lse, local_lse, atol=atol, rtol=rtol)

    # Check grads
    ring_out.sum().backward()
    out.sum().backward()
    ring_dq, ring_dk, ring_dv = [x.transpose(1, 2) for x in (q.grad, k.grad, v.grad)]
    dqkv = qkv.grad
    local_dqkv = split_batch_zigzag(dqkv, sp_group)
    assert_close(ring_dq, local_dqkv[:, :, 0], atol=atol, rtol=rtol)
    assert_close(ring_dk, local_dqkv[:, :, 1], atol=atol, rtol=rtol)
    assert_close(ring_dv, local_dqkv[:, :, 2], atol=atol, rtol=rtol)


@parameterize("seq_len", [4096])
@parameterize("bs", [2])
@parameterize("nheads", [5])
@parameterize("d", [128])
@parameterize("dtype", [torch.bfloat16])
def check_packed_seq(seq_len, bs, nheads, d, dtype):
    device = get_current_device()
    sp_group = dist.group.WORLD
    sp_stream = torch.cuda.Stream()
    atol = rtol = 5e-3

    # Prepare varlen attention mask
    padding_mask = torch.ones((bs, seq_len), dtype=torch.int, device=device)
    padding_mask[bs // 2 :, seq_len // 2 :] = 0
    padding_mask[: bs // 2, (seq_len // 4) * 3 :] = 0
    attn_mask = ColoAttention.prepare_attn_kwargs(
        (bs, 1, seq_len, seq_len), dtype, padding_mask.device, q_padding_mask=padding_mask, is_causal=True
    )
    input_embeds = torch.randn(bs, seq_len, nheads, d, device=device, dtype=dtype, requires_grad=True)

    # Forward
    q, k, v = [input_embeds.clone().transpose(1, 2) for _ in range(3)]
    colo_out = ColoAttention.attention(q, k, v, **attn_mask)

    input_embeds, _, attn_mask = RingAttention.prepare_varlen_batch(input_embeds, padding_mask, sp_group, bs)
    q_ring, k_ring, v_ring = [input_embeds.clone().transpose(1, 2) for _ in range(3)]
    ring_out = RingAttention.attention(q_ring, k_ring, v_ring, sp_group, sp_stream, **attn_mask)

    # Check output
    colo_out = split_batch_zigzag(colo_out, sp_group)
    assert_close(colo_out, ring_out, atol=atol, rtol=rtol)
    # Check grads
    colo_out.backward()
    ring_out.backward()
    assert_close(q.grad, q_ring.grad, atol=atol, rtol=rtol)
    assert_close(k.grad, k_ring.grad, atol=atol, rtol=rtol)
    assert_close(v.grad, v_ring.grad, atol=atol, rtol=rtol)


def launch(rank, world_size, port):
    colossalai.launch(rank, world_size, "localhost", port)
    # check_ring_attn()
    check_packed_seq()


@rerun_if_address_is_in_use()
def test_ring_attn():
    spawn(launch, nprocs=8)


if __name__ == "__main__":
    test_ring_attn()
