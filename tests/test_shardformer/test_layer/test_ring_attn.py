import torch
import torch.distributed as dist
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer import AttnMaskType, ColoAttention
from colossalai.shardformer.layer.attn import AttnMaskType, RingAttention
from colossalai.shardformer.layer.utils import split_batch_zigzag, split_varlen_zigzag
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device


@parameterize("seq_len", [4096])
@parameterize("bs", [1])
@parameterize("nheads", [5])
@parameterize("d", [128])
@parameterize("dtype", [torch.float16, torch.bfloat16])
def check_ring_attn(seq_len, bs, nheads, d, dtype):
    torch.cuda.manual_seed(2)
    device = get_current_device()
    sp_group = dist.group.WORLD
    sp_stream = torch.cuda.Stream()

    # Some outliers may seem large, but our errors are still lower than
    # than Megatron-LM context parallel's
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
    ring_out = ring_out.transpose(1, 2)
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


@parameterize("seqlen", [16])
@parameterize("bs", [2])
@parameterize("nheads", [5])
@parameterize("d", [128])
@parameterize("dtype", [torch.float16, torch.bfloat16])
def check_packed_seq(seqlen, bs, nheads, d, dtype):
    device = get_current_device()
    sp_group = dist.group.WORLD
    sp_size = dist.get_world_size()
    sp_stream = torch.cuda.Stream()
    atol = rtol = 7e-3

    # Prepare varlen attention mask
    padding_mask = torch.ones((bs, seqlen), dtype=torch.int, device=device)
    # padding_mask[: bs // 2, (seqlen // 4) * 3 :] = 0
    padding_mask[:, seqlen // 2 :] = 0
    mask_info = ColoAttention.prepare_attn_kwargs(
        (bs, 1, seqlen, seqlen), dtype, padding_mask.device, q_padding_mask=padding_mask, is_causal=True
    )
    # input_embeds = torch.randn(bs, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    input_embeds = (
        torch.arange(seqlen, device=device, dtype=dtype, requires_grad=True)
        .repeat(bs, nheads, d, 1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    q, k, v = [input_embeds.clone().transpose(1, 2) for _ in range(3)]

    # Forward
    # out = ColoAttention.attention(q, k, v, **mask_info)
    flat_input = input_embeds.view(-1, nheads, d)[padding_mask.flatten().nonzero().squeeze()]
    qkv = torch.stack([flat_input] * 3, dim=1)
    qkv.retain_grad()
    out, lse, _ = flash_attn_varlen_qkvpacked_func(
        qkv, mask_info["cu_seqlens_q"], mask_info["max_seqlen_q"], return_attn_probs=True, causal=True
    )

    input_embeds, _, mask_info = RingAttention.prepare_varlen_batch(input_embeds, padding_mask, sp_group)
    # Test the splitting function
    local_input = split_varlen_zigzag(
        flat_input, mask_info["cu_seqlens"] * sp_size, sp_group, mask_info["max_seqlen"] * sp_size
    )
    assert (local_input == input_embeds.view(-1, nheads, d)[mask_info["valid_indices"]]).all()
    del local_input, flat_input

    q_ring, k_ring, v_ring = [input_embeds.clone().transpose(1, 2) for _ in range(3)]
    q_ring.retain_grad()
    k_ring.retain_grad()
    v_ring.retain_grad()
    ring_out, ring_lse = RingAttention.attention(
        q_ring, k_ring, v_ring, sp_group, sp_stream, **mask_info, pad_output=False, return_softmax=True
    )

    # Check output
    # ring_out, out = [x.transpose(1, 2) for x in (ring_out, out)] # to (B, Sq, nHeads, D)
    # out = split_varlen_zigzag(out, mask_info["cu_seqlens"] * sp_size, sp_group, mask_info["max_seqlen"] * sp_size, is_2d=True)
    lse = lse.transpose(0, 1)
    out, lse = split_varlen_zigzag(
        [out, lse], mask_info["cu_seqlens"] * sp_size, sp_group, mask_info["max_seqlen"] * sp_size
    )
    # assert_close(lse, ring_lse, atol=atol, rtol=rtol)
    assert_close(out, ring_out, atol=atol, rtol=rtol)

    # Check grads
    out.sum().backward()
    ring_out.sum().backward()
    dq, dk, dv = [
        split_varlen_zigzag(
            qkv.grad[:, i], mask_info["cu_seqlens"] * sp_size, sp_group, mask_info["max_seqlen"] * sp_size
        )
        for i in range(3)
    ]
    dq_ring, dk_ring, dv_ring = [
        x.transpose(1, 2).reshape(-1, nheads, d)[mask_info["valid_indices"]]
        for x in (q_ring.grad, k_ring.grad, v_ring.grad)
    ]
    assert_close(dq, dq_ring, atol=atol, rtol=rtol)
    assert_close(dk, dk_ring, atol=atol, rtol=rtol)
    assert_close(dv, dv_ring, atol=atol, rtol=rtol)


def launch(rank, world_size, port):
    colossalai.launch(rank, world_size, "localhost", port)
    # check_packed_seq()
    check_ring_attn()


@rerun_if_address_is_in_use()
def test_ring_attn():
    spawn(launch, nprocs=8)


if __name__ == "__main__":
    test_ring_attn()
