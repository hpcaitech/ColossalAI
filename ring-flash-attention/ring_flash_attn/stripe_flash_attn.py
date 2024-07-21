import torch
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from .utils import RingComm, update_out_and_lse


def stripe_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, use ring flash attn instead"
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if step <= comm.rank:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q[:, 1:],
                k[:, :-1],
                v[:, :-1],
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse, slice_=(slice(None), slice(1, None)))

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def stripe_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal, "stripe flash attn only supports causal attention, if not causal, ring flash attn instead"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        shift_causal = step > kv_comm.rank
        softmax_lse_1 = None
        if not shift_causal:
            _flash_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                block_dq_buffer,
                block_dk_buffer,
                block_dv_buffer,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )
        else:
            if softmax_lse_1 is None:
                # lazy init, since the last rank does not need softmax_lse_1
                softmax_lse_1 = softmax_lse[:, :, 1:].contiguous()
            _flash_attn_backward(
                dout[:, 1:],
                q[:, 1:],
                k[:, :-1],
                v[:, :-1],
                out[:, 1:],
                softmax_lse_1,
                block_dq_buffer[:, 1:],
                block_dk_buffer[:, :-1],
                block_dv_buffer[:, :-1],
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                rng_state=None,
            )

        if dq is None:
            dq = block_dq_buffer.to(torch.float32)
            dk = block_dk_buffer.to(torch.float32)
            dv = block_dv_buffer.to(torch.float32)
        else:
            if not shift_causal:
                dq += block_dq_buffer
            else:
                dq[:, 1:] += block_dq_buffer[:, 1:]
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk = next_dk
            dv = next_dv

            if not shift_causal:
                dk = block_dk_buffer + dk
                dv = block_dv_buffer + dv
            else:
                dk[:, :-1] += block_dk_buffer[:, :-1]
                dv[:, :-1] += block_dv_buffer[:, :-1]

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = d_kv_comm.send_recv(dv, dv_comm_buffer)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class StripeFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = stripe_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = stripe_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def stripe_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return StripeFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def stripe_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return StripeFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def stripe_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return StripeFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
