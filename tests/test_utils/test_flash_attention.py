import pytest
import torch
from einops import rearrange

from colossalai.kernel.cuda_native.flash_attention import HAS_FLASH_ATTN, HAS_TRITON

if HAS_FLASH_ATTN:
    from colossalai.kernel.cuda_native.flash_attention import (
        flash_attention_q_k_v, flash_attention_q_kv, flash_attention_qkv)

if HAS_TRITON:
    from colossalai.kernel.cuda_native.flash_attention import triton_flash_attention


def baseline_attention(Z, N_CTX, H, q, k, v, sm_scale):
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


@pytest.mark.skipif(HAS_TRITON == False, reason="triton is not available")
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 4, 2, 16)])
def test_triton_flash_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    sm_scale = 0.3
    dout = torch.randn_like(q)

    ref_out = baseline_attention(Z, N_CTX, H, q, k, v, sm_scale)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    tri_out = triton_flash_attention(q, k, v, sm_scale)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-3)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-3)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-3)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-3)


@pytest.mark.skipif(HAS_FLASH_ATTN == False, reason="flash is not available")
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 4, 2, 16)])
def test_flash_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    sm_scale = 0.3
    dout = torch.randn_like(q)

    # reference implementation
    ref_out = baseline_attention(Z, N_CTX, H, q, k, v, sm_scale)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # flash implementation
    q, k, v = map(lambda x: rearrange(x, 'z h n d -> (z n) h d'), [q, k, v])
    dout = rearrange(dout, 'z h n d -> (z n) h d').detach()
    for i in range(3):
        if i == 0:
            tri_out = flash_attention_q_k_v(q, k, v, sm_scale, Z, N_CTX, N_CTX, causal=True)
        elif i == 1:
            kv = torch.cat((k.unsqueeze(1), v.unsqueeze(1)), dim=1)
            tri_out = flash_attention_q_kv(q, kv, sm_scale, Z, N_CTX, N_CTX, causal=True)
        else:
            qkv = torch.cat((q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)), dim=1)
            tri_out = flash_attention_qkv(qkv, sm_scale, Z, N_CTX, causal=True)

        tri_out.backward(dout, retain_graph=True)

        if i == 0:
            tri_dq, tri_dk, tri_dv, = torch.autograd.grad(tri_out, (q, k, v), dout)
            tri_out, tri_dq, tri_dk, tri_dv = map(lambda x: rearrange(x, '(z n) h d -> z h n d', z=Z),
                                                (tri_out, tri_dq, tri_dk, tri_dv))
        elif i == 1:
            tri_dq, tri_dkv, = torch.autograd.grad(tri_out, (q, kv), dout)
            tri_dk, tri_dv = torch.chunk(tri_dkv, 2, dim=1)
            tri_out, tri_dq, tri_dk, tri_dv = map(lambda x: rearrange(x, '(z n) h d -> z h n d', z=Z),
                                                (tri_out, tri_dq, tri_dk.squeeze(1), tri_dv.squeeze(1)))
        else:
            tri_dqkv, = torch.autograd.grad(tri_out, (qkv), dout)
            tri_dq, tri_dk, tri_dv = torch.chunk(tri_dqkv, 3, dim=1)
            tri_out, tri_dq, tri_dk, tri_dv = map(lambda x: rearrange(x, '(z n) h d -> z h n d', z=Z),
                                                (tri_out, tri_dq.squeeze(1), tri_dk.squeeze(1), tri_dv.squeeze(1)))

        # compare
        assert torch.allclose(ref_out, tri_out, atol=1e-3)
        assert torch.allclose(ref_dv, tri_dv, atol=1e-3)
        assert torch.allclose(ref_dk, tri_dk, atol=1e-3)
        assert torch.allclose(ref_dq, tri_dq, atol=1e-3)


if __name__ == '__main__':
    test_flash_attention(3, 4, 2, 16)
