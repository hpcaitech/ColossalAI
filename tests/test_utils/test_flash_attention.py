import pytest
import torch
from einops import rearrange

from colossalai.kernel.cuda_native.flash_attention import HAS_FLASH_ATTN, HAS_TRITON, TRITON_AVALIABLE

if HAS_FLASH_ATTN:
    from colossalai.kernel.cuda_native.flash_attention import flash_attention

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


@pytest.mark.skipif(HAS_FLASH_ATTN == False, reason="triton is not available")
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 2, 16, 8)])
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
    if TRITON_AVALIABLE:
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
    else:
        try:
            tri_out = flash_attention(q, k, v, sm_scale, Z, N_CTX)
        except RuntimeError:
            pass
        else:
            raise TypeError("Error type not match!")


@pytest.mark.skipif(HAS_FLASH_ATTN == False, reason="triton is not available")
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 2, 16, 8)])
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
    tri_out = flash_attention(q, k, v, sm_scale, Z, N_CTX)
    dout = rearrange(dout, 'z h n d -> (z n) h d').detach()
    tri_out.backward(dout, retain_graph=True)
    tri_dq, tri_dk, tri_dv, = torch.autograd.grad(tri_out, (q, k, v), dout)
    tri_out, tri_dq, tri_dk, tri_dv = map(lambda x: rearrange(x, '(z n) h d -> z h n d', z=Z),
                                          (tri_out, tri_dq, tri_dk, tri_dv))

    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-3)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-3)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-3)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-3)
