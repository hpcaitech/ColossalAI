import torch
import pytest
from einops import rearrange
from colossalai.kernel.cuda_native.flash_attention import flash_attention, triton_flash_attention, triton_check


TRITON_AVALIABLE = triton_check()


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 2, 1024, 64)])
def test_triton_flash_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    sm_scale = 0.3
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
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
        assert 1e-3 > torch.mean(torch.abs(ref_out - tri_out))
        assert 1e-3 > torch.mean(torch.abs(ref_dv - tri_dv))
        assert 1e-3 > torch.mean(torch.abs(ref_dk - tri_dk))
        assert 1e-3 > torch.mean(torch.abs(ref_dq - tri_dq))
    else:
        try:
            tri_out = flash_attention(q, k, v, sm_scale, Z, N_CTX)
        except RuntimeError:
            pass
        else:
            raise TypeError("Error type not match!")


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(3, 2, 2048, 64)])
def test_flash_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0, std=.5).requires_grad_()
    sm_scale = 0.3
    dout = torch.randn_like(q)
    
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    
    # triton implementation
    q, k, v = map(lambda x: rearrange(x, 'z h n d -> (z n) h d'), [q, k, v])
    tri_out = flash_attention(q, k, v, sm_scale, Z, N_CTX)
    dout = rearrange(dout, 'z h n d -> (z n) h d').detach()
    tri_out.backward(dout, retain_graph=True)
    tri_dq, tri_dk, tri_dv, = torch.autograd.grad(tri_out, (q, k, v), dout)
    tri_out, tri_dq, tri_dk, tri_dv = map(lambda x: rearrange(x, '(z n) h d -> z h n d', z=Z), (tri_out, tri_dq, tri_dk, tri_dv))

    # compare
    assert 1e-4 > torch.mean(torch.abs(ref_out - tri_out))
    assert 1e-4 > torch.mean(torch.abs(ref_dv - tri_dv))
    assert 1e-4 > torch.mean(torch.abs(ref_dk - tri_dk))
    assert 1e-4 > torch.mean(torch.abs(ref_dq - tri_dq))
