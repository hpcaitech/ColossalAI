import torch


def loose_close(a, b, dtype: torch.dtype = torch.float32, name=""):
    rtol = None
    atol = None
    if dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3
    else:
        assert dtype is torch.float32
        rtol = 1e-5
        atol = 1e-5

    a = a.detach().to(dtype)
    b = b.detach().to(dtype).to(a.device)

    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"{name} not close {a.mean()} {b.mean()}"
