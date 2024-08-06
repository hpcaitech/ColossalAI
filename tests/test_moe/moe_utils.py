import torch


def assert_loose_close(a, b, dtype: torch.dtype = torch.float32, name=""):
    assert loose_close(a, b, dtype), f"{name} not close {a.mean()} {b.mean()}"


def loose_close(a, b, dtype: torch.dtype = torch.float32):
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
        rtol = 1e-05
        atol = 1e-08

    a = a.detach().to(dtype)
    b = b.detach().to(dtype).to(a.device)

    return torch.allclose(a, b, rtol=rtol, atol=atol)


def check_model_equal(model1, model2):
    assert set(model1.state_dict().keys()) == set(model2.state_dict().keys())
    for i, ((name, p1), p2) in enumerate(zip(model1.named_parameters(), model2.parameters())):
        assert_loose_close(p1, p2, p1.dtype)
