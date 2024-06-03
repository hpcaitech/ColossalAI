import pytest
import torch

from colossalai.nn.optimizer import CPUAdam, HybridAdam
from colossalai.testing import clear_cache_before_run, parameterize
from tests.kit.model_zoo import model_zoo


def move_some_params_to_cuda(model, torch_model):
    model.embed.weight.data = model.embed.weight.cuda()
    torch_model.embed.weight.data = model.embed.weight.cuda()
    model.ln1.weight.data = model.ln1.weight.cuda()
    torch_model.ln1.weight.data = model.ln1.weight.cuda()


def check_params_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(p, torch_p, atol=1e-3), f"diff: {torch.abs(p - torch_p)}"


# TODO Something wrong with ci when running this test.
@pytest.mark.skip(reason="skip because of something wrong with CI")
@clear_cache_before_run()
@parameterize("nvme_offload_fraction", [0.0, 0.5, 1.0])
@parameterize("nvme_offload_dir", ["./offload", None])
@parameterize("adam_cls", [CPUAdam, HybridAdam])
def test_nvme_adam(nvme_offload_fraction, nvme_offload_dir, adam_cls):
    model_builder, data_gen_fn, *_ = next(iter(model_zoo.get_sub_registry("custom_simple_net").values()))
    model = model_builder()
    torch_model = model_builder()
    move_some_params_to_cuda(model, torch_model)
    optimizer = adam_cls(
        model.parameters(), lr=0.1, nvme_offload_fraction=nvme_offload_fraction, nvme_offload_dir=nvme_offload_dir
    )
    torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.1)
    with torch.no_grad():
        for p, torch_p in zip(model.parameters(), torch_model.parameters()):
            torch_p.copy_(p)
            p.grad = torch.rand_like(p)
            torch_p.grad = p.grad

        for _ in range(3):
            optimizer.step()
            torch_optimizer.step()
            check_params_equal(model, torch_model)


if __name__ == "__main__":
    test_nvme_adam(0.5, "./offload", CPUAdam)
