import pytest
import torch

from colossalai.nn.optimizer import CPUAdam, HybridAdam
from colossalai.testing import clear_cache_before_run, parameterize
from tests.components_to_test.registry import non_distributed_component_funcs


def move_some_params_to_cuda(model, torch_model):
    model.embed.weight.data = model.embed.weight.cuda()
    torch_model.embed.weight.data = model.embed.weight.cuda()
    model.ln1.weight.data = model.ln1.weight.cuda()
    torch_model.ln1.weight.data = model.ln1.weight.cuda()


def check_params_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(p, torch_p, atol=1e-3), f'diff: {torch.abs(p - torch_p)}'


@clear_cache_before_run()
@parameterize('nvme_offload_fraction', [0.0, 0.5, 1.0])
@parameterize('nvme_offload_dir', ['./offload', None])
@parameterize('adam_cls', [CPUAdam, HybridAdam])
def test_nvme_adam(nvme_offload_fraction, nvme_offload_dir, adam_cls):
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    model = model_builder()
    torch_model = model_builder()
    move_some_params_to_cuda(model, torch_model)
    optimizer = adam_cls(model.parameters(),
                         lr=0.1,
                         nvme_offload_fraction=nvme_offload_fraction,
                         nvme_offload_dir=nvme_offload_dir)
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


if __name__ == '__main__':
    test_nvme_adam(0.5, './offload', CPUAdam)
