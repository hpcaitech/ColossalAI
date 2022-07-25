import pytest
import torch
from tests.components_to_test.registry import non_distributed_component_funcs
from colossalai.nn.optimizer import CPUAdam


def check_params_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert torch.allclose(p, torch_p, atol=1e-3), f'diff: {torch.abs(p - torch_p)}'


@pytest.mark.parametrize('nvme_offload_factor', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('nvme_offload_dir', ['./offload', None])
def test_nvme_adam(nvme_offload_factor, nvme_offload_dir):
    get_components_func = non_distributed_component_funcs.get_callable('simple_net')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()
    model = model_builder()
    torch_model = model_builder()
    optimizer = CPUAdam(model.parameters(),
                        lr=0.1,
                        nvme_offload_factor=nvme_offload_factor,
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
    test_nvme_adam(0.5, './offload')
