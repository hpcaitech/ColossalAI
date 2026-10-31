import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["qwen2", "qwen3"])
def family(request):
    return request.param
