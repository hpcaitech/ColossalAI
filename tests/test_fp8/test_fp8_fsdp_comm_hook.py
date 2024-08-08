import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing import assert_close

from colossalai import launch
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn

# example modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 50)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


@parameterize("mode", ["grad", "params"])
def run_model(mode):
    rank = dist.get_rank()

    from colossalai.quantization.utils import patch_fsdp_params_comm_hook

    patch_fsdp_params_comm_hook()

    def get_grads_after_one_iteration(grad_hook=None, params_hook=None):
        torch.manual_seed(0)
        # create model and move it to GPU with id rank
        model = ToyModel().to(rank)
        fsdp_model = FSDP(model)

        if grad_hook is not None:
            fsdp_model.register_comm_hook(None, grad_hook)

        if params_hook is not None:
            fsdp_model.register_params_comm_hook(None, params_hook)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(fsdp_model.parameters(), lr=0.001)

        optimizer.zero_grad()
        outputs = fsdp_model(torch.randn(20, 100))
        labels = torch.randn(20, 50).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()

        torch.distributed.barrier()

        grad_dict = {}
        for name, params in fsdp_model.named_parameters():
            grad_dict[name] = params.grad
        return grad_dict

    from colossalai.quantization.fp8 import fp8_compress_fsdp_grad_comm_hook, fp8_compress_fsdp_params_comm_hook

    if mode == "grad":
        grad_dict = get_grads_after_one_iteration()
        for hook in [
            fp8_compress_fsdp_grad_comm_hook,
        ]:
            grad_dict_w_hook = get_grads_after_one_iteration(grad_hook=hook)
            if dist.get_rank() == 0:
                for name in grad_dict:
                    assert_close(grad_dict[name], grad_dict_w_hook[name], rtol=0.1, atol=0.1)
    elif mode == "params":
        grad_dict = get_grads_after_one_iteration()
        for hook in [
            fp8_compress_fsdp_params_comm_hook,
        ]:
            grad_dict_w_hook = get_grads_after_one_iteration(params_hook=hook)
            if dist.get_rank() == 0:
                for name in grad_dict:
                    assert_close(grad_dict[name], grad_dict_w_hook[name], rtol=0.1, atol=0.1)
    else:
        raise NotImplementedError


def demo_basic(rank, world_size, port):
    print(f"Running basic FSDP example on rank {rank}.")
    launch(rank=rank, world_size=world_size, port=port, host="localhost")
    run_model()
    cleanup()


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.2.0"), reason="torch version < 2.2.0.")
@rerun_if_address_is_in_use()
def test_fsdp():
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    spawn(demo_basic, n_gpus)


if __name__ == "__main__":
    test_fsdp()
