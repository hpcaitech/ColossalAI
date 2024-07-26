import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing import assert_close

from colossalai.quantization.utils import patch_fsdp_params_comm_hook

patch_fsdp_params_comm_hook()

# example modified from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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


def demo_basic(rank, world_size):
    print(f"Running basic FSDP example on rank {rank}.")
    setup(rank, world_size)

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

    grad_dict = get_grads_after_one_iteration()
    for hook in [
        fp8_compress_fsdp_grad_comm_hook,
    ]:
        grad_dict_w_hook = get_grads_after_one_iteration(grad_hook=hook)
        if dist.get_rank() == 0:
            for name in grad_dict:
                assert_close(grad_dict[name], grad_dict_w_hook[name], rtol=0.1, atol=0.1)

    for hook in [
        fp8_compress_fsdp_params_comm_hook,
    ]:
        grad_dict_w_hook = get_grads_after_one_iteration(params_hook=hook)
        if dist.get_rank() == 0:
            for name in grad_dict:
                assert_close(grad_dict[name], grad_dict_w_hook[name], rtol=0.1, atol=0.1)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
