"""Small, assertion-based runner qualification; this does not test ColossalAI."""

import argparse
import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def environment():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "distributed_available": dist.is_available(),
        "nccl_available": dist.is_available() and dist.is_nccl_available(),
    }


def check_collectives_and_training(device, rank, local_rank):
    # Exercise actual tensor computation, then verify the numerical result.
    left = torch.ones((128, 128), device=device)
    right = torch.full((128, 128), 2.0, device=device)
    torch.testing.assert_close(left @ right, torch.full_like(left, 256.0), rtol=0, atol=0)

    reduced = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(reduced, torch.tensor([3.0], device=device), rtol=0, atol=0)

    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.zero_()
    ddp = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)

    # Different data on each rank makes missing gradient synchronization observable.
    # Global x = [1, 2, 3, 4], y = 3*x. mean(x**2) = 7.5, so grad = 15*(w-3).
    inputs = torch.tensor([[1.0 + 2 * rank], [2.0 + 2 * rank]], device=device)
    targets = 3 * inputs
    expected_weight = 0.0
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(ddp(inputs), targets)
        require(torch.isfinite(loss).item(), "Non-finite training loss")
        loss.backward()

        expected_gradient = 15.0 * (expected_weight - 3.0)
        torch.testing.assert_close(
            ddp.module.weight.grad,
            torch.full_like(ddp.module.weight, expected_gradient),
            rtol=1e-5,
            atol=1e-5,
        )
        optimizer.step()
        expected_weight -= 0.01 * expected_gradient
        torch.testing.assert_close(
            ddp.module.weight,
            torch.full_like(ddp.module.weight, expected_weight),
            rtol=1e-5,
            atol=1e-5,
        )

    weight = ddp.module.weight.detach().contiguous()
    gathered = [torch.empty_like(weight) for _ in range(2)]
    dist.all_gather(gathered, weight)
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    return float(weight.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Inspect the build without initializing CUDA")
    modes.add_argument("--cpu-control", action="store_true", help="Validate test math on CPU/Gloo only")
    args = parser.parse_args()

    metadata = environment()
    if args.check:
        require(metadata["distributed_available"], "PyTorch distributed is unavailable")
        require(metadata["nccl_available"], "PyTorch was not built with NCCL")
        require(metadata["cuda_build"] is not None, "PyTorch was not built with CUDA")
        print(json.dumps({"mode": "environment_check", "gpu_tested": False, **metadata}), flush=True)
        return

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    require(world_size == 2 and rank in (0, 1) and local_rank in (0, 1), "Launch exactly two torchrun workers")

    if args.cpu_control:
        require(torch.cuda.device_count() == 0, "CPU control must run in a container without GPU access")
        device = torch.device("cpu")
        backend = "gloo"
    else:
        require(torch.cuda.is_available(), "CUDA unavailable: GPU test cannot be skipped or passed")
        require(torch.cuda.device_count() == 2, "Expose exactly two assigned GPUs to this container")
        require(metadata["nccl_available"], "NCCL unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
        properties = torch.cuda.get_device_properties(device)
        print(
            json.dumps({"rank": rank, "device": str(device), "name": properties.name, "uuid": str(properties.uuid)}),
            flush=True,
        )

    torch.set_num_threads(1)
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=45))
    try:
        weight = check_collectives_and_training(device, rank, local_rank)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
        if rank == 0:
            marker = "E1_CPU_CONTROL_PASS" if args.cpu_control else "E1_GPU_SMOKE_PASS"
            print(
                json.dumps(
                    {
                        "result": marker,
                        "gpu_tested": not args.cpu_control,
                        "backend": backend,
                        "world_size": world_size,
                        "checks": ["matmul", "all_reduce_sum", "ddp_gradients", "sgd_updates", "equal_weights"],
                        "final_weight": weight,
                        **metadata,
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
