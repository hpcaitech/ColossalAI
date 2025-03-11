import ray
import ray.util.collective as cc
import torch
from coati.distributed.comm import ray_broadcast_object, ray_broadcast_tensor_dict

from colossalai.testing import parameterize


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.group_name = "default"
        cc.init_collective_group(world_size, rank, backend="nccl", group_name=self.group_name)

    def run_ray_broadcast_object(self, obj, src, device):
        # ray_broadcast_object
        received_obj = ray_broadcast_object(obj, src, device, group_name=self.group_name)
        return received_obj

    def run_ray_broadcast_tensor_dict(self, tensor_dict, src, device):
        # ray_broadcast_tensor_dict
        received_dict = ray_broadcast_tensor_dict(tensor_dict, src, device, group_name=self.group_name)
        return received_dict

    def destroy_worker(self):
        cc.destroy_collective_group(self.group_name)


@parameterize(
    "test_config",
    [
        {
            "precision": torch.bfloat16,
            "device": "npu",
            "num_devices": 4,
        },
    ],
)
def test_comm(test_config):
    ray.init(num_gpus=4)
    src = 0
    device = test_config["device"]
    # create 4
    workers = [Worker.remote(i, test_config["num_devices"]) for i in range(test_config["num_devices"])]

    #############
    # 1. test ray_broadcast_object
    #############
    # init broadcast_object data
    test_obj = {"data": torch.tensor([1, 2, 3]), "message": "hello"}

    # run run_ray_broadcast_object
    results = [worker.run_ray_broadcast_object.remote(test_obj, src, device) for worker in workers]

    # get result
    results = ray.get(results)

    for i, result in enumerate(results):
        print(f"ray_broadcast_object Rank {i} received object: {result}")

    #############
    # 2. test ray_broadcast_tensor_dict
    #############
    test_tensor_dict = {
        "tensor1": torch.tensor([1, 2, 3], device=device),
        "tensor2": torch.tensor([[4, 5], [6, 7]], device=device),
    }

    # run ray_broadcast_tensor_dict
    results = [worker.run_ray_broadcast_tensor_dict.remote(test_tensor_dict, src, device) for worker in workers]

    # get result
    results = ray.get(results)

    for i, result in enumerate(results):
        print(f"run_ray_broadcast_tensor_dict Rank {i} received object: {result}")

    # destory workers
    for worker in workers:
        worker.destroy_worker.remote()
    ray.shutdown()


if __name__ == "__main__":
    test_comm()
