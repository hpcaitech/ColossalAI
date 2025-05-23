import argparse
import time

import ray
import ray.util.collective as cc
import torch
from coati.distributed.comm import ray_broadcast_tensor_dict
from vllm import LLM, SamplingParams

from colossalai.testing import parameterize

parser = argparse.ArgumentParser(description="VLLM args.")
parser.add_argument(
    "-m", "--model_path", type=str, default="/home/duanjunwen/models/Qwen/Qwen2.5-14B", help="The model path. "
)
parser.add_argument("-l", "--max_length", type=int, default=8192, help="Max sequence length")
parser.add_argument("-w", "--world_size", type=int, default=8, help="Gpu nums")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
parser.add_argument(
    "-i", "--input_texts", type=str, default="Find all prime numbers up to 100.", help="Prompts inputs. "
)
args = parser.parse_args()

# Create a sampling params object.


@ray.remote(num_cpus=args.world_size, num_gpus=0, resources={"NPU": args.world_size})
class Worker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.group_name = "default"
        cc.init_collective_group(world_size, rank, backend="hccl", group_name=self.group_name)
        self.llm = LLM(model=args.model_path, max_model_len=args.max_length, tensor_parallel_size=args.world_size)
        self.sampling_params = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length
        )

    def run_ray_broadcast_object(self, obj, src, device):
        # Create an LLM.
        outputs = self.llm.generate(args.input_texts, self.sampling_params)
        return outputs

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
            "num_devices": 1,
        },
    ],
)
def test_comm(test_config):
    ray.init(address="local", namespace="ray-example")
    # ray.init(_node_ip_address="10.0.0.3", namespace="ray-vllm")
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
    # for i in range(5):
    # if i > 2:
    torch.npu.synchronize()
    start_time = time.time()
    results = [worker.run_ray_broadcast_object.remote(test_obj, src, device) for worker in workers]

    # get result
    results = ray.get(results)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"total_time {total_time}")

    for i, result in enumerate(results):
        print(f"ray_broadcast_object Rank {i} received object: {result}")

    # destory workers
    for worker in workers:
        worker.destroy_worker.remote()
    ray.shutdown()


if __name__ == "__main__":
    test_comm()
