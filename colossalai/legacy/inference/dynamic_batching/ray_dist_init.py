import logging
import os
from typing import List

import ray
import ray.util.collective as collective
import torch
from transformers import AutoModelForCausalLM

import colossalai
from colossalai.inference.async_manager import start_dynamic_batching
from colossalai.inference.dynamic_batching.get_tokenizer import get_tokenizer
from colossalai.inference.dynamic_batching.io_struct import RequestOutput
from colossalai.inference.dynamic_batching.ray_init_config import EngineArgsClass, RooterArgsClass
from colossalai.inference.dynamic_batching.sampling_params import SamplingParams
from colossalai.inference.tensor_parallel.engine import TPInferEngine
from colossalai.shardformer import ShardConfig
from colossalai.testing import free_port

ray_serve_logger = logging.getLogger("ray.serve")


def log_cuda_info(scope_name: str):
    ray_serve_logger.info(f" {scope_name}: ray.get_gpu_ids(): {ray.get_gpu_ids()}")
    ray_serve_logger.info(
        f" {scope_name}: CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'NO DEVICES FOUND!')}"
    )
    if torch.cuda.is_available():
        ray_serve_logger.info(
            f" {scope_name}: cuda current_device: {torch.cuda.current_device()}, cuda device count: {torch.cuda.device_count()}"
        )
    else:
        ray_serve_logger.info(f" {scope_name}: cuda is not available!")


@ray.remote(num_gpus=1)
class Worker:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_batch_size: int,
        max_input_len: int,
        max_output_len: int,
        router_config: RooterArgsClass,
    ):
        log_cuda_info("Worker.init")
        self.tensor_parallel_size = tensor_parallel_size
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.router_config = router_config

    def setup(self, world_size, rank, port):
        # initialize a ray collective group, otherwise colossalai distributed env won't be built successfully
        collective.init_collective_group(world_size, rank, "nccl", "default")
        # initialize and set distributed environment
        colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
        ray_serve_logger.info(f"Worker with rank {rank} (world size {world_size}) setting up..")
        log_cuda_info("Worker.setup")

        # Load model
        self.tokenizer = get_tokenizer(tokenizer_name=self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, pad_token_id=self.tokenizer.pad_token_id, torch_dtype=torch.float16
        )
        shard_config = ShardConfig(
            enable_tensor_parallelism=True if world_size > 1 else False, extra_kwargs={"inference_only": True}
        )
        self.infer_engine = TPInferEngine(
            self.model, shard_config, self.max_batch_size, self.max_input_len, self.max_output_len
        )
        self.start_dynamic_batching = start_dynamic_batching(self.router_config, self.infer_engine, [])

        return True

    # def generate(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> List[str]:
    #     ray_serve_logger.info(f"text: {prompt}")

    #     final_outputs = self.start_dynamic_batching.generate(prompt, sampling_params, request_id)

    #     return final_outputs

    def add_input(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        self.start_dynamic_batching.add_input(request_id, prompt, sampling_params)

    def abort(self, request_id: str):
        self.start_dynamic_batching.abort(request_id)

    def step(self) -> List[RequestOutput]:
        return self.start_dynamic_batching._step()

    def add_req(self, prompt_ids: List[int], sampling_params: SamplingParams, request_id: str, prompt: str):
        self.start_dynamic_batching.add_req(prompt_ids, sampling_params, request_id, prompt)

    def is_running(self):
        return self.start_dynamic_batching.is_running()


class Driver:
    def __init__(self, router_config: RooterArgsClass, engine_config: EngineArgsClass):
        log_cuda_info("Driver:init")
        model_path = engine_config.model
        tensor_parallel_size = engine_config.tensor_parallel_size

        self.num_workers = tensor_parallel_size
        self.workers = []
        init_rets = []

        # Just grab a free port on localhost
        # NOTE workers in this communication group listen to the same port
        available_port = free_port()

        for i in range(self.num_workers):
            worker_name = "worker_idx_{}".format(i)
            w = Worker.options(name=worker_name).remote(
                model_path,
                self.num_workers,
                engine_config.max_batch_size,
                engine_config.max_input_len,
                engine_config.max_output_len,
                router_config,
            )
            self.workers.append(w)
            init_rets.append(w.setup.remote(self.num_workers, i, available_port))
        _options = {
            "group_name": "default_driver",
            "world_size": self.num_workers,
            "ranks": [i for i in range(self.num_workers)],
            "backend": "nccl",
        }
        collective.create_collective_group(self.workers, **_options)
        _ = ray.get(init_rets)

    def add_input(self, request_id: str, prompt: str, sampling_params: SamplingParams):
        ray.get([w.add_input.remote(request_id, prompt, sampling_params) for w in self.workers])

    def abort(self, request_id: str):
        ray.get([w.abort.remote(request_id) for w in self.workers])

    def step(self):
        results = ray.get([w.step.remote() for w in self.workers])
        outputs = results[0]  # get any one of the copies
        return outputs

    def add_req(self, request_id: str, prompt_ids: List[int], sampling_params: SamplingParams, prompt: str):
        ray.get([w.add_req.remote(prompt_ids, sampling_params, request_id, prompt) for w in self.workers])

    def is_running(self):
        results = ray.get([w.is_running.remote() for w in self.workers])
        return any(results)
