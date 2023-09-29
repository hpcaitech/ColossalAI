import logging
import os
from dataclasses import dataclass
from typing import Any, List, Union

import ray
import ray.util.collective as collective
import starlette
import torch
from ray import serve
from transformers import BloomForCausalLM, BloomTokenizerFast

import colossalai
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
    def __init__(self, model_path: str, tp_size: int, max_batch_size: int, max_input_len: int, max_output_len: int):
        log_cuda_info("Worker.init")
        self.tp_size = tp_size
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def setup(self, world_size, rank, port):
        # initialize a ray collective group, otherwise colossalai distributed env won't be built successfully
        collective.init_collective_group(world_size, rank, "nccl", "default")
        # initialize and set distributed environment
        colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
        ray_serve_logger.info(f"Worker with rank {rank} (world size {world_size}) setting up..")
        log_cuda_info("Worker.setup")

        # Load model
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = BloomForCausalLM.from_pretrained(
            self.model_path, pad_token_id=self.tokenizer.eos_token_id, torch_dtype=torch.float16
        )

        shard_config = ShardConfig(enable_tensor_parallelism=True if world_size > 1 else False, inference_only=True)
        self.infer_engine = TPInferEngine(
            self.model, shard_config, self.max_batch_size, self.max_input_len, self.max_output_len
        )
        self.generate_kwargs = dict(max_new_tokens=self.max_output_len, do_sample=False)

        return True

    def generate(self, text: Union[str, List[str]]) -> str:
        input_tokens = self.tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True)
        ray_serve_logger.info(f"text: {text},\ninput_tokens: {input_tokens}")

        model_output = self.infer_engine.generate(input_tokens, **self.generate_kwargs)
        ray_serve_logger.info(f"model_output.shape: {model_output.shape}")

        text_output = []
        for i in range(len(model_output)):
            text_output.append(self.tokenizer.decode(model_output[i]))
        ray_serve_logger.info(f"output: {text_output}")

        return text_output


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0})
class Driver:
    def __init__(self, config):
        log_cuda_info("Driver:init")
        model_path = config.model_path
        tp_size = config.tp_size

        self.num_workers = tp_size
        self.workers = []
        init_rets = []

        # Just grab a free port on localhost
        # NOTE workers in this communication group listen to the same port
        available_port = free_port()

        for i in range(self.num_workers):
            worker_name = "worker_idx_{}".format(i)
            w = Worker.options(name=worker_name).remote(
                model_path, self.num_workers, config.max_batch_size, config.max_input_len, config.max_output_len
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

    # set batch wait delay in seconds and maximum number of sequences in a batch
    @serve.batch(batch_wait_timeout_s=0.8, max_batch_size=4)
    async def batch_generate(self, requests: List[str]):
        ray_serve_logger.info(f"Driver.batch_generate: requests length: {len(requests)}\n requests: {requests}")
        results = ray.get([w.generate.remote(requests) for w in self.workers])
        text_res = results[0]  # get any one of the copies
        return text_res

    async def __call__(self, request: starlette.requests.Request) -> Any:
        return await self.batch_generate(request.query_params["text"])


@dataclass
class Config:
    """temp config"""

    model_path: str
    tp_size: int = 2
    max_batch_size: int = 4
    max_input_len: int = 128
    max_output_len: int = 32


# *** add model path manually into the config***
driver_config = Config(model_path="ADD_MODEL_PATH_HRER")
app = Driver.bind(config=driver_config)
