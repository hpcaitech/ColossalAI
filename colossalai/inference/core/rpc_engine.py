import asyncio
from itertools import count
from time import sleep
from typing import List, Tuple, Union

import rpyc
import torch
import torch.nn as nn
from rpyc.utils.server import ThreadedServer
from torch import multiprocessing as mp
from transformers import AutoConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.configuration_utils import PretrainedConfig

from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.config import InferenceConfig, InputMetaData
from colossalai.inference.executor.rpc_worker import rpcWorkerService
from colossalai.inference.utils import find_available_ports
from colossalai.logging import get_dist_logger
from colossalai.shardformer.policies.base_policy import Policy

from .engine import InferenceEngine
from .request_handler import RPCRequestHandler

__all__ = ["RPCInferenceEngine"]


def run_server(host, port, event: mp.Event = None):
    server = ThreadedServer(
        rpcWorkerService, port=port, protocol_config={"allow_public_attrs": True, "allow_all_attrs": True}
    )
    if event:
        event.set()
    server.start()


class RPCInferenceEngine(InferenceEngine):
    """
    InferenceEngine which manages the inference process..

    NOTE This `RPCInferenceEngine` is designed for multiple-card/online serving.
    Original `InferenceEngine` is designed for single card and offline service, though it supports multi-card offline inference.

    Args:
        model_or_path (nn.Module or str): Path or nn.Module of this model, Currently we don't support `nn.Module` Format
        tokenizer Optional[(Union[PreTrainedTokenizer, PreTrainedTokenizerFast])]: Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        inference_config: InferenceConfig,
        verbose: bool = False,
        model_policy: Policy = None,
    ) -> None:
        """
        If you input a real model loaded by transformers, the init will take quite a long time
        Currently we don't support model(nn.Module) format as the param.
        """

        torch.multiprocessing.set_start_method("spawn", force=True)

        self.inference_config = inference_config
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.verbose = verbose
        self.logger = get_dist_logger(__name__)

        try:
            if isinstance(model_or_path, str):
                self.model_config = AutoConfig.from_pretrained(
                    model_or_path, trust_remote_code=True, torch_dtype=self.dtype
                )
            elif isinstance(model_or_path, nn.Module):
                self.logger.error(
                    f"An exception occurred during loading model Config: For {__class__.__name__}, we don't support param like nn.Module currently\n"
                )
                # self.model_config = model_or_path.config
            else:
                self.logger.error(
                    f"An exception occurred during loading model Config: Please pass right param for {__class__.__name__}\n"
                )
        except Exception as e:
            self.logger.error(
                f"An exception occurred during loading model Config: {e}, The path should be transformers-like\n"
            )
        self.generation_config = inference_config.to_generation_config(self.model_config)

        self.tp_size = inference_config.tp_size
        self.events = [mp.Event() for _ in range(self.tp_size)]

        # This operation will init the dist env and models
        self.workers: List[rpcWorkerService] = []
        self.init_workers()

        asyncio.run(self.init_model(model_or_path, model_policy))

        # init the scheduler and logic block manager
        self.request_handler = self.init_scheduler(self.inference_config, self.model_config)

        # init the physical cache
        alloc_shape = self.request_handler.cache_manager.get_physical_cache_shape()
        self.init_device_cache(alloc_shape)

        self.use_cuda_graph = self.inference_config.use_cuda_graph
        self.high_precision = inference_config.high_precision
        self.dtype = inference_config.dtype

        # Model and relatable attrs of speculative decoding will be set by `enable_spec_dec`
        self.use_spec_dec = False
        self.drafter_model = None
        self.drafter = None
        self.use_glide = False
        self.n_spec_tokens = self.inference_config.max_n_spec_tokens

        self.counter = count()
        self._verify_args()

        self.logger.info("engine init over ")

    def _verify_args(self) -> None:
        """Verify the input args"""
        if not isinstance(self.inference_config, InferenceConfig):
            raise TypeError("Invalid type of inference config provided.")
        if not isinstance(self.tokenizer, (PreTrainedTokenizerFast, PreTrainedTokenizer)):
            raise TypeError(
                f"the tokenizer type must be PreTrainedTokenizer or PreTrainedTokenizerFast, but got {type(self.tokenizer)}"
            )

    def init_workers(self):
        rpc_ports = find_available_ports(self.tp_size)
        self.worker_processes = []
        # mp.set_start_method('spawn')
        for event, rpc_port in zip(self.events, rpc_ports):
            p = mp.Process(target=run_server, args=("localhost", rpc_port, event))
            p.start()
            self.worker_processes.append(p)
            self.logger.info(f"Starting RPC Worker on localhost:{rpc_port}...")

        # Wait for all servers to start
        for event in self.events:
            event.wait()
            event.clear()

        sleep(0.05)

        self.logger.info(f"init rpc server done.")

        for rpc_port in rpc_ports:
            try:
                conn = rpyc.connect(
                    "localhost",
                    rpc_port,
                    config={"allow_pickle": True, "allow_public_attrs": True, "allow_all_attrs": True},
                )
                self.workers.append(conn.root)
            except:
                raise Exception("conn error!")
        self.logger.info(f"Build RPC Connection Success! Begin to load model...")
        asyncio.run(self.init_worker_env())
        self.logger.info(f"init dist env over")

    async def async_parallel_wrapper(self, f, *args, **kwargs):
        async_res = rpyc.async_(f)(*args, **kwargs)
        await asyncio.to_thread(async_res.wait)
        assert async_res.ready
        return async_res.value

    async def init_worker_env(self):
        assert len(self.workers) == self.tp_size, "init workers first"

        dist_group_port = find_available_ports(1)[0]
        init_tasks = [
            self.async_parallel_wrapper(
                worker.init_dist_env, rank, self.inference_config.tp_size, "127.0.0.1", dist_group_port
            )
            for rank, worker in enumerate(self.workers)
        ]

        await asyncio.gather(*init_tasks)

    async def init_model(self, model_or_path: Union[nn.Module, str], model_policy: Policy = None):
        assert len(self.workers) == self.tp_size, "init workers first"

        inference_config_param = self.inference_config.to_rpc_param()
        model_path = model_or_path
        model_policy_param = model_policy.to_rpc_param() if model_policy else None

        init_tasks = [
            self.async_parallel_wrapper(worker.init_model, inference_config_param, model_path, model_policy_param)
            for rank, worker in enumerate(self.workers)
        ]

        await asyncio.gather(*init_tasks)

    def init_scheduler(self, inference_config: InferenceConfig, model_config: PretrainedConfig) -> RPCRequestHandler:
        return RPCRequestHandler(inference_config, model_config)

    async def _init_device_cache(self, alloc_shape: Tuple[int, int, int, int]):
        assert len(self.workers) == self.tp_size, "init workers first"

        init_tasks = [self.async_parallel_wrapper(worker.init_cache, alloc_shape) for worker in self.workers]

        await asyncio.gather(*init_tasks)

    def init_device_cache(self, alloc_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]):
        asyncio.run(self._init_device_cache(alloc_shape))

    def prepare_input(self, batch: BatchBucket) -> Tuple[List[int], InputMetaData]:
        input_ids = batch.get_1D_inputs()
        sequence_lengths = batch.get_sequence_lengths()

        if batch.is_prompts:
            n_tokens = sequence_lengths.sum().item()
        else:
            n_tokens = batch.current_batch_size
            if batch.use_spec_dec:
                n_tokens = batch.num_tokens_to_verify + 1
                assert n_tokens == input_ids.size(0)
                n_tokens = n_tokens * batch.current_batch_size

        batch_token_ids = None
        config_dict = self.generation_config.to_dict()
        # process repetition_penalty, no_repeat_ngram_size
        for type in ["repetition_penalty", "no_repeat_ngram_size"]:
            if type in config_dict and config_dict[type] is not None:
                batch_token_ids = batch.batch_token_ids

        # only when we have the graph for specific decoding batch size can we use the cuda graph for inference
        use_cuda_graph = False
        if self.use_cuda_graph and not batch.is_prompts and batch.current_batch_size in self.graph_runners.keys():
            use_cuda_graph = True

        input_meta_data = InputMetaData(
            block_tables=batch.get_block_table_tensor(),
            sequence_lengths=sequence_lengths,
            fd_inter_tensor=None,
            batch_size=batch.current_batch_size,
            is_prompts=batch.is_prompts,
            use_cuda_kernel=self.inference_config.use_cuda_kernel,
            use_cuda_graph=use_cuda_graph,
            high_precision=self.high_precision,
            kv_seq_len=sequence_lengths.max().item(),
            head_dim=batch.head_dim,
            dtype=batch.dtype,
            use_spec_dec=batch.use_spec_dec,
            num_tokens_to_verify=batch.num_tokens_to_verify,
            batch_token_ids=batch_token_ids,
        )

        return input_ids.tolist(), input_meta_data

    async def step_(self, input_token_ids, input_meta_data: InputMetaData):
        assert len(self.workers) == self.tp_size, "init workers first"

        init_tasks = [
            self.async_parallel_wrapper(
                worker.execute_model_forward,
                input_token_ids,
                input_meta_data.to_rpc_param(),
                self.generation_config_dict,
            )
            for worker in self.workers
        ]
        ret = await asyncio.gather(*init_tasks)

        return ret[0]

    def step(self) -> List[str]:
        batch = self.request_handler.schedule()

        input_token_ids, input_meta_data = self.prepare_input(batch)
        # TODO: padding_id is used for generating attn_mask and will be removed if nopad version is supported.
        next_tokens = asyncio.run(self.step_(input_token_ids, input_meta_data))

        # update the request_handler
        next_tokens = torch.tensor(next_tokens, dtype=torch.int)
        self.request_handler.append_next_tokens(next_tokens)
        finished_sequences = self.request_handler.update()
        return finished_sequences

    def kill_workers(self):
        """
        I don't find a good way to implicit invoke self.kill_workers
        """
        assert len(self.workers) != 0
        for proc in self.worker_processes:
            proc.kill()
            proc.join()
        self.logger.info(f"worker killed, serving end")

    def __del__(self):
        self.kill_workers()
