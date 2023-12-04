import os
from typing import Tuple

import tensorrt as trt
import tensorrt_llm
import torch
import torch.nn as nn
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import Network, net_guard

from colossalai.inference.trtllm.args_utils import BuilderArgsConfig
from colossalai.inference.trtllm.utils import get_engine_name, save_config, serialize_engine, to_onnx


class EngineBuilderBase:
    def __init__(self):
        self._builder_args_config: BuilderArgsConfig = None
        self._builder: Builder = None
        self._trt_model: nn.Module = None
        self._network: Network = None
        self._engine_name: str = None

    def set_config(self, config: BuilderArgsConfig):
        """
        Before generating the engine, you need to initialize builder_args_config first.
        """

        self._builder_args_config = config

    def _process_config(self) -> None:
        """
        After initializing builder_args_config,
        we need to implement the codes to process configs according to the characteristics of the specific model.
        """

    def _generate_network(self, rank: int) -> None:
        """
        In this interface, We need to implement the codes to create network and set plugins for the specific model.
        """

    def _set_model(self, rank: int) -> None:
        """
        In this interface, We need to initialize trt module for the specific model.
        """

    def _get_builder_config(self, cache=None) -> tensorrt_llm.builder.BuilderConfig:
        """
        In this interface, We need to implement the codes to create and return builder_config for the specific model.
        """

    def _build_rank_engine(
        self, rank, builder_config: tensorrt_llm.builder.BuilderConfig
    ) -> Tuple[trt.IHostMemory, dict]:
        # Get trt_model
        self._set_model(rank)
        with net_guard(self._network):
            # Prepare
            self._network.set_named_parameters(self._trt_model.named_parameters())

            # Forward
            inputs = self._trt_model.prepare_inputs(
                self._builder_args_config.max_batch_size,
                self._builder_args_config.max_input_len,
                self._builder_args_config.max_output_len,
                True,
                self._builder_args_config.max_beam_width,
                self._builder_args_config.max_num_tokens,
            )
            self._trt_model(*inputs)
            if self._builder_args_config.enable_debug_output:
                # mark intermediate nodes' outputs
                for k, v in self._trt_model.named_network_outputs():
                    v = v.trt_tensor
                    v.name = k
                    self._network.trt_network.mark_output(v)
                    v.dtype = dtype
            if self._builder_args_config.visualize:
                model_path = os.path.join(self._builder_args_config.output_dir, "test.onnx")
                to_onnx(self._network.trt_network, model_path)

        tensorrt_llm.graph_rewriting.optimize(self._network)

        engine = None

        # Network -> Engine
        engine = self._builder.build_engine(self._network, builder_config)
        config_path = os.path.join(self._builder_args_config.output_dir, "config.json")
        engine_config = save_config(builder_config, config_path, self._builder_args_config.save_engine, rank)
        return engine, engine_config

    def build(self, rank: int, model_name: str) -> Tuple[bytearray, dict]:
        # Preparatory work
        self._process_config()

        torch.cuda.set_device(rank % self._builder_args_config.gpus_per_node)
        logger.set_level(self._builder_args_config.log_level)
        if not os.path.exists(self._builder_args_config.output_dir):
            os.makedirs(self._builder_args_config.output_dir)

        # when doing serializing build, all ranks share one engine
        self._builder = Builder()

        cache = None
        for cur_rank in range(self._builder_args_config.world_size):
            # skip other ranks if parallel_build is enabled
            if self._builder_args_config.parallel_build and cur_rank != rank:
                continue
            builder_config = self._get_builder_config(cache)
            self._engine_name = get_engine_name(
                model_name,
                self._builder_args_config.dtype,
                self._builder_args_config.tp_size,
                self._builder_args_config.pp_size,
                cur_rank,
            )

            # generate Network
            self._generate_network(rank)

            engine, engine_config = self._build_rank_engine(cur_rank, builder_config)
            assert engine is not None, f"Failed to build engine for rank {cur_rank}"

            if cur_rank == 0:
                # Use in-memory timing cache for multiple builder passes.
                if not self._builder_args_config.parallel_build:
                    cache = builder_config.trt_builder_config.get_timing_cache()

            byte_engine = serialize_engine(
                engine,
                os.path.join(self._builder_args_config.output_dir, self._engine_name),
                self._builder_args_config.save_engine,
            )

        if rank == 0 and self._builder_args_config.save_engine:
            ok = self._builder.save_timing_cache(
                builder_config, os.path.join(self._builder_args_config.output_dir, "model.cache")
            )
            assert ok, "Failed to save timing cache."

        return byte_engine, engine_config
