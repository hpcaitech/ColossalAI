from tensorrt_llm.network import Network
import argparse
import tensorrt as trt
from pathlib import Path
import torch.nn as nn
from .args_utils import BuilderArgsConfig
from tensorrt_llm.logger import logger
from .utils import to_onnx, get_engine_name, serialize_engine

class EngineBuilderBase:
    
    def __init__(self):
        self._builder_args_config: BuilderArgsConfig = None
        self._builder: Builder = None
        self._builder_config: tensorrt_llm.builder.BuilderConfig = None
        self._trt_model: nn.Module = None
        self._network: Network = None
        self._engine_name: str = None
        
    def set_config(self, config: BuilderArgsConfig):
        self._builder_args_config = config
        
    def _process_config(self) -> None:
        pass
    
    def _generate_network(self) -> None:
        pass
    
    def _get_model(self) -> None:
        pass
    
    def _get_builder_config(self) -> None:
        pass
         
    def _build_rank_engine(self, rank) -> trt.IHostMemory:
        # Get trt_model
        self._get_model()
        with net_guard(self._network):
            # Prepare
            self._network.set_named_parameters(self._trt_model.named_parameters())
    
            # Forward
            inputs = self._trt_model.prepare_inputs(self._builder_args_config.max_batch_size,
                                              self._builder_args_config.max_input_len,
                                              self._builder_args_config.max_output_len,
                                              True,
                                              self._builder_args_config.max_beam_width,
                                              self._builder_args_config.max_num_tokens)
            self._trt_model(*inputs)
            if self._builder_args_config.enable_debug_output:
                # mark intermediate nodes' outputs
                for k, v in self._trt_model.named_network_outputs():
                    v = v.trt_tensor
                    v.name = k
                    self._network.trt_network.mark_output(v)
                    v.dtype = dtype
            if self._builder_args_config.visualize:
                model_path = os.path.join(self._builder_args_config.output_dir, 'test.onnx')
                to_onnx(self._network.trt_network, model_path)
    
        tensorrt_llm.graph_rewriting.optimize(self._network)
    
        engine = None
    
        # Network -> Engine
        engine = self._builder.build_engine(self._network, self._builder_config)
        if rank == 0:
            config_path = os.path.join(self._builder_args_config.output_dir, 'config.json')
            self._builder.save_config(self._builder_config, config_path)
        return engine
    
    def build(self, model_name: str, rank: int) -> None:
        
        # Preparatory work
        self._process_config()        
        
        torch.cuda.set_device(rank % self._builder_args_config.gpus_per_node)
        logger.set_level(self._builder_args_config.log_level)
        if not os.path.exists(self._builder_args_config.output_dir):
            os.makedirs(self._builder_args_config.output_dir)
        
        cache = None
        for cur_rank in range(self._builder_args_config.world_size):
            # skip other ranks if parallel_build is enabled
            if self._builder_args_config.parallel_build and cur_rank != rank:
                continue
            self._builder_config = self._get_builder_config()
            self._engine_name = get_engine_name(model_name, self._builder_args_config.dtype, self._builder_args_config.tp_size,
                                               self._builder_args_config.pp_size, cur_rank)
            
            # generate Network
            self._generate_network()
            
            engine = self._build_rank_engine(cur_rank)
            assert engine is not None, f'Failed to build engine for rank {cur_rank}'
    
            if cur_rank == 0:
                # Use in-memory timing cache for multiple builder passes.
                if not self._builder_args_config.parallel_build:
                    cache = self._builder_config.trt_builder_config.get_timing_cache()
    
            serialize_engine(engine, os.path.join(self._builder_args_config.output_dir, engine_name))
    
        if rank == 0:
            ok = self._builder.save_timing_cache(
                self._builder_config, os.path.join(args.output_dir, "model.cache"))
            assert ok, "Failed to save timing cache."