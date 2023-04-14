# WIP


from coati.trainer.strategies import Strategy
from coati.trainer.strategies import NaiveStrategy
from coati.models.base import Actor, RewardModel, Critic

import numpy as np
import torch
from torch._C._distributed_rpc import _is_current_rpc_agent_set

import colossalai
from colossalai.pipeline.pipeline_process_group import ppg
from colossalai.pipeline.rpc._pipeline_schedule import OneFOneBPipelineEngine
from colossalai.fx import ColoTracer
from colossalai.fx.passes.adding_split_node_pass import balanced_split_pass, split_with_split_nodes_pass
from colossalai.pipeline.middleware.adaptor import get_fx_topology


import os
from functools import partial
import random

rpc_is_initialized = _is_current_rpc_agent_set

class PipelineModel(torch.nn.Module):
    '''
    Actor has 2 kinds of jobs: forward and generate. 
        better to just pipelinize the inner model
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 stage_num: int,
                 num_microbatches: int,
                 data_kwargs = None,
                 ):
        super().__init__()
        # create partition module
        def create_partition_module(pp_rank:int, stage_num: int, model, data_kwargs):
            model.eval()
            tracer = ColoTracer()
            meta_args = {k: v.to('meta') for k, v in data_kwargs.items()}
            graph = tracer.trace(root=model, meta_args=meta_args)
            gm = torch.fx.GraphModule(model, graph, model.__class__.__name__)
            annotated_model = balanced_split_pass(gm, stage_num)
            top_module, split_submodules = split_with_split_nodes_pass(annotated_model, merge_output=True)
            topo = get_fx_topology(top_module)
            for submodule in split_submodules:
                if isinstance(submodule, torch.fx.GraphModule):
                    setattr(submodule, '_topo', topo)
            return split_submodules[pp_rank + 1]
    
        def partition(model, data_kwargs: dict, pp_rank: int, chunk: int, stage_num: int):
            partition = create_partition_module(pp_rank, stage_num, model, data_kwargs)
            return partition
        self.inference_engine = OneFOneBPipelineEngine(
            partition_fn=partial(partition, model, data_kwargs),
            stage_num=stage_num,
            num_microbatches=num_microbatches,
            device='cuda',
        )

    def forward(self,
                **model_inputs):
        return self.inference_engine.forward_backward(**model_inputs, forward_only=True)



class PPStrategy(NaiveStrategy):
    """
        Strategy for Pipeline inference (inference only!)
        
        master node only
    """
    def __init__(
        self,
        seed: int = 42
    ):
        self.seed = seed
        super().__init__()
        
        
    def setup_distributed(self) -> None:
        colossalai.launch_from_torch({}, seed=self.seed)
        ppg.set_global_info(rank = int(os.environ['RANK']),
                            world_size=int(os.environ['WORLD_SIZE']),
                            dp_degree=1,
                            tp_degree=1,
                            num_worker_threads=128,
                            device="cuda")
        
    def model_init_context(self):
        return super().model_init_context()
    
    def setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, Actor) or \
            isinstance(model, RewardModel) or \
            isinstance(model, Critic):
            model.model = PipelineModel(model.model)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

