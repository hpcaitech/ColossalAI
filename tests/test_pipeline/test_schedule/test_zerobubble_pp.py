from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import HybridParallelPlugin, MoeHybridParallelPlugin
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import OptimizerWrapper
from colossalai.logging import disable_existing_loggers
from colossalai.pipeline.schedule.v_schedule import PipelineGraph, ScheduledNode
from colossalai.pipeline.schedule.zero_bubble_pp import ZeroBubbleVPipeScheduler
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import assert_loose_close

NUM_BATCH = 8
NUM_TOK_PER_BATCH, NUM_EXPERTS = 4, 4
NUM_LAYERS = 8
HIDDEN_SIZE_PER_HEAD = 4
NUM_HEADS = 4
TOP_K = 1


class MlpModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_layers,
        stage_index=None,
        stage_mgr: PipelineStageManager = None,
    ):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(in_dim, out_dim, bias=None) for _ in range(num_layers)])

    def forward(
        self,
        data: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
        stage_index=None,
        stage_mgr: PipelineStageManager = None,
        model_chunk_id: int = None,
    ):
        if stage_mgr is None:
            hidden_states = data
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states
        else:
            # Set not used layer to None
            held_layers = self.layers[stage_index[0] : stage_index[1]]

            # fwd end
            if stage_mgr.is_first_stage() and stage_mgr.model_chunk_id == 1:
                return held_layers(hidden_states)
            # fwd start
            elif stage_mgr.is_first_stage() and stage_mgr.model_chunk_id == 0:
                return {"hidden_states": held_layers(data)}
            # fwd middle
            else:
                return {"hidden_states": held_layers(hidden_states)}

    def no_sync(self):
        return nullcontext()


def assert_optim_param_groups(optim_base_param_groups, optim_pp_param_groups):
    for (key_base, val_base), (key_pp, val_pp) in zip(optim_base_param_groups.items(), optim_pp_param_groups.items()):
        if key_base == key_pp:
            if key_base != "params":
                assert val_base == val_pp


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


# 1) Test manual v_schedule with multiple microbatch
@parameterize(
    "test_config",
    [
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
    ],
)
def run_fwd_bwd_iter_input(test_config):
    # init dist
    rank = dist.get_rank()
    pp_size = test_config["pp_size"]
    pg_mesh = ProcessGroupMesh(pp_size)
    num_microbatch = test_config["num_microbatches"]
    num_model_chunk = test_config["num_model_chunk"]
    # stage_manager
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=num_model_chunk
    )

    # schedule list
    zbv_schedule = [
        # stage 0
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=0, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=0, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=0, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=0, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=0, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=3),
        ],
        # stage 1
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=1, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=1, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=1, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=1, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=1, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=0, minibatch=3),
        ],
        # stage 2
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=2, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=2, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=2, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=2, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=2, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=2, minibatch=3),
        ],
        # stage 3
        [
            # microbatch 0
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=0),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=0),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=0),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=0),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=0),
            # microbatch 1
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=1),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=1),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=1),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=1),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=1),
            # microbatch 2
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=2),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=2),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=2),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=2),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=2),
            # microbatch 3
            # chunk 0 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="F", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=0, stage=3, minibatch=3),
            # chunk 1 fwd
            ScheduledNode(type="RECV_FORWARD", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="F", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="SEND_FORWARD", chunk=1, stage=3, minibatch=3),
            # chunk 1 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="B", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="W", chunk=1, stage=3, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=1, stage=3, minibatch=3),
            # chunk 0 bwd
            ScheduledNode(type="RECV_BACKWARD", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="B", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="W", chunk=0, stage=3, minibatch=3),
            ScheduledNode(type="SEND_BACKWARD", chunk=0, stage=3, minibatch=3),
        ],
    ]

    scheduler = ZeroBubbleVPipeScheduler(
        schedule=zbv_schedule,  # hint: send whole schedule or local schedule only ?
        stage_manager=stage_manager,
        num_model_chunks=pp_size,
        num_microbatch=num_microbatch,
        overlap_p2p=False,
    )

    # loss func
    def criterion(x, *args, **kwargs):
        return (x * x).mean()

    # init model and input
    batch_size = 4
    num_layers = 8
    in_dim = out_dim = 8
    print(f"Before init Model: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    data_iter = [torch.rand(batch_size, in_dim, out_dim, requires_grad=True).to(rank)]

    input_base = [t.clone() for t in data_iter]
    model_base = deepcopy(model)

    if rank == 0:
        # layer 0 & 7 to chunk 0 on rank0
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 0 or idx == 7:
                local_chunk.append(sub_model)
    elif rank == 1:
        # layer 1 & 6 to chunk 1 on rank1
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 1 or idx == 6:
                local_chunk.append(sub_model)
    elif rank == 2:
        # layer 2 & 5 to chunk 2 on rank2
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 2 or idx == 5:
                local_chunk.append(sub_model)
    else:
        # layer 3 & 4 to chunk 3 on rank3
        local_chunk = torch.nn.ModuleList().to(rank)
        for idx, sub_model in enumerate(model.layers):
            if idx == 3 or idx == 4:
                local_chunk.append(sub_model)
    # init optimizer
    optimizer_base = torch.optim.SGD(model_base.parameters(), lr=1e-5)
    optimizer_pp = OptimizerWrapper(torch.optim.SGD(local_chunk.parameters(), lr=1e-5))

    print(
        f"After init Model & input: {torch.cuda.memory_allocated()/1024**3 :.3f} GB on device {stage_manager.get_rank()};"
    )

    torch.cuda.synchronize()
    result = scheduler.forward_backward_step(
        model_chunk=local_chunk,
        data_iter=iter(data_iter),
        criterion=criterion,
        optimizer=optimizer_pp,
        return_loss=True,
        return_outputs=True,
    )

    optimizer_pp.step()

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    output_base = model_base(input_base[0])
    loss_base = criterion(output_base)
    loss_base.backward()
    optimizer_base.step()
    print(f"After base fwd & bwd: {torch.cuda.memory_allocated()/1024**3 :.3f} GB;")

    ##########################
    # assert weight
    ##########################
    if rank == 0:
        # layer 0
        assert_close(local_chunk[0].weight, model_base.layers[0].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[0].weight.grad)
        # layer 7
        assert_close(local_chunk[1].weight, model_base.layers[7].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[7].weight.grad)
    if rank == 1:
        # layer 1
        assert_close(local_chunk[0].weight, model_base.layers[1].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[1].weight.grad)
        # layer 6
        assert_close(local_chunk[1].weight, model_base.layers[6].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[6].weight.grad)
    if rank == 2:
        # layer 2
        assert_close(local_chunk[0].weight, model_base.layers[2].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[2].weight.grad)
        # layer 5
        assert_close(local_chunk[1].weight, model_base.layers[5].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[5].weight.grad)
    if rank == 3:
        # layer 3
        assert_close(local_chunk[0].weight, model_base.layers[3].weight)
        assert_close(local_chunk[0].weight.grad, model_base.layers[3].weight.grad)
        # layer 4
        assert_close(local_chunk[1].weight, model_base.layers[4].weight)
        assert_close(local_chunk[1].weight.grad, model_base.layers[4].weight.grad)


# 2) add optimizer base 1)
@parameterize(
    "test_config",
    [
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
        {
            "batch_size": 8,
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 8,
            "zero_stage": 1,
            "precision": "bf16",
            "num_model_chunk": 2,
        },
    ],
)
def run_fwd_bwd_vschedule_with_optim(test_config):
    # init dist
    rank = dist.get_rank()
    pp_size = test_config["pp_size"]
    pg_mesh = ProcessGroupMesh(pp_size)
    num_microbatch = test_config["num_microbatches"]
    num_model_chunk = test_config["num_model_chunk"]
    # stage_manager
    stage_manager = PipelineStageManager(
        pg_mesh, pipeline_axis=0, enable_interleave=True, num_model_chunks=num_model_chunk, use_zbv=True
    )

    h, a, s = 4096, 32, 1024
    mem_f = 34 * h + 5 * a * s
    mem_w = -32 * h
    mem_b = -mem_w - mem_f
    graph = PipelineGraph(
        n_stage=pp_size,
        n_micro=num_microbatch,
        f_cost=1,
        b_cost=1,
        w_cost=1,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
        # max_mem=mem_f * (p * 2 + m_offset),
    )

    zbv_schedule = graph.get_v_schedule()

    scheduler = ZeroBubbleVPipeScheduler(
        schedule=zbv_schedule,  # hint: send whole schedule or local schedule only ?
        stage_manager=stage_manager,
        num_model_chunks=num_model_chunk,
        num_microbatch=num_microbatch,
        overlap_p2p=False,
    )

    # init loss func
    def criterion(x, *args, **kwargs):
        x = x["hidden_states"]
        return (x * x).mean()

    def criterion_base(x, *args, **kwargs):
        return (x * x).mean()

    # init model and input
    batch_size = test_config["batch_size"]
    num_layers = 8
    assert num_layers % num_model_chunk == 0, f"Model with {num_layers} layer can not dist on {num_model_chunk} chunk"
    in_dim = out_dim = 1024
    before_init_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"Before init Model: {before_init_memory :.3f} GB on device {stage_manager.get_rank()};")
    model = MlpModel(in_dim=in_dim, out_dim=out_dim, num_layers=num_layers).to(rank)
    data_iter = {"data": torch.rand(batch_size, in_dim, out_dim, requires_grad=True).to(rank)}
    input_base = {k: v.clone() for k, v in data_iter.items()}
    model_base = deepcopy(model)
    model_pp = deepcopy(model)
    layers_per_stage = stage_manager.distribute_layers(len(model.layers))
    stage_manager.stage_indices = stage_manager.get_stage_index(layers_per_stage)

    model_pp._forward = model_pp.forward

    model_pp.forward = partial(model_pp._forward, stage_mgr=stage_manager)

    # init optimizer
    optimizer_base = torch.optim.SGD(model_base.parameters(), momentum=0.1, lr=1e-5)
    optimizer_pp = OptimizerWrapper(torch.optim.SGD(model_pp.parameters(), momentum=0.1, lr=1e-5))

    after_init_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"After init Model & input: {after_init_memory :.5f} GB on device {stage_manager.get_rank()};")

    torch.cuda.synchronize()
    result = scheduler.forward_backward_step(
        model_chunk=model_pp,
        data_iter=iter([data_iter]),
        criterion=criterion,
        optimizer=optimizer_pp,
        return_loss=True,
        return_outputs=True,
    )

    optimizer_pp.step()

    after_pp_step_memory = torch.cuda.memory_allocated() / 1024**3

    # assert memory
    if rank != 0:
        # w.grad: hid_dim * hid_dim * microbatch * 4(fp32) * 2 (2 layer in each stage) / 1024**3
        # output: hid_dim * hid_dim * microbatch * 4(fp32) / 1024**3
        # optim: state hid_dim * hid_dim * 4(fp32) * 2 (2 layer in each stage) / 1024**3
        print(
            f" num_microbatch {num_microbatch} rank {rank}: {(after_pp_step_memory - after_init_memory)} <= {(in_dim * in_dim * 4 * 5 * batch_size / 1024**3)}"
        )
        assert (after_pp_step_memory - after_init_memory) <= (in_dim * in_dim * 4 * 5 * batch_size / 1024**3)
    else:
        # rank0 will also hold output;
        print(
            f" num_microbatch {num_microbatch} rank {rank}: {round((after_pp_step_memory - after_init_memory), 5)} <= {round((in_dim * in_dim * 4 * 5 * batch_size / 1024**3 + batch_size * in_dim * in_dim * 4 / 1024**3), 5)}"
        )
        assert round((after_pp_step_memory - after_init_memory), 5) <= round(
            (in_dim * in_dim * 4 * 5 * batch_size / 1024**3 + batch_size * in_dim * in_dim * 4 / 1024**3), 5
        )

    ##########################
    # Fwd bwd for base
    ##########################
    # fwd & bwd
    # output_base = model_base(input_base["data"])
    output_base = model_base.forward(data=input_base["data"])
    loss_base = criterion_base(output_base)
    loss_base.backward()
    optimizer_base.step()

    ##########################
    # assert loss & output
    ##########################
    # only chunk 1 stage 0 hold loss and output
    if rank == 0:
        assert_close(result["loss"], loss_base)
        assert_close(result["outputs"]["hidden_states"], output_base)

    # ##########################
    # # assert weight & optim state
    # ##########################
    optim_base_state = optimizer_base.state_dict()["state"]
    optim_pp_state = optimizer_pp.state_dict()["state"]
    optim_base_param_groups = optimizer_base.state_dict()["param_groups"][0]
    optim_pp_param_groups = optimizer_pp.state_dict()["param_groups"][0]

    if rank == 0:
        # layer 0
        assert_close(model_pp.layers[0].weight, model_base.layers[0].weight)
        assert_close(model_pp.layers[0].weight.grad, model_base.layers[0].weight.grad)
        assert_close(optim_pp_state[0]["momentum_buffer"], optim_base_state[0]["momentum_buffer"])
        # layer 7
        assert_close(model_pp.layers[7].weight, model_base.layers[7].weight)
        assert_close(model_pp.layers[7].weight.grad, model_base.layers[7].weight.grad)
        assert_close(optim_pp_state[7]["momentum_buffer"], optim_base_state[7]["momentum_buffer"])
    if rank == 1:
        # layer 1
        assert_close(model_pp.layers[1].weight, model_base.layers[1].weight)
        assert_close(model_pp.layers[1].weight.grad, model_base.layers[1].weight.grad)
        assert_close(optim_pp_state[1]["momentum_buffer"], optim_base_state[1]["momentum_buffer"])
        # layer 6
        assert_close(model_pp.layers[6].weight, model_base.layers[6].weight)
        assert_close(model_pp.layers[6].weight.grad, model_base.layers[6].weight.grad)
        assert_close(optim_pp_state[6]["momentum_buffer"], optim_base_state[6]["momentum_buffer"])
    if rank == 2:
        # layer 2
        assert_close(model_pp.layers[2].weight, model_base.layers[2].weight)
        assert_close(model_pp.layers[2].weight.grad, model_base.layers[2].weight.grad)
        assert_close(optim_pp_state[2]["momentum_buffer"], optim_base_state[2]["momentum_buffer"])
        # layer 5
        assert_close(model_pp.layers[5].weight, model_base.layers[5].weight)
        assert_close(model_pp.layers[5].weight.grad, model_base.layers[5].weight.grad)
        assert_close(optim_pp_state[5]["momentum_buffer"], optim_base_state[5]["momentum_buffer"])
    if rank == 3:
        # layer 3
        assert_close(model_pp.layers[3].weight, model_base.layers[3].weight)
        assert_close(model_pp.layers[3].weight.grad, model_base.layers[3].weight.grad)
        assert_close(optim_pp_state[3]["momentum_buffer"], optim_base_state[3]["momentum_buffer"])
        # layer 4
        assert_close(model_pp.layers[4].weight, model_base.layers[4].weight)
        assert_close(model_pp.layers[4].weight.grad, model_base.layers[4].weight.grad)
        assert_close(optim_pp_state[4]["momentum_buffer"], optim_base_state[4]["momentum_buffer"])

    # assert optim param_groups
    assert_optim_param_groups(optim_base_param_groups, optim_pp_param_groups)


@parameterize(
    "config",
    [
        (1, 2, 1, 1, 2),
        (1, 1, 2, 2, 1),
        (1, 2, 1, 2, 1),
        (1, 2, 2, 1, 1),
        (1, 1, 4, 1, 1),
    ],
)
def run_with_booster_moehybridplugin(config: Tuple[int, ...]):
    stage, ep_size, pp_size, tp_size, sp_size = config
    num_microbatches = pp_size
    dist.get_world_size()
    rank = dist.get_rank()
    dtype, precision = torch.float16, "fp16"
    torch.cuda.set_device(dist.get_rank())

    ########
    # init base model
    ########
    assert pp_size <= NUM_LAYERS, "pp_size should be less than or equal to NUM_LAYERS"
    config = MixtralConfig(
        hidden_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
        intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        attn_implementation="flash_attention_2",
    )

    # init model with the same seed
    seed_all(10086)

    torch_model = MixtralModel(config).to(dtype).cuda()
    # TODO: Support MixtralForCausalLM
    # torch_model = MixtralForCausalLM(config).to(dtype).cuda()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)
    # init schedule
    h, a, s = config.hidden_size, config.num_attention_heads, 1024
    mem_f = 34 * h + 5 * a * s
    mem_w = -32 * h
    mem_b = -mem_w - mem_f
    graph = PipelineGraph(
        n_stage=pp_size,
        n_micro=num_microbatches,
        f_cost=1,
        b_cost=1,
        w_cost=1,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
        # max_mem=mem_f * (p * 2 + m_offset),
    )

    zbv_schedule = graph.get_v_schedule()

    # init MoeHybridPlugin
    plugin = MoeHybridParallelPlugin(
        pp_size=pp_size,
        num_microbatches=pp_size,
        tp_size=tp_size,
        sp_size=sp_size,
        ep_size=ep_size,
        zero_stage=stage,
        enable_sequence_parallelism=sp_size > 1,
        sequence_parallelism_mode="all_to_all" if sp_size > 1 else None,
        overlap_communication=False,
        initial_scale=1,
        precision=precision,
        find_unused_parameters=True,
        pp_style="zbv",
        scheduler_nodes=zbv_schedule,
        num_model_chunks=2,
    )

    dp_size = plugin.dp_size

    booster = Booster(plugin=plugin)

    ########
    # init pp model
    ########

    parallel_model = deepcopy(torch_model)
    parallel_optimizer = torch.optim.SGD(parallel_model.parameters(), lr=1)
    parallel_model, parallel_optimizer, _, _, _ = booster.boost(parallel_model, parallel_optimizer)
    # create different input along dp axis
    seed_all(1453 + rank)

    torch_model.train()
    parallel_model.train()
    for _ in range(2):
        # gen random input
        input_embeddings = torch.rand(
            NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
        ).cuda()
        dist.all_reduce(
            input_embeddings, group=plugin.pp_group
        )  # pp inputs except the first stage doesn't matter, but need to be replicate for torch model check

        dist.all_reduce(input_embeddings, group=plugin.tp_group)  # tp group duplicate input
        dist.all_reduce(input_embeddings, group=plugin.sp_group)  # sp group duplicate input

        # run the model with hybrid parallel
        if booster.plugin.stage_manager is not None:
            # for test with pp
            data_iter = iter([{"inputs_embeds": input_embeddings}])
            sharded_output = booster.execute_pipeline(
                data_iter,
                parallel_model,
                lambda x, y: x.last_hidden_state.mean(),
                parallel_optimizer,
                return_loss=True,
                return_outputs=True,
            )
            # stage 0 chunk 0
            if (
                booster.plugin.stage_manager.is_first_stage(ignore_chunk=True)
                and rank == dist.get_process_group_ranks(plugin.pp_group)[0]
            ):
                parallel_output = sharded_output["loss"]
            else:
                parallel_output = torch.tensor(12345.0, device="cuda")
            # broadcast along pp axis
            dist.broadcast(parallel_output, src=dist.get_process_group_ranks(plugin.pp_group)[0], group=plugin.pp_group)

        else:
            # for test without pp
            parallel_output = parallel_model(inputs_embeds=input_embeddings.to(dtype)).last_hidden_state.mean()
            parallel_optimizer.backward(parallel_output)
        parallel_optimizer.step()
        parallel_optimizer.zero_grad()
        dist.all_reduce(parallel_output, group=plugin.dp_group)

        # ===================================================================================
        # run normal model with all dp(different) inputs
        all_inputs = [input_embeddings.clone() for _ in range(dp_size)]
        dist.all_gather(all_inputs, input_embeddings, group=plugin.dp_group)
        torch_output_sum = 0
        for input_data_ in all_inputs:
            torch_output = torch_model(inputs_embeds=input_data_.to(dtype)).last_hidden_state.mean()
            torch_output.backward()
            torch_output_sum += torch_output.detach()
        # avg dp grads follows zero optimizer
        for p in torch_model.parameters():
            if p.grad is not None:
                p.grad /= dp_size
        torch_optimizer.step()
        torch_optimizer.zero_grad()
        assert_loose_close(parallel_output, torch_output_sum, dtype=dtype)
    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


@parameterize(
    "config",
    [
        # Pass
        (1, 2, 2, 1),
        (1, 2, 1, 2),
        (1, 1, 2, 2),
        (1, 4, 1, 1),
    ],
)
def run_with_booster_hybridplugin(config: Tuple[int, ...]):
    stage, pp_size, tp_size, sp_size = config
    num_microbatches = pp_size
    dist.get_world_size()
    rank = dist.get_rank()
    dtype, precision = torch.float16, "fp16"
    torch.cuda.set_device(dist.get_rank())

    ########
    # init base model
    ########
    assert pp_size <= NUM_LAYERS, "pp_size should be less than or equal to NUM_LAYERS"
    config = LlamaConfig(
        hidden_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
        intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        attn_implementation="flash_attention_2",
    )

    # init model with the same seed
    seed_all(10086)

    torch_model = LlamaModel(config).to(dtype).cuda()
    # TODO: Support MixtralForCausalLM
    # torch_model = MixtralForCausalLM(config).to(dtype).cuda()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)
    # init schedule
    h, a, s = config.hidden_size, config.num_attention_heads, 1024
    mem_f = 34 * h + 5 * a * s
    mem_w = -32 * h
    mem_b = -mem_w - mem_f
    graph = PipelineGraph(
        n_stage=pp_size,
        n_micro=num_microbatches,
        f_cost=1,
        b_cost=1,
        w_cost=1,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
    )

    zbv_schedule = graph.get_v_schedule()

    # init HybridParallelPlugin
    plugin = HybridParallelPlugin(
        pp_size=pp_size,
        num_microbatches=pp_size,
        tp_size=tp_size,
        sp_size=sp_size,
        zero_stage=stage,
        enable_sequence_parallelism=sp_size > 1,
        sequence_parallelism_mode="all_to_all" if sp_size > 1 else None,
        overlap_communication=False,
        initial_scale=1,
        precision=precision,
        find_unused_parameters=True,
        pp_style="zbv",
        scheduler_nodes=zbv_schedule,
        num_model_chunks=2,
    )

    dp_size = plugin.dp_size

    booster = Booster(plugin=plugin)

    ########
    # init pp model
    ########

    parallel_model = deepcopy(torch_model)
    parallel_optimizer = torch.optim.SGD(parallel_model.parameters(), lr=1)
    parallel_model, parallel_optimizer, _, _, _ = booster.boost(parallel_model, parallel_optimizer)
    # create different input along dp axis
    seed_all(1453 + rank)

    torch_model.train()
    parallel_model.train()
    for _ in range(2):
        # gen random input
        input_embeddings = torch.rand(
            NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
        ).cuda()
        dist.all_reduce(
            input_embeddings, group=plugin.pp_group
        )  # pp inputs except the first stage doesn't matter, but need to be replicate for torch model check

        dist.all_reduce(input_embeddings, group=plugin.tp_group)  # tp group duplicate input
        dist.all_reduce(input_embeddings, group=plugin.sp_group)  # sp group duplicate input

        # run the model with hybrid parallel
        if booster.plugin.stage_manager is not None:
            # for test with pp
            data_iter = iter([{"inputs_embeds": input_embeddings}])
            sharded_output = booster.execute_pipeline(
                data_iter,
                parallel_model,
                lambda x, y: x.last_hidden_state.mean(),
                parallel_optimizer,
                return_loss=True,
                return_outputs=True,
            )
            # stage 0 chunk 0
            if (
                booster.plugin.stage_manager.is_first_stage(ignore_chunk=True)
                and rank == dist.get_process_group_ranks(plugin.pp_group)[0]
            ):
                parallel_output = sharded_output["loss"]
            else:
                parallel_output = torch.tensor(12345.0, device="cuda")
            # broadcast along pp axis
            dist.broadcast(parallel_output, src=dist.get_process_group_ranks(plugin.pp_group)[0], group=plugin.pp_group)

        else:
            # for test without pp
            parallel_output = parallel_model(inputs_embeds=input_embeddings.to(dtype)).last_hidden_state.mean()
            parallel_optimizer.backward(parallel_output)
        parallel_optimizer.step()
        parallel_optimizer.zero_grad()
        dist.all_reduce(parallel_output, group=plugin.dp_group)

        # ===================================================================================
        # run normal model with all dp(different) inputs
        all_inputs = [input_embeddings.clone() for _ in range(dp_size)]
        dist.all_gather(all_inputs, input_embeddings, group=plugin.dp_group)
        torch_output_sum = 0
        for input_data_ in all_inputs:
            torch_output = torch_model(inputs_embeds=input_data_.to(dtype)).last_hidden_state.mean()
            torch_output.backward()
            torch_output_sum += torch_output.detach()
        # avg dp grads follows zero optimizer
        for p in torch_model.parameters():
            if p.grad is not None:
                p.grad /= dp_size
        torch_optimizer.step()
        torch_optimizer.zero_grad()
        assert_loose_close(parallel_output, torch_output_sum, dtype=dtype)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_with_booster_moehybridplugin()
    run_with_booster_hybridplugin()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_pp():
    spawn(
        run_dist,
        nprocs=4,
    )


# python -m pytest -s tests/test_pipeline/test_schedule/test_zerobubble_pp.py
if __name__ == "__main__":
    test_pp()
