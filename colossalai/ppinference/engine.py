import re
from functools import partial
from types import MethodType
from typing import Callable, List, Optional, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.schedule.generate import GenerateSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer._utils import getattr_

from .inference_config import InferenceConfig
from .microbatch_manager import MicroBatchManager
from .policy.gpt2_ppinfer import GPT2LMHeadModelPipelinePolicy
from .utils import get_suffix_name, set_tensors_to_none


class PPInferEngine:

    def __init__(
        self,
        gerneration_config: InferenceConfig,
        model: nn.Module,
        pp_fwd: Callable,
    ) -> None:
        self.gerneration_config = gerneration_config
        self.pg_mesh = ProcessGroupMesh(gerneration_config.pp_size)
        self.stage_manager = PipelineStageManager(self.pg_mesh, 0, True)
        self.mb_manager = MicroBatchManager(self.gerneration_config)
        self.schedule = GenerateSchedule(self.stage_manager, self.mb_manager)
        self.model = self._shardformer(model)

    def inference(self, input_list):
        out = self.schedule.generate_step(self.model, iter(input_list))
        return out

    def _shardformer(self, model):
        shardconfig = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=False,
            enable_fused_normalization=False,
            enable_all_optimization=False,
            enable_flash_attention=False,
            enable_jit_fused=False,
            enable_sequence_parallelism=False,
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, GPT2LMHeadModelPipelinePolicy())
        return shard_model.cuda()
