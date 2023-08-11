from typing import List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_

from .inference_config import InferenceConfig
from .utils import get_suffix_name, set_tensors_to_none


class PPInferEngine:

    def __init__(
        self,
        gerneration_config: InferenceConfig,
        model: nn.Module,
    ) -> None:
        self.gerneration_config = gerneration_config
        self.model = model
        self.pg_mesh = ProcessGroupMesh(gerneration_config.pp_size)
        self.stage_manager = PipelineStageManager(self.pg_mesh, 0, True)
        self._stage_to_module = {}

        self._partion_model()

    def _partion_model(self):
        # get module list
        self.module_list = self._recursive_partion(self.model, [], '')

        # allocate module to each stage
        module_num = len(self.module_list)
        stage_size = self.gerneration_config.pp_size
        for stage in range(stage_size):
            start = stage * (module_num // stage_size) + min(stage, module_num % stage_size)
            end = start + (module_num // stage_size) + (1 if stage < module_num % stage_size else 0)
            self._stage_to_module[stage] = self.module_list[start:end]

        # release layers dose not belong to current stage
        self._release_unheld_layers()

        # load model to cuda
        print(dist.get_rank(), torch.cuda.memory_allocated())
        self.model = self.model.cuda()
        print(dist.get_rank(), torch.cuda.memory_allocated())

    def _recursive_partion(self, module: nn.Module, module_list: List[str], suffix: str):
        for name, child in module.named_children():
            suffix_name = get_suffix_name(suffix, name)
            if child.__class__.__name__ in self.gerneration_config.stage_unit:
                module_list.append(suffix_name)
            else:
                self._recursive_partion(child, module_list, suffix_name)
        return module_list

    def _release_unheld_layers(self):
        r"""
        Release the unheld layers in the model
        """
        held_layers = self._stage_to_module[self.stage_manager.stage]
        set_tensors_to_none(self.model, include=set(self.module_list) - set(held_layers))
