import re
from functools import partial
from types import MethodType
from typing import Callable, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.schedule.generate import GenerateSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_

from .inference_config import InferenceConfig
from .microbatch_manager import MicroBatchManager
from .utils import get_suffix_name, set_tensors_to_none


class PPInferEngine:

    def __init__(
        self,
        gerneration_config: InferenceConfig,
        model: nn.Module,
        pp_fwd: Callable,
    ) -> None:
        self.gerneration_config = gerneration_config
        self.model = model
        self.pg_mesh = ProcessGroupMesh(gerneration_config.pp_size)
        self.stage_manager = PipelineStageManager(self.pg_mesh, 0, True)
        self.held_layer = None
        self.mb_manager = MicroBatchManager(self.gerneration_config)
        self.schedule = GenerateSchedule(self.stage_manager, self.mb_manager)

        self._partion_model()
        self._inject_fwd(pp_fwd)

    def inference(self, input_list):
        out = self.schedule.generate_step(self.model, iter(input_list))
        return out
        # print(out)

    def _inject_fwd(self, pp_fwd: Callable):
        stage_index = self._get_stage_index()
        print(stage_index)
        new_fwd = partial(pp_fwd, stage_manager=self.stage_manager, stage_index=stage_index)
        bound_method = MethodType(new_fwd, self.model)
        setattr(self.model, 'forward', bound_method)

    def _partion_model(self):
        # get module list
        module_list = self._recursive_partion(self.model, [], '')

        # allocate module to each stage
        module_num = len(module_list)
        stage_size = self.gerneration_config.pp_size
        stage = self.stage_manager.stage
        start = stage * (module_num // stage_size) + min(stage, module_num % stage_size)
        end = start + (module_num // stage_size) + (1 if stage < module_num % stage_size else 0)
        self.held_layer = module_list[start:end]

        # release layers dose not belong to current stage
        self._release_unheld_layers(module_list)

        # load model to cuda
        self.model = self.model.cuda()
        print(dist.get_rank(), torch.cuda.memory_allocated())

    def _recursive_partion(self, module: nn.Module, module_list: List[str], suffix: str):
        if len(list(module.children())) == 0:
            module_list.append(suffix)
        for name, child in module.named_children():
            suffix_name = get_suffix_name(suffix, name)
            if child.__class__.__name__ in self.gerneration_config.stage_unit:
                module_list.append(suffix_name)
            else:
                self._recursive_partion(child, module_list, suffix_name)
        return module_list

    def _partion_batch(self):
        pass

    def _release_unheld_layers(self, module_list: List[str]):
        r"""
        Release the unheld layers in the model
        """
        held_layers = self.held_layer
        set_tensors_to_none(self.model, include=set(module_list) - set(held_layers))

    def _get_stage_index(self):
        re_pattern = r'\[\d+\]'
        prog = re.compile(re_pattern)
        stage_idx = []
        for item in self.held_layer:
            result = prog.search(item)
            if result:
                idx = result.group().replace('[', '').replace(']', '')
                stage_idx.append(int(idx))

        return [min(stage_idx), max(stage_idx) + 1]
