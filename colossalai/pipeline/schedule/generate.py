from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Union

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_map

from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.ppinference.microbatch_manager import MicroBatchManager
from colossalai.utils.cuda import get_current_device

from ._utils import detach, get_batch_size, get_micro_batch, merge_batch, model_forward, retain_grad, to_device
from .base import PipelineSchedule


class GenerateSchedule(PipelineSchedule):

    def __init__(self, stage_manager: PipelineStageManager, mb_manager: MicroBatchManager) -> None:
        super().__init__(stage_manager)
        self.comm = PipelineP2PCommunication(stage_manager)
        self.mb_manager = mb_manager
        self.microbatch_size = mb_manager.pp_inference_config.micro_batch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None
        self.num_microbatches: Optional[int] = None

    def load_batch(self, data_iter: Iterable, device: Optional[torch.device] = None) -> None:
        """Load a batch from data iterator.

        Args:
            data_iter (Iterable): Data iterator.
            device (Optional[torch.device], optional): Target device. Defaults to None.
        """
        batch = next(data_iter)
        if device is not None:
            batch = tree_map(partial(to_device, device=device), batch)
        self.batch = batch
        self.batch_size = get_batch_size(batch)
        self.microbatch_offset = 0
        assert self.batch_size % self.microbatch_size == 0, \
            f"Batch size should divided by the number of microbatches, {self.batch_size}, {self.num_microbatches}"
        self.num_microbatches = self.batch_size // self.microbatch_size

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_current_device()), micro_batch)

    @torch.no_grad()
    def generate_step(self,
                      model: Module,
                      data_iter: Iterable,
                      outputs: Optional[List[Any]] = None) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        output_sequence = []
        self.load_batch(data_iter)
        model.eval()

        # prepare for warmup
        num_warmup_microbatch = self.stage_manager.num_stages - self.stage_manager.stage
        num_warmup_microbatch = min(num_warmup_microbatch, self.num_microbatches)
        num_microbatch_remaining = self.num_microbatches - num_warmup_microbatch

        # run warmup round
        for _ in range(self.num_microbatches):
            micro_batch = None
            if self.stage_manager.is_first_stage():
                micro_batch = self.load_micro_batch()
            input_obj = self.comm.recv_forward()
            # if self.stage_manager.is_first_stage:
            # print("warmup",self.microbatch_offset,micro_batch, input_obj)
            hidden_states = None if input_obj is None else {'hidden_states': input_obj.get('hidden_states', None)}
            output_obj = model_forward(model, micro_batch, hidden_states)
            self.comm.send_forward(output_obj)
            if self.stage_manager.is_last_stage():
                output_sequence.append(output_obj['hidden_states'])

        return output_sequence
