from functools import partial
from typing import Any, Dict, Iterable, Optional, Union

import torch
import torch.cuda
from torch.nn import Module
from torch.utils._pytree import tree_map

from colossalai.inference.pipeline.microbatch_manager import MicroBatchManager, Status
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.utils.cuda import get_current_device

from ._utils import get_batch_size, get_micro_batch, model_forward, to_device
from .base import PipelineSchedule


class GenerateSchedule(PipelineSchedule):
    '''
    GenerateSchedule is a class that handles the pipeline parallel inference.
    In our schedule, we place tie weight layer, embedding and lm_head in the same device to save space, so in
    this schedule, the out for each encoding progress is on rank0.

    Args:
        stage_manager (PipelineStageManager): Pipeline stage manager.
        mb_manager (MicroBatchManager): Micro batch manager.
    '''

    def __init__(self, stage_manager: PipelineStageManager, mb_manager: MicroBatchManager) -> None:
        super().__init__(stage_manager)
        self.comm = PipelineP2PCommunication(stage_manager)
        self.mb_manager = mb_manager
        self.microbatch_size = mb_manager.micro_batch_size
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
        self.round = self.num_microbatches // self.stage_manager.num_stages

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.

        Returns:
            Any: Micro batch.
        """
        micro_batch = get_micro_batch(self.batch, self.microbatch_offset, self.microbatch_size)
        self.microbatch_offset += self.microbatch_size
        return tree_map(partial(to_device, device=get_current_device()), micro_batch)

    def _prepare_stage_inputs(self):
        # first stage and in prefill phase
        if self.stage_manager.is_first_stage() and self.mb_manager.cur_state is Status.PREFILL:
            pre_stage_out = None
            model_inputs = self.load_micro_batch()
            hidden_states = None
        # first stage and in generate phase
        elif self.stage_manager.is_first_stage():
            pre_stage_out = self.comm.recv_forward()
            model_inputs = self._prepare_next_token(pre_stage_out)
            hidden_states = None
        # not first stage and in gererate phase
        else:
            pre_stage_out = self.comm.recv_forward()
            model_inputs = {
                'past_key_values': self.mb_manager.cur_kv_cache
            } if self.mb_manager.cur_kv_cache is not None else None
            hidden_states = pre_stage_out
        return pre_stage_out, model_inputs, hidden_states

    def _prepare_next_token(self, inputs: Dict[str, torch.Tensor]):
        new_mask = self.mb_manager.cur_descrption.attn_mask
        new_mask = torch.cat((new_mask, torch.ones((new_mask.shape[0], 1), dtype=torch.int64, device='cuda')), dim=-1)
        self.mb_manager.cur_descrption.attn_mask = new_mask
        past_key_values = self.mb_manager.cur_descrption.kv_cache

        return dict(input_ids=inputs['new_token'], attention_mask=new_mask, past_key_values=past_key_values)

    def get_token_id(self, hidden_state: torch.Tensor) -> torch.Tensor:
        last_hidden_state = hidden_state[:, -1]
        input_ids = torch.argmax(last_hidden_state, dim=-1).unsqueeze(1)
        return input_ids

    @torch.no_grad()
    def generate_step(self, model: Module, data_iter: Iterable) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run.
            data_iter (Iterable): Data iterator.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        output_sequence = []
        self.load_batch(data_iter)
        model.eval()

        # run by round
        for _ in range(self.round):
            state = Status.PREFILL
            while self.mb_manager.is_micro_batch_done() is False:
                pre_stage_out, model_inputs, hidden_states = self._prepare_stage_inputs()

                output_obj = model_forward(model, model_inputs, hidden_states)

                past_key_values = output_obj.get('past_key_values', None)
                state = self.mb_manager.step(model_inputs, pre_stage_out, past_key_values)
                if self.stage_manager.is_last_stage():
                    new_token = self.get_token_id(output_obj['hidden_states'])
                    self.mb_manager.add_new_tokens(new_token)
                    if state is not Status.DONE:
                        self.comm.send_forward({'new_token': new_token})
                    elif state is Status.DONE:
                        output_sequence.extend(self.mb_manager.export_new_tokens())
                else:
                    self.comm.send_forward({'hidden_states': output_obj['hidden_states']})
                self.mb_manager.next()
        return output_sequence
