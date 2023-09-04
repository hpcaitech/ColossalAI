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

    def _prepare_inputs_for_interval_stage(self):
        '''
        Prepare inputs for interval stage, for all the interval stage, the inputs is just the past_key_values

        Returns:
            dict: inputs for interval stage, `{'past_key_values': torch.Tensor}` or `None`
        '''
        model_inputs = {
            'past_key_values': self.mb_manager.cur_kv_cache
        } if self.mb_manager.cur_kv_cache is not None else None
        return model_inputs

    def _prepare_inputs_for_new_token(self, new_token: torch.Tensor):
        '''
        Prepare inputs for new token, the inputs is a dict with `input_ids`, `attention_mask` and `past_key_values`
        `input_ids` is the new token, `attention_mask` is the previous mask add `1` in the end,
        `past_key_values` is the past_key_values save in the micro batch manager

        Returns:
            dict: inputs for new token, `{'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'past_key_values': torch.Tensor}`
        '''
        new_mask = self.mb_manager.cur_descrption.attn_mask
        past_key_values = self.mb_manager.cur_descrption.kv_cache

        return dict(input_ids=new_token, attention_mask=new_mask, past_key_values=past_key_values)

    def _get_token_id(self, hidden_state: torch.Tensor) -> torch.Tensor:
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
            while self.mb_manager.is_micro_batch_done() is False:
                inputs_dict = None
                new_token = None
                output_dict = None

                # First stage and in PREFILL phase, just load the inputs
                if self.stage_manager.is_first_stage() and self.mb_manager.cur_state is Status.PREFILL:
                    inputs_dict = self.load_micro_batch()
                    output_dict = model_forward(model, inputs_dict, None)
                    self.mb_manager.step(inputs_dict, output_dict, None)
                # In GENERATE phase
                else:
                    # Get hidden_states from previous stage
                    hidden_states = self.comm.recv_forward()
                    if self.stage_manager.is_first_stage():
                        # First just generate a new token
                        assert hidden_states is not None, "When first stage in GENERATE phase, the hidden states should not be None"
                        logits = model_forward(model, None, hidden_states)
                        assert 'logits' in logits, f"When first stage in GENERATE phase, the ouput should have attribute `logits`, but has {output.keys()}"
                        new_token = self._get_token_id(logits['logits'])
                        self.mb_manager.step(None, None, new_token)
                        # If the current micro batch is not DONE, go through blocks
                        if self.mb_manager.cur_state is Status.GENERATE:
                            inputs_dict = self._prepare_inputs_for_new_token(new_token)
                            output_dict = model_forward(model, inputs_dict, None)
                            self.mb_manager.step(inputs_dict, output_dict, None)
                    else:
                        assert hidden_states is not None, "When not first stage, the hidden states should not be None"
                        inputs_dict = self._prepare_inputs_for_interval_stage()
                        output_dict = model_forward(model, inputs_dict, hidden_states)
                        self.mb_manager.step(inputs_dict, output_dict, None)

                # Current microbatch is not DONE, send hidden_state to next stage
                if not self.stage_manager.is_first_stage() or self.mb_manager.cur_state is Status.GENERATE:
                    self.comm.send_forward({'hidden_states': output_dict['hidden_states']})

                self.mb_manager.next()

            # All microbatch in current round is DONE
            if self.stage_manager.is_first_stage():
                output_sequence.extend(self.mb_manager.export_new_tokens())
            self.mb_manager.clear()
        return output_sequence
