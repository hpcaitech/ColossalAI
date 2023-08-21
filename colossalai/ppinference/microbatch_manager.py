from typing import Dict

import torch

from .inference_config import InferenceConfig

__all__ = 'MicroBatchManager'

PREFILL = 1
GENERATE = 2
DONE = 3


class MicroBatchDescription():

    def __init__(
        self,
        mb_inputs: torch.Tensor,
        inter_inputs,
        new_length: int,
    ) -> None:
        if mb_inputs is not None:
            assert mb_inputs.get('input_ids') is not None and mb_inputs.get('attention_mask') is not None
            self.mb_length = mb_inputs['input_ids'].shape[-1]
            self.attn_mask = mb_inputs['attention_mask']
            self.input_ids = mb_inputs['input_ids']

        elif inter_inputs is not None:
            assert inter_inputs.get('hidden_states') is not None
            # print(inter_inputs['hidden_states'].shape)
            self.mb_length = inter_inputs['hidden_states'].shape[-2]
        else:
            raise ValueError('mb_inputs and inter_inputs can not be None at the same time')

        self.target_length = self.mb_length + new_length
        self.kv_cache = ()

    def update(self, kv_cache):
        self.kv_cache = kv_cache

    @property
    def cur_length(self):
        """
        Return the current sequnence length of micro batch, when there is no kv_cache, the length is mb_length,
        otherwise the sequence length is `kv_cache[0][0].shape[-2]` plus 1

        """
        if len(self.kv_cache) == 0:
            return self.mb_length
        return self.kv_cache[0][0].shape[-2] + 1

    @property
    def state(self):
        """
        Return the state of current micro batch, when current length is equal to target length,
        the state is DONE, otherwise GENERATE

        """
        if self.cur_length == self.target_length:
            return DONE
        else:
            return GENERATE


class MicroBatchManager():

    def __init__(
        self,
        pp_inference_config: InferenceConfig,
    ):
        self.pp_inference_config = pp_inference_config
        self.mb_descrption_buffer = {}
        self.buffer_size = pp_inference_config.micro_batch_buffer_size
        self.idx = 0

    def _add_descrption(self, mb_inputs: Dict[str, torch.Tensor], inter_inputs: Dict[str, torch.Tensor]):
        self.mb_descrption_buffer[self.idx] = MicroBatchDescription(mb_inputs, inter_inputs,
                                                                    self.pp_inference_config.new_length)

    def _update_descrption(self, present_kv):
        self.mb_descrption_buffer[self.idx].update(present_kv)

    def _remove_descrption(self):
        self.mb_descrption_buffer.pop(self.idx)

    def step(self, mb_inputs=None, inter_inputs=None, present_kv=None):
        """
        Update the state if microbatch manager

        Args:
            mb_inputs (int, optional): The input of first stage when in prefill, should be a dict like {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor}.
            inter_inputs ([type], optional): The input of intermediate stage (the output of previous stage), should be a dict like {'hidden_state': torch.Tensor}.
            present_kv ([type], optional): The kvcache of current microbatch in current stage.
        """
        if self.mb_descrption_buffer.get(self.idx) is None:
            self._add_descrption(mb_inputs, inter_inputs)
        self._update_descrption(present_kv)
        state = self.cur_state
        if state == DONE:
            self._remove_descrption()
        return state

    def next(self):
        self.idx = (self.idx + 1) % self.buffer_size

    @property
    def cur_descrption(self) -> MicroBatchDescription:
        return self.mb_descrption_buffer.get(self.idx)

    @property
    def cur_kv_cache(self):
        return self.cur_descrption.kv_cache

    @property
    def cur_state(self):
        """
        Return the state of current micro batch, when current descrption is None, the state is PREFILL

        """
        if self.cur_descrption is None:
            return PREFILL
        return self.cur_descrption.state
