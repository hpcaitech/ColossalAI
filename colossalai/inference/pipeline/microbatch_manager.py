from enum import Enum
from typing import Dict

import torch

__all__ = 'MicroBatchManager'


class Status(Enum):
    PREFILL = 1
    GENERATE = 2
    DONE = 3


class MicroBatchDescription():

    def __init__(
        self,
        mb_inputs: Dict[str, torch.Tensor],
        interval_inputs: Dict[str, torch.Tensor],
        new_length: int,
    ) -> None:
        if mb_inputs is not None:
            assert mb_inputs.get('input_ids') is not None and mb_inputs.get('attention_mask') is not None
            self.mb_length = mb_inputs['input_ids'].shape[-1]
            self.attn_mask = mb_inputs['attention_mask']
            self.input_ids = mb_inputs['input_ids']

        elif interval_inputs is not None:
            assert interval_inputs.get('hidden_states') is not None
            self.mb_length = interval_inputs['hidden_states'].shape[-2]
        else:
            raise ValueError('mb_inputs and interval_inputs can not be None at the same time')

        self.target_length = self.mb_length + new_length
        self.kv_cache = ()
        self.new_tokens = None

    def update_kvcache(self, kv_cache):
        self.kv_cache = kv_cache

    def update_newtokens(self, new_token: torch.Tensor):
        if self.new_tokens is None:
            self.new_tokens = new_token
        else:
            self.new_tokens = torch.cat([self.new_tokens, new_token], dim=-1)

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
            return Status.DONE
        else:
            return Status.GENERATE


class MicroBatchManager():
    '''
    MicroBatchManager is a class that manages the micro batch.

    Args:
        new_length (int): the new length of the input sequence.
        micro_batch_size (int): the micro batch size.
        micro_batch_buffer_size (int): the buffer size for micro batch. Normally, it should be the same as the number of pipeline stages.
    '''

    def __init__(self, new_length: int, micro_batch_size: int, micro_batch_buffer_size: int):
        self.new_length = new_length
        self.micro_batch_size = micro_batch_size
        self.buffer_size = micro_batch_buffer_size
        self.mb_descrption_buffer = {}
        self.new_tokens_buffer = {}
        self.idx = 0

    def _add_descrption(self, mb_inputs: Dict[str, torch.Tensor], inter_inputs: Dict[str, torch.Tensor]):
        self.mb_descrption_buffer[self.idx] = MicroBatchDescription(mb_inputs, inter_inputs, self.new_length)

    def _update_descrption(self, present_kv):
        self.mb_descrption_buffer[self.idx].update_kvcache(present_kv)

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
        # self.next()
        return state

    def next(self):
        self.idx = (self.idx + 1) % self.buffer_size

    def is_micro_batch_done(self):
        if len(self.mb_descrption_buffer) == 0:
            return False
        for mb in self.mb_descrption_buffer.values():
            if mb.state != Status.DONE:
                return False
        self.mb_descrption_buffer.clear()
        return True

    def add_new_tokens(self, new_token):
        self.cur_descrption.update_newtokens(new_token)

    def export_new_tokens(self):
        list = self.cur_descrption.new_tokens.tolist()
        return list

    @property
    def cur_descrption(self) -> MicroBatchDescription:
        return self.mb_descrption_buffer.get(self.idx)

    @property
    def cur_kv_cache(self):
        if self.cur_descrption is None:
            return None
        return self.cur_descrption.kv_cache

    @property
    def cur_state(self):
        """
        Return the state of current micro batch, when current descrption is None, the state is PREFILL

        """
        if self.cur_descrption is None:
            return Status.PREFILL
        return self.cur_descrption.state
