from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from transformers import BloomForCausalLM, LlamaForCausalLM
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy

from .batch_infer_state import BatchInferState
from .kvcache_manager import MemoryManager

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2

_supported_models = ['LlamaForCausalLM', 'LlamaModel', 'BloomForCausalLM']


class TPInferEngine:

    def __init__(self,
                 model: nn.Module,
                 max_batch_size: int,
                 max_input_len: int,
                 max_output_len: int,
                 dtype: torch.dtype = torch.float16,
                 device: str = 'cuda') -> None:
        self.model = model
        self.sharded_model = None

        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_total_token_num = self.max_batch_size * (self.max_input_len + self.max_output_len)

        # Constraints relatable with specs of devices
        assert self.max_batch_size <= 64, "Max batch size exceeds the constraint"
        assert self.max_input_len + self.max_output_len <= 2048, "Max length exceeds the constraint"

        torch.device(device=device)
        self.dtype = dtype

        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.head_num = self.model.config.num_attention_heads
        self.layer_num = self.model.config.num_hidden_layers

        self.tp_size = -1    # to be set with given shard config in self.prepare_shard_config
        self.cache_manager = None

    def _init_manager(self) -> None:
        assert self.tp_size >= 1, "TP size not initialized without providing a valid ShardConfig"
        assert self.head_num % self.tp_size == 0, f"Cannot shard {self.head_num} heads with tp size {self.tp_size}"
        self.head_num //= self.tp_size    # update sharded number of heads
        self.cache_manager = MemoryManager(self.max_total_token_num, self.dtype, self.head_num, self.head_dim,
                                           self.layer_num)

    def prepare_with_shard_config(self, shard_config: Optional[ShardConfig] = None) -> ShardConfig:
        """ Prepare the engine with a given ShardConfig, or create a default one with tp size 1 """
        self.tp_size = 1
        if shard_config is None:
            shard_config = ShardConfig(
                tensor_parallel_process_group=None,
                pipeline_stage_manager=None,
                enable_tensor_parallelism=False,
                enable_fused_normalization=False,
                enable_all_optimization=False,
                enable_flash_attention=False,
                enable_jit_fused=False,
                inference_only=True,
            )
        else:
            shard_config.inference_only = True
            shard_config.pipeline_stage_manager = None
            if shard_config.enable_tensor_parallelism:
                self.tp_size = shard_config.tensor_parallel_size
        self._init_manager()

        return shard_config

    def shard_model_by(self, shardformer: ShardFormer) -> None:
        """ Shard the model and store the sharded model by given ShardFormer """
        assert self.tp_size == shardformer.shard_config.tensor_parallel_size, \
            "Discrepancy between the tp size of TPInferEngine and the tp size of shard config"
        model_name = self.model.__class__.__name__
        assert model_name in self._supported_models(), f"Unsupported model cls {model_name} for TP inference."
        policy = get_autopolicy(self.model, inference_only=True)
        self.sharded_model, _ = shardformer.optimize(self.model, policy)
        self.sharded_model = self.sharded_model.cuda()

    @staticmethod
    def _supported_models() -> List[str]:
        return _supported_models

    def generate(self, input_tokens, generate_kwargs) -> torch.Tensor:
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = dict(input_ids=input_tokens, attention_mask=torch.ones_like(input_tokens, dtype=torch.bool))
        if self.sharded_model is not None:
            return self.generate_by_set_infer_state(input_tokens, generate_kwargs)

        return self.model.generate(**input_tokens, **generate_kwargs)

    @torch.no_grad()
    def generate_by_set_infer_state(self, input_tokens, generate_kwargs) -> torch.Tensor:
        """
        Generate output tokens by setting BatchInferState as an attribute to the model and calling model.generate

        Args:
            inputs: should be one of the following types
                1. BatchEncoding or dict (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
        """

        # for testing, always use sharded model
        assert self.sharded_model is not None, "sharded model does not exist"

        batch_infer_state = self.prepare_batch_state(input_tokens)
        assert batch_infer_state.max_len_in_batch <= self.max_input_len, "max length in batch exceeds limit"

        # set BatchInferState for the current batch as attr to model
        # NOTE this is not an expectable way to pass BatchInferState during inference
        #   we might want to rewrite generate function (e.g. generate_by_pass_infer_state)
        #   and pass BatchInferState via model forward
        model = self.sharded_model
        if isinstance(model, LlamaForCausalLM):
            model = self.sharded_model.model
        elif isinstance(model, BloomForCausalLM):
            model = self.sharded_model.transformer
        setattr(model, 'infer_state', batch_infer_state)

        generate_kwargs.update(max_new_tokens=self.max_output_len)

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = dict(input_ids=input_tokens)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].cuda()

        outputs = self.sharded_model.generate(**input_tokens, **generate_kwargs, early_stopping=False)

        return outputs

    def prepare_batch_state(self, inputs) -> BatchInferState:
        """
        Create and prepare BatchInferState used for inference during model forwrad,
        by processing each sequence of the given inputs

        Args:
            inputs: should be one of the following types
                1. BatchEncoding or dict (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
                NOTE For torch.Tensor inputs representing a batch of inputs, we are unable to retrieve
                    the actual length (e.g. number of tokens) of each input without attention mask
                    Hence, for torch.Tensor with shape [bs, l] where bs > 1, we will assume
                    all the inputs in the batch has the maximum length l
        Returns:
            BatchInferState: the states for the current batch during inference
        """
        if not isinstance(inputs, (BatchEncoding, dict, list, torch.Tensor)):
            raise TypeError(f"inputs type {type(inputs)} is not supported in prepare_batch_state")

        input_ids_list = None
        attention_mask = None

        if isinstance(inputs, (BatchEncoding, dict)):
            input_ids_list = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        else:
            input_ids_list = inputs
        if isinstance(input_ids_list[0], int):    # for a single input
            input_ids_list = [input_ids_list]
            attention_mask = [attention_mask] if attention_mask is not None else attention_mask

        batch_size = len(input_ids_list)

        seq_start_indexes = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
        start_index = 0

        max_len_in_batch = -1
        if isinstance(inputs, (BatchEncoding, dict)):
            for i, attn_mask in enumerate(attention_mask):
                if isinstance(attn_mask, torch.Tensor):
                    curr_seq_len = int(torch.sum(attn_mask))
                else:
                    curr_seq_len = int(sum(attn_mask))
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        else:
            for i, input_ids in enumerate(input_ids_list):
                curr_seq_len = len(input_ids)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch

        block_loc = torch.empty((batch_size, self.max_input_len + self.max_output_len), dtype=torch.long, device='cuda')
        batch_infer_state = BatchInferState(batch_size, max_len_in_batch)
        batch_infer_state.seq_len = seq_lengths.to('cuda')    # might want to assign specific device
        batch_infer_state.start_loc = seq_start_indexes.to('cuda')
        batch_infer_state.block_loc = block_loc
        batch_infer_state.decode_layer_id = 0
        batch_infer_state.past_key_values_len = 0
        batch_infer_state.is_context_stage = True
        batch_infer_state.set_cache_manager(self.cache_manager)
        return batch_infer_state

    # TODO might want to implement the func that generates output tokens by passing BatchInferState
    #      as an arg into model.forward
    #      requires rewriting model generate and replacing model forward
    @torch.no_grad()
    def generate_by_pass_infer_state(self,
                                     input_tokens,
                                     max_out_length: int,
                                     generation_config: Optional[GenerationConfig] = None,
                                     stopping_criteria: Optional[StoppingCriteriaList] = None,
                                     prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
                                     **model_kwargs) -> torch.Tensor:
        # if batch_size >= 4:
        #     assert self.sharded_model is not None, "sharded model does not exist"
        #     batch_infer_state = self.prepare_batch_state(input_tokens)
        #     batch_size = batch_infer_state.batch_size
        #     assert batch_infer_state.max_len_in_batch <= self.max_input_len
        #     # record sequences finish status, add early stopping, etc,
        #     for _ in range(min(max_out_length, self.max_output_len)):
        #         # ...
        #         self.sharded_model.forward(..., **model_kwargs)
        # else:
        #     Use original model to generate
        raise NotImplementedError("generate by passing BatchInferState is not implemented.")

    # NOTE might want to use in rewritten generate method: use after model.forward
    # BatchInferState is created and kept during generation
    # after each iter of model forward, we should update BatchInferState
    def update_batch_state(self, infer_state: Optional[BatchInferState]) -> None:
        batch_size = infer_state.batch_size
        device = infer_state.start_loc.device
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device=device)
        infer_state.seq_len += 1

    # TODO might want to create a sequence pool
    #   add a single request/sequence/input text at a time and record its length
    #   In other words, store the actual length of input tokens representing a single input text
    #   E.g. "Introduce landmarks in Beijing"
    #       => add request
    #       => record token length and other necessary information to be used
    #       => engine hold all these necessary information until `generate` (or other name) is called,
    #       => put information already recorded in batchinferstate and pass it to model forward
    #       => clear records in engine
    def add_request():
        raise NotImplementedError()
