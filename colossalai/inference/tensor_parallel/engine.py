from typing import Any, Callable, List, Optional, Set, Union

import torch
import torch.nn as nn
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy

from .batch_infer_state import BatchInferState
from .kvcache_manager import MemoryManager

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


class TPInferEngine:

    def __init__(self,
                 model: nn.Module,
                 max_batch_size,
                 max_input_len,
                 max_output_len,
                 dtype=torch.float16,
                 tp_size=1) -> None:
        self.model = model
        self.sharded_model = None

        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_total_token_num = self.max_batch_size * (self.max_input_len + self.max_output_len)

        # Constraints relatable with specs of devices
        assert self.max_batch_size <= 64
        assert self.max_input_len + self.max_output_len <= 2048

        # NOTE For now, we focus on tensor parallel.
        # We might want to merge pp and dp inference in future.
        self.tp_size = tp_size
        self.pp_size = 1
        self.dp_size = 1
        self.dtype = dtype

        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.head_num = self.model.config.num_attention_heads // self.tp_size
        self.layer_num = self.model.config.num_hidden_layers
        self.cache_manager = MemoryManager(self.max_total_token_num, self.dtype, self.head_num, self.head_dim,
                                           self.layer_num)
        self.pg_mesh = ProcessGroupMesh(self.dp_size, self.pp_size, self.tp_size)

    def create_shard_config(self) -> ShardConfig:
        """ create a ShardConfig consistent with configs and attributes of the engine """
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
        if self.tp_size > 1:
            shard_config.enable_tensor_parallelism = True
            tp_process_group = self.pg_mesh.get_group_along_axis(TP_AXIS)
            shard_config.tensor_parallel_process_group = tp_process_group
        return shard_config

    def shard_model_by(self, shardformer: ShardFormer) -> None:
        """ Shard the model and store the sharded model by given ShardFormer """
        assert self.tp_size == shardformer.shard_config.tensor_parallel_size, \
            "Discrepancy between the tp size of TPInferEngine and the tp size of shard config"
        model_name = self.model.__class__.__name__
        assert model_name in self._supported_model(), f"Unsupported model cls {model_name} for TP inference."
        policy = get_autopolicy(self.model, inference_only=True)
        self.sharded_model, _ = shardformer.optimize(self.model, policy)
        self.sharded_model = self.sharded_model.cuda()

    def _supported_model(self) -> List[str]:
        supported_models = ['LlamaForCausalLM', 'LlamaModel', 'BloomForCausalLM']
        return supported_models

    @torch.no_grad()
    def generate_by_set_infer_state(self, input_tokens, generate_kwargs, early_stopping=False) -> torch.Tensor:
        """
        Generate output tokens by setting BatchInferState as an attribute to the model and calling model.generate

        Args:
            inputs: should be one of the following types
                1. BatchEncoding (e.g. tokenizer batch_encode)
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
        if hasattr(self.sharded_model, 'model'):
            model = self.sharded_model.model
        elif hasattr(self.sharded_model, 'transformer'):
            model = self.sharded_model.transformer
        setattr(model, 'infer_state', batch_infer_state)

        generate_kwargs.update(max_new_tokens=self.max_output_len)

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = dict(input_ids=input_tokens)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

        outputs = self.sharded_model.generate(**input_tokens, **generate_kwargs, early_stopping=early_stopping)

        print(f"outputs.shape {outputs.shape}")
        return outputs

    def prepare_batch_state(self, inputs: [BatchEncoding, torch.Tensor]) -> BatchInferState:
        """
        Create and prepare BatchInferState used for inference during model forwrad,
        by processing each sequence of the given inputs

        Args:
            inputs: should be one of the following types
                1. BatchEncoding (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
                NOTE For torch.Tensor inputs representing a batch of inputs, we are unable to retrieve
                    the actual length (e.g. number of tokens) of each input without attention mask
                    Hence, for torch.Tensor with shape [bs, l] where bs > 1, we will assume
                    all the inputs in the batch has the maximum length l
        Returns:
            BatchInferState: the states for the current batch during inference
        """
        if not isinstance(inputs, (BatchEncoding, list, torch.Tensor)):
            raise TypeError(f"inputs type {type(inputs)} is not supported in prepare_batch_state")

        if isinstance(inputs, BatchEncoding):
            attn_masks = inputs['attention_mask']
            batch_size = attn_masks.shape[0]
            max_len_in_batch = attn_masks.shape[1]
        elif isinstance(inputs, list):
            batch_size = len(inputs)
        else:
            batch_size = inputs.shape[0]

        seq_start_indexes = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        start_index = 0

        max_len_in_batch = -1
        if isinstance(inputs, BatchEncoding):
            for i, attn_mask in enumerate(attn_masks):
                curr_seq_len = torch.sum(attn_mask)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        else:
            for i, input_ids in enumerate(inputs):
                curr_seq_len = len(input_ids)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch

        block_loc = torch.empty((batch_size, self.max_input_len + self.max_output_len), dtype=torch.long, device="cuda")
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

        # input_ids = input_tokens['input_ids']

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
        #     # Use original model
        #     orig_model = self.model
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
