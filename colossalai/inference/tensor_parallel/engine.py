from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import BloomForCausalLM, LlamaForCausalLM
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy

from .batch_infer_state import BatchInferState
from .kvcache_manager import MemoryManager

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2

_supported_models = ['LlamaForCausalLM', 'LlamaModel', 'BloomForCausalLM']


class TPInferEngine:
    """Engine class for tensor parallel inference.

    Args:
        model (Module): original model, e.g. huggingface CausalLM
        shard_config (ShardConfig): The config for sharding original model
        max_batch_size (int): maximum batch size
        max_input_len (int): maximum input length of sequence
        max_output_len (int): maximum output length of output tokens
        dtype (torch.dtype): datatype used to init KV cache space
        device (str): device the KV cache of engine to be initialized on

    Examples:
        >>> # define model and shard config for your inference
        >>> model = ...
        >>> generate_kwargs = ...
        >>> shard_config = ShardConfig(enable_tensor_parallelism=True, inference_only=True)
        >>> infer_engine = TPInferEngine(model, shard_config, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN)
        >>> outputs = infer_engine.generate(input_ids, **generate_kwargs)
    """

    def __init__(self,
                 model: nn.Module,
                 shard_config: ShardConfig,
                 max_batch_size: int,
                 max_input_len: int,
                 max_output_len: int,
                 dtype: torch.dtype = torch.float16,
                 device: str = 'cuda') -> None:
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_total_token_num = self.max_batch_size * (self.max_input_len + self.max_output_len)

        # Constraints relatable with specs of devices and model
        # This may change into an optional arg in the future
        assert self.max_batch_size <= 64, "Max batch size exceeds the constraint"
        assert self.max_input_len + self.max_output_len <= 4096, "Max length exceeds the constraint"

        self.dtype = dtype

        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_num = model.config.num_attention_heads
        self.layer_num = model.config.num_hidden_layers

        self.tp_size = -1    # to be set with given shard config in self.prepare_shard_config
        self.cache_manager = None

        self.shard_config = shard_config
        self.model = None
        # optimize the original model by sharding with ShardFormer
        self._optimize_model(model=model.to(device))

    def _init_manager(self) -> None:
        assert self.tp_size >= 1, "TP size not initialized without providing a valid ShardConfig"
        assert self.head_num % self.tp_size == 0, f"Cannot shard {self.head_num} heads with tp size {self.tp_size}"
        self.head_num //= self.tp_size    # update sharded number of heads
        self.cache_manager = MemoryManager(self.max_total_token_num, self.dtype, self.head_num, self.head_dim,
                                           self.layer_num)

    def _optimize_model(self, model: nn.Module) -> None:
        """
        Optimize the original model by sharding with ShardFormer.
        In further generation, use the sharded model instead of original model.
        """
        # NOTE we will change to use an inference config later with additional attrs we want
        assert self.shard_config.inference_only is True
        shardformer = ShardFormer(shard_config=self.shard_config)
        self._prepare_with_shard_config(shard_config=self.shard_config)
        self._shard_model_by(shardformer, model)

    def _prepare_with_shard_config(self, shard_config: Optional[ShardConfig] = None) -> ShardConfig:
        """ Prepare the engine with a given ShardConfig.

        Args:
            shard_config (ShardConfig): shard config given to specify settings of the engine.
                If not provided, a default ShardConfig with tp size 1 will be created.
        """
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

    def _shard_model_by(self, shardformer: ShardFormer, model: nn.Module) -> None:
        """ Shard original model by the given ShardFormer and store the sharded model. """
        assert self.tp_size == shardformer.shard_config.tensor_parallel_size, \
            "Discrepancy between the tp size of TPInferEngine and the tp size of shard config"
        model_name = model.__class__.__name__
        assert model_name in self.supported_models, f"Unsupported model cls {model_name} for TP inference."
        policy = get_autopolicy(model, inference_only=True)
        self.model, _ = shardformer.optimize(model, policy)
        self.model = self.model.cuda()

    @property
    def supported_models(self) -> List[str]:
        return _supported_models

    def generate(self, input_tokens: Union[BatchEncoding, dict, list, torch.Tensor], **generate_kwargs) -> torch.Tensor:
        """Generate token sequence.

        Args:
            input_tokens: could be one of the following types
                1. BatchEncoding or dict (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
        Returns:
            torch.Tensor: The returned sequence is given inputs + generated_tokens.
        """
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = dict(input_ids=input_tokens, attention_mask=torch.ones_like(input_tokens, dtype=torch.bool))
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].cuda()
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs.update(max_new_tokens=self.max_output_len)

        return self._generate_by_set_infer_state(input_tokens, **generate_kwargs)

    def prepare_batch_state(self, inputs) -> BatchInferState:
        """
        Create and prepare BatchInferState used for inference during model forwrad,
        by processing each sequence of the given inputs.

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
                curr_seq_len = len(attn_mask)
                # if isinstance(attn_mask, torch.Tensor):
                #     curr_seq_len = int(torch.sum(attn_mask))
                # else:
                #     curr_seq_len = int(sum(attn_mask))
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        else:
            length = max(len(input_id) for input_id in input_ids_list)
            for i, input_ids in enumerate(input_ids_list):
                curr_seq_len = length
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch
        block_loc = torch.empty((batch_size, self.max_input_len + self.max_output_len), dtype=torch.long, device='cuda')
        batch_infer_state = BatchInferState(batch_size, max_len_in_batch)
        batch_infer_state.seq_len = seq_lengths.to('cuda')
        batch_infer_state.start_loc = seq_start_indexes.to('cuda')
        batch_infer_state.block_loc = block_loc
        batch_infer_state.decode_layer_id = 0
        batch_infer_state.past_key_values_len = 0
        batch_infer_state.is_context_stage = True
        batch_infer_state.set_cache_manager(self.cache_manager)
        return batch_infer_state

    @torch.no_grad()
    def _generate_by_set_infer_state(self, input_tokens, **generate_kwargs) -> torch.Tensor:
        """
        Generate output tokens by setting BatchInferState as an attribute to the model and calling model.generate

        Args:
            inputs: should be one of the following types
                1. BatchEncoding or dict (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
        """

        # for testing, always use sharded model
        assert self.model is not None, "sharded model does not exist"

        batch_infer_state = self.prepare_batch_state(input_tokens)
        assert batch_infer_state.max_len_in_batch <= self.max_input_len, "max length in batch exceeds limit"

        # set BatchInferState for the current batch as attr to model
        # NOTE this is not a preferable way to pass BatchInferState during inference
        #   we might want to rewrite generate function (e.g. _generate_by_pass_infer_state)
        #   and pass BatchInferState via model forward
        model = self.model
        if isinstance(model, LlamaForCausalLM):
            model = self.model.model
        elif isinstance(model, BloomForCausalLM):
            model = self.model.transformer
        setattr(model, 'infer_state', batch_infer_state)

        outputs = self.model.generate(**input_tokens, **generate_kwargs, early_stopping=False)

        # NOTE In future development, we're going to let the scheduler to handle the cache,
        #      instead of freeing space explicitly at the end of generation
        self.cache_manager.free_all()

        return outputs

    # TODO might want to implement the func that generates output tokens by passing BatchInferState
    #      as an arg into model.forward.
    #      It requires rewriting model generate and replacing model forward.
    @torch.no_grad()
    def _generate_by_pass_infer_state(self,
                                      input_tokens,
                                      max_out_length: int,
                                      generation_config: Optional[GenerationConfig] = None,
                                      stopping_criteria: Optional[StoppingCriteriaList] = None,
                                      prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
                                      **model_kwargs) -> torch.Tensor:

        raise NotImplementedError("generate by passing BatchInferState is not implemented.")

    # might want to use in rewritten generate method: use after model.forward
    # BatchInferState is created and kept during generation
    # after each iter of model forward, we should update BatchInferState
    def _update_batch_state(self, infer_state: Optional[BatchInferState]) -> None:
        batch_size = infer_state.batch_size
        device = infer_state.start_loc.device
        infer_state.start_loc = infer_state.start_loc + torch.arange(0, batch_size, dtype=torch.int32, device=device)
        infer_state.seq_len += 1

    # might want to create a sequence pool
    # add a single request/sequence/input text at a time and record its length
    # In other words, store the actual length of input tokens representing a single input text
    #   E.g. "Introduce landmarks in Beijing"
    #       => add request
    #       => record token length and other necessary information to be used
    #       => engine hold all these necessary information until `generate` (or other name) is called,
    #       => put information already recorded in batchinferstate and pass it to model forward
    #       => clear records in engine
    def add_request():
        raise NotImplementedError()
