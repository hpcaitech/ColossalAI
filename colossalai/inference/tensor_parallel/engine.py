from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import warnings
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BloomForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.tokenization_utils_base import BatchEncoding
try:
    from vllm import LLM
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    
    USE_CONTINOUS_BATCHING = True
    
except ImportError:
    warnings.warn("vllm is not installed, continuous batching will not be supported.")
    USE_CONTINOUS_BATCHING = False

from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy

from .batch_infer_state import BatchInferState
from .kvcache_manager import MemoryManager
from .utils import init_to_get_rotary, replace_page_attention

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2

_supported_models = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,
    "LlamaModel": LlamaForCausalLM,
    "BloomForCausalLM": BloomForCausalLM
}


class TPInferEngine:
    """Engine class for tensor parallel inference.

    Args:
        model (str, nn.Module): The name, path or instance of a HuggingFace Transformers model.
        shard_config (ShardConfig): The config for sharding original model
        max_batch_size (int): maximum batch size
        max_input_len (int): maximum input length of sequence
        max_output_len (int): maximum output length of output tokens
        trust_remote_code (bool): Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tokenizer (str, PreTrainedTokenizer, PreTrainedTokenizerFast): The name, path or instance of a HuggingFace Transformers tokenizer.
        use_continous_batching (bool): whether to use continous_batching
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
                 model: Union[str, nn.Module],
                 shard_config: ShardConfig = None,
                 max_batch_size: int = 8,
                 max_input_len: int = 16,
                 max_output_len: int = 8,
                 trust_remote_code: bool = False,
                 use_continous_batching: bool = False,
                 tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
                 dtype: torch.dtype = torch.float16,
                 device: str = 'cuda') -> None:

        if tokenizer is None:
            print("model: ", model)
            assert isinstance(model, str), \
                "when tokenizer is None, model must be string."
            tokenizer = model

        self.tp_size = 1
        self.use_continous_batching = use_continous_batching and USE_CONTINOUS_BATCHING

        if shard_config != None and shard_config.enable_tensor_parallelism:
            self.tp_size = shard_config.tensor_parallel_size

        if self.use_continous_batching:
            assert isinstance(model, str) and isinstance(tokenizer, str), \
                "when using continous_batching, model and tokenizer must be string."
            self.llm_engine = LLM(model=model,
                                  tokenizer=tokenizer,
                                  trust_remote_code=trust_remote_code,
                                  tensor_parallel_size=self.tp_size)
            #TODO We will replace multiple models' attention forward with shardformer in vllm to achieve multi-stream optimization later.
            # kv_cache_stream = torch.cuda.Stream()
            # self.model = replace_page_attention(self.llm_engine.llm_engine.workers[0].model, kv_cache_stream)
            self.model = self.llm_engine.llm_engine.workers[0].model
        else:
            self.model, self.tokenizer = self._get_model_and_tokenizer(model, tokenizer, trust_remote_code)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.half()
        self.model = self.model.to(device)
        self.shard_config = shard_config

        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_total_token_num = self.max_batch_size * (self.max_input_len + self.max_output_len)

        # Constraints relatable with specs of devices and model
        # This may change into an optional arg in the future
        assert self.max_batch_size <= 64, "Max batch size exceeds the constraint"
        assert self.max_input_len + self.max_output_len <= 4096, "Max length exceeds the constraint"

        self.dtype = dtype

        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.head_num = self.model.config.num_attention_heads
        self.layer_num = self.model.config.num_hidden_layers

        self.cache_manager = None
        
        self._optimize_model()

    def _get_model_and_tokenizer(self, model: str, tokenizer: str, trust_remote_code: bool) -> nn.Module:

        if isinstance(model, nn.Module) and isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            return model, tokenizer

        supported_model = model

        try:
            config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        except ValueError as e:
            if (not trust_remote_code and "requires you to execute the configuration file" in str(e)):
                err_msg = ("Failed to load the model config. If the model is a custom "
                           "model not yet available in the HuggingFace transformers "
                           "library, consider setting `trust_remote_code=True` in LLM "
                           "or using the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e
        architectures = getattr(config, "architectures", [])
        for arch in architectures:
            if arch in _supported_models:
                if isinstance(tokenizer, str):
                    if arch == "LlamaForCausalLM" or arch == "LLaMAForCausalLM" or arch == "LlamaModel":
                        tokenizer = LlamaTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
                if isinstance(model, str):
                    supported_model = _supported_models[arch].from_pretrained(model,
                                                                              pad_token_id=tokenizer.eos_token_id)

                return supported_model, tokenizer
        raise ValueError(f"Model architectures {architectures} are not supported for now. "
                         f"Supported architectures: {self.supported_models}")

    def _init_manager(self) -> None:
        assert self.tp_size >= 1, "TP size not initialized without providing a valid ShardConfig"
        assert self.head_num % self.tp_size == 0, f"Cannot shard {self.head_num} heads with tp size {self.tp_size}"
        self.head_num //= self.tp_size    # update sharded number of heads
        self.cache_manager = MemoryManager(self.max_total_token_num, self.dtype, self.head_num, self.head_dim,
                                           self.layer_num)

    def _optimize_model(self) -> None:
        """
        Optimize the original model by sharding with ShardFormer.
        In further generation, use the sharded model instead of original model.
        """
        # NOTE we will change to use an inference config later with additional attrs we want
        if not self.use_continous_batching:
            assert self.shard_config.inference_only is True
            shardformer = ShardFormer(shard_config=self.shard_config)
            self._prepare_with_shard_config(shard_config=self.shard_config)
            self._shard_model_by(shardformer)

    def _prepare_with_shard_config(self, shard_config: Optional[ShardConfig] = None) -> ShardConfig:
        """ Prepare the engine with a given ShardConfig.

        Args:
            shard_config (ShardConfig): shard config given to specify settings of the engine.
                If not provided, a default ShardConfig with tp size 1 will be created.
        """
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
        self._init_manager()

        return shard_config

    def _shard_model_by(self, shardformer: ShardFormer) -> None:
        """ Shard original model by the given ShardFormer and store the sharded model. """
        assert self.tp_size == shardformer.shard_config.tensor_parallel_size, \
            "Discrepancy between the tp size of TPInferEngine and the tp size of shard config"
        model_name = self.model.__class__.__name__
        assert model_name in self.supported_models, f"Unsupported model cls {model_name} for TP inference."
        if model_name == "LlamaForCausalLM":
            init_to_get_rotary(self.model.model, base=10000)
        policy = get_autopolicy(self.model, inference_only=True)
        self.model, _ = shardformer.optimize(self.model, policy)
        self.model = self.model.cuda()

    @property
    def supported_models(self) -> List[str]:
        return list(_supported_models.keys())

    def generate(self,
                 prompts: Optional[Union[str, List[str]]] = None,
                 prompt_token_ids: Optional[Union[BatchEncoding, dict, list, torch.Tensor]] = None,
                 **generate_kwargs) -> Union[List[RequestOutput], torch.Tensor]:
        """Generate token sequence.

        Args:
            prompts: A list of prompts to generate completions for.
            prompt_token_ids: could be one of the following types
                1. BatchEncoding or dict (e.g. tokenizer batch_encode)
                2. list of input token ids (e.g. appended result of tokenizer encode)
                3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
                If None, we use the tokenizer to convert the prompts to token IDs.
        Returns:
            Union[List[RequestOutput], torch.Tensor]: The returned sequence is given inputs + generated_tokens.
        """

        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if isinstance(prompt_token_ids, (BatchEncoding, dict)):
                prompt_token_len = len(prompt_token_ids['input_ids'])
            elif isinstance(prompt_token_ids, torch.Tensor):
                prompt_token_len = prompt_token_ids.shape[0]
            else:
                prompt_token_len = len(prompt_token_ids)
            if len(prompts) != prompt_token_len:
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

        if self.use_continous_batching:
            if not isinstance(prompt_token_ids, list):
                raise TypeError(f"prompt_token_ids type must be list, when using continous batching.")
            sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_output_len)
            return self.llm_engine.generate(prompts=prompts,
                                            prompt_token_ids=prompt_token_ids,
                                            sampling_params=sampling_params)

        if prompt_token_ids is None:
            input_tokens = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
        else:
            input_tokens = prompt_token_ids
            
        if isinstance(input_tokens, list):
            input_tokens = torch.Tensor(input_tokens).cuda()

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
