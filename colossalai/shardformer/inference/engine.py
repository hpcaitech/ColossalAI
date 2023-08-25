from functools import partial
from types import MethodType
from typing import Any, Callable, List, Optional, Set, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.generation import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.tokenization_utils_base import BatchEncoding

from colossalai.cluster import ProcessGroupMesh
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.inference import BatchInferState, MemoryManager
# from colossalai.shardformer.policies.bloom import BloomModelInferPolicy
from colossalai.shardformer.policies.auto_policy import get_autopolicy

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


class InferenceEngine:

    def __init__(self, model: nn.Module, max_batch_size, max_input_len, max_output_len, tp_size=1) -> None:
        self.model = model
        self.sharded_model = None

        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_total_token_num = self.max_batch_size * (self.max_input_len + self.max_output_len)
        assert self.max_batch_size <= 64
        assert self.max_input_len + self.max_output_len <= 2048

        self.tp_size = tp_size
        self.pp_size = 1    # only consider tp for now
        self.dp_size = 1    # only consider tp for now

        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.head_num = self.model.config.num_attention_heads // self.tp_size
        self.layer_num = self.model.config.num_hidden_layers
        self.cache_manager = MemoryManager(self.max_total_token_num, torch.float16, self.head_num, self.head_dim,
                                           self.layer_num)

        # self.pg_mesh = ProcessGroupMesh(self.dp_size, self.pp_size, self.tp_size)
        # self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS)

    def shard_model_by(self, shardformer: ShardFormer) -> None:
        # TODO Might want to use infer policy only when bs >= 4
        assert self.tp_size == shardformer.shard_config.tensor_parallel_size, "Engine tp size != shardformer tp size"
        # shardformer.shard_config.tensor_parallel_process_group = self.tp_group
        model_name = self.model.__class__.__name__
        policy = get_autopolicy(self.model, inference_only=True)
        if model_name == 'LlamaForCausalLM':
            self.sharded_model, _ = shardformer.optimize(self.model, policy)
        elif model_name == 'BloomForCausalLM':
            self.sharded_model, _ = shardformer.optimize(self.model, policy)
        else:
            raise ValueError(f'Unsupported model "{model_name}" for inference')
        self.sharded_model = self.sharded_model.cuda()

    # NOTE input_tokens is expected to be BatchEncoding,
    # instead of only input token ids
    @torch.no_grad()
    def generate_by_pass_infer_state(self,
                                     input_tokens,
                                     max_out_length: int,
                                     generation_config: Optional[GenerationConfig] = None,
                                     stopping_criteria: Optional[StoppingCriteriaList] = None,
                                     prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
                                     **model_kwargs) -> torch.Tensor:

        input_ids = input_tokens['input_ids']
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        if batch_size >= 4:
            assert self.sharded_model is not None, "sharded model does not exist"

            batch_infer_state = self.prepare_batch_state(input_tokens)
            batch_size = batch_infer_state.batch_size
            assert batch_infer_state.max_len_in_batch <= self.max_input_len

            # record sequences finish status, add early stopping, etc,

            for _ in range(min(max_out_length, self.max_output_len)):
                # ...
                self.sharded_model.forward(..., **model_kwargs)
        else:
            # Use original model
            orig_model = self.model

            for _ in range(min(max_out_length, self.max_output_len)):

                if prepare_inputs_fn is None and hasattr(orig_model, 'prepare_inputs_for_generation'):
                    prepare_inputs_fn = orig_model.prepare_inputs_for_generation

                model_inputs = prepare_inputs_fn(input_ids, **
                                                 model_kwargs) if prepare_inputs_fn is not None else input_tokens
                outputs = orig_model(**model_inputs)

                # next_token_logits = outputs['logits'][:, -1, :]
                next_token_logits = outputs.logits[:, -1, :]
                # pre-process distribution
                # next_token_logits = logits_processor(input_ids, next_token_logits)

                # sample
                # probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
                # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # consider greedy only for now
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                # finished sentences should have their next token be a padding token

                # if eos_token_id is not None:
                #     if pad_token_id is None:
                #         raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                #     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # # update generated ids, model inputs for next step
                # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                # if update_model_kwargs_fn is not None:
                #     model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)

                # # if eos_token was found in one sentence, set sentence to finished
                # if eos_token_id is not None:
                #     unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

                # # stop when each sentence is finished if early_stopping=True
                # if early_stopping and _is_sequence_finished(unfinished_sequences):
                #     break

            return input_ids

    @torch.no_grad()
    def generate_by_set_infer_state(self, input_tokens, generate_kwargs, early_stopping=False):

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

        # add logging
        generate_kwargs.update(max_new_tokens=self.max_output_len)

        # convert to dict
        if isinstance(input_tokens, torch.Tensor):
            input_tokens = dict(input_ids=input_tokens)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
                print(f" input_tokens[{t}].shape: {input_tokens[t].shape}")

        outputs = self.sharded_model.generate(**input_tokens, **generate_kwargs, early_stopping=early_stopping)

        print(f"outputs.shape {outputs.shape}")
        return outputs

    #  inputs should be one of the following types
    #   1. BatchEncoding (e.g. tokenizer batch_encode)
    #   2. list of input token ids (e.g. appended result of tokenizer encode)
    #   3. torch.Tensor (e.g. tokenizer encode with return_tensors='pt')
    #   NOTE For torch.Tensor inputs representing a batch of inputs, we are unable to retrieve
    #       the actual length (e.g. number of tokens) of each input without attention mask
    #       Hence, for torch.Tensor with shape [bs, l] where bs > 1, we will assume
    #       all the inputs in the batch has the maximum length l
    def prepare_batch_state(self, inputs: [BatchEncoding, torch.Tensor]) -> BatchInferState:
        # records length based on attention mask
        # Any better method?
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

        block_loc = torch.empty(batch_size, self.max_input_len + self.max_output_len, dtype=torch.long, device="cuda")
        seq_start_indexes = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        start_index = 0
        if isinstance(inputs, BatchEncoding):
            for i, attn_mask in enumerate(attn_masks):
                curr_seq_len = torch.sum(attn_mask)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
        else:
            max_len_in_batch = -1
            for i, input_ids in enumerate(inputs):
                curr_seq_len = len(input_ids)
                seq_lengths[i] = curr_seq_len
                seq_start_indexes[i] = start_index
                start_index += curr_seq_len
                max_len_in_batch = curr_seq_len if curr_seq_len > max_len_in_batch else max_len_in_batch

        batch_infer_state = BatchInferState(batch_size, max_len_in_batch)
        batch_infer_state.seq_len = seq_lengths.to('cuda')    # might want to assign specific device
        batch_infer_state.start_loc = seq_start_indexes.to('cuda')
        batch_infer_state.block_loc = block_loc
        # NOTE BatchInferState.total_token_num revised (not pushed yet)
        #       Now we want actual total token num based on seq_len, instead of dummy ones in test
        #       (Could still use the dummy one for testing usage)
        batch_infer_state.set_cache_manager(self.cache_manager)
        batch_infer_state.decode_layer_id = 0
        batch_infer_state.past_key_values_len = 0
        batch_infer_state.is_context_stage = True
        return batch_infer_state

    # BatchInferState is created and kept during generation
    # after each iter of model forward, we should update BatchInferState
    # NOTE use in rewritten generate method: use after model.forward
    def update_batch_state(self, infer_state: Optional[BatchInferState]) -> None:
        # self.b_start_loc = self.b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        # self.b_seq_len += 1
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
        pass
