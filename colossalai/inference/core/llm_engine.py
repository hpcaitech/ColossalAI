import time
from itertools import count
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from colossalai.accelerator import get_accelerator
from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.batch_bucket import BatchBucket
from colossalai.inference.config import InferenceConfig, InputMetaData, ModelShardInferenceConfig
from colossalai.inference.graph_runner import CUDAGraphRunner
from colossalai.inference.modeling.policy import model_policy_map
from colossalai.inference.sampler import search_tokens
from colossalai.inference.spec import Drafter, GlideInput
from colossalai.inference.struct import Sequence
from colossalai.inference.utils import get_model_size, has_index_file
from colossalai.interface import ModelWrapper
from colossalai.lazy import LazyInitContext
from colossalai.logging import get_dist_logger
from colossalai.shardformer.policies.base_policy import Policy

from .base_engine import BaseEngine
from .request_handler import RequestHandler

PP_AXIS, TP_AXIS = 0, 1

_supported_models = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "BaichuanForCausalLM": AutoModelForCausalLM,
}

_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class LLMEngine(BaseEngine):
    """
    InferenceEngine which manages the inference process..

    Args:
        model_or_path (nn.Module or str): Path or nn.Module of this model.
        tokenizer Optional[(Union[PreTrainedTokenizer, PreTrainedTokenizerFast])]: Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        inference_config: InferenceConfig = None,
        verbose: bool = False,
        model_policy: Union[Policy, type[Policy]] = None,
    ) -> None:
        self.inference_config = inference_config
        self.dtype = inference_config.dtype
        self.high_precision = inference_config.high_precision

        self.verbose = verbose
        self.logger = get_dist_logger(__name__)
        self.model_shard_infer_config = inference_config.to_model_shard_inference_config()

        self.init_model(model_or_path, model_policy, self.model_shard_infer_config)

        self.generation_config = inference_config.to_generation_config(self.model_config)
        self.generation_config_dict = self.generation_config.to_dict()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.request_handler = RequestHandler(self.inference_config, self.model_config)
        self.k_cache, self.v_cache = self.request_handler.get_kvcache()
        # DISCUSS maybe move this into batch info?

        self.counter = count()

        self.use_cuda_graph = self.inference_config.use_cuda_graph
        if self.use_cuda_graph:
            self.graph_runners: Dict[int, CUDAGraphRunner] = {}
            self.graph_memory_pool = None  # Set during graph capture.
            if verbose:
                self.logger.info("Colossal AI CUDA Graph Capture on")

            self.capture_model(self.k_cache, self.v_cache)

        # Model and relatable attrs of speculative decoding will be set by `enable_spec_dec`
        self.use_spec_dec = self.inference_config.use_spec_dec

        self.drafter_model = None
        self.drafter = None
        self.use_glide = False
        self.n_spec_tokens = self.inference_config.max_n_spec_tokens

        self._verify_args()

    def init_model(
        self,
        model_or_path: Union[nn.Module, str],
        model_policy: Union[Policy, Type[Policy]] = None,
        model_shard_infer_config: ModelShardInferenceConfig = None,
    ):
        """
        Shard model or/and Load weight

        Args:
            model_or_path Union[nn.Module, str]: path to the checkpoint or model of transformer format.
            model_policy (Policy): the policy to replace the model.
            model_inference_config: the configuration for modeling initialization when inference.
            model_shard_infer_config (ModelShardInferenceConfig): the configuration for init of module when inference.
        """
        pretrained_path = None
        if isinstance(model_or_path, str):
            import colossalai.interface.pretrained as pretrained_utils

            try:
                hf_config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=True, torch_dtype=self.dtype)
                arch = getattr(hf_config, "architectures")[0]
                if arch in _supported_models.keys():
                    if arch == "BaichuanForCausalLM":
                        self.logger.warning(
                            "Attention ! We use lazy init by default, which could be faster for model loading. For baichuan model, the output maybe have a slight difference with transformers"
                        )
                    ctx = LazyInitContext(default_device="cuda")
                    with ctx:
                        model = _supported_models[arch].from_pretrained(
                            model_or_path, trust_remote_code=True, torch_dtype=self.dtype
                        )
                    pretrained_path = pretrained_utils.get_pretrained_path(model)
                else:
                    # TODO(char-1ee): if the model not supported, use transformers APIs to load and generate
                    raise ValueError(f"Model {arch} is not supported.")

            except Exception as e:
                self.logger.error(
                    f"An exception occurred during loading model: {e}, model should be loaded by transformers\n"
                )
        else:
            model = model_or_path

        self.model_config = model.config

        torch.cuda.empty_cache()
        init_gpu_memory = torch.cuda.mem_get_info()[0]

        self.device = get_accelerator().get_current_device()
        if self.verbose:
            self.logger.info(f"the device is {self.device}")

        model = model.to(self.dtype).eval()

        if self.verbose:
            self.logger.info(
                f"Before the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(model)} GB, model's device is: {model.device}"
            )

        if model_policy is None:
            prefix = "nopadding" if not self.inference_config.pad_input else "padding"
            model_policy_key = f"{prefix}_{getattr(self.model_config, 'model_type', None)}"
            model_policy = model_policy_map.get(model_policy_key)

        if not isinstance(model_policy, Policy):
            try:
                model_policy = model_policy()
            except Exception as e:
                raise ValueError(f"Unable to instantiate model policy: {e}")

        assert isinstance(model_policy, Policy), f"Invalid type of model policy: {type(model_policy)}"
        pg_mesh = ProcessGroupMesh(self.inference_config.pp_size, self.inference_config.tp_size)
        tp_group = pg_mesh.get_group_along_axis(TP_AXIS)

        self.model = self._shardformer(
            model,
            model_policy,
            model_shard_infer_config,
            None,
            tp_group=tp_group,
        )

        self.model = ModelWrapper(model).to(self.device)

        if self.verbose:
            self.logger.info(
                f"After the shard, Rank: [{dist.get_rank()}], model size: {get_model_size(self.model)} GB, model's device is: {model.device}"
            )

        if pretrained_path:
            from colossalai.inference.core.plugin import InferCheckpoint_io

            cpt_io = InferCheckpoint_io()
            if_has_index_file, model_index_file = has_index_file(pretrained_path)
            assert if_has_index_file, "the model path is invalid"
            cpt_io.load_model(self.model, model_index_file)

        free_gpu_memory, _ = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        if self.verbose:
            self.logger.info(
                f"Rank [{dist.get_rank()}], Model Weight Max Occupy {peak_memory / (1024 ** 3)} GB, Model size: {get_model_size(self.model)} GB"
            )

    @torch.inference_mode()
    def capture_model(self, k_cache: List[torch.Tensor], v_cache: List[torch.Tensor]):
        assert self.use_cuda_graph, "please turn on the cuda graph"

        if self.verbose:
            self.logger.info("Colossal AI CUDA Graph Capture begin")

        t_capture_begin = time.perf_counter()

        block_size = self.inference_config.block_size
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        max_context_len_to_capture = self.inference_config.max_context_len_to_capture
        max_num_blocks = (max_context_len_to_capture + block_size - 1) // block_size
        input_tokens_ids = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        # self.graph_block_tables = np.zeros((max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)
        self.graph_block_tables = np.full((max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), -1, dtype=np.int32)
        self.graph_block_tables[:, 0] = np.arange(max_num_blocks, max_num_blocks + max(_BATCH_SIZES_TO_CAPTURE))
        self.graph_block_tables[0, :] = np.arange(
            0, max_num_blocks
        )  # NOTE this is a hack to insure cuda grpah could capture the fixed cuda kernel grid in flash decoding, to make the first seqlen as the max_seq_len
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()
        output_tensor = torch.zeros(
            (max_batch_size, self.model_config.num_attention_heads * head_dim), dtype=self.dtype, device=self.device
        )
        fd_inter_tensor = self.request_handler.running_bb.fd_inter_tensor

        max_num_seqs = self.inference_config.max_batch_size
        batch_size_capture_list = [bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= max_num_seqs]
        sequence_lengths = torch.ones(max_batch_size, dtype=torch.int).cuda()
        # NOTE this is a hack to insure cuda grpah could capture the fixed cuda kernel grid in flash decoding, to make the first seqlen as the max_seq_len
        sequence_lengths[0] = torch.tensor(
            self.inference_config.max_context_len_to_capture - 1, dtype=torch.int32
        ).cuda()

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            if self.verbose:
                self.logger.info(f"batch size {batch_size} graph capturing")

            input_meta_data = InputMetaData(
                block_tables=block_tables[:batch_size],
                sequence_lengths=sequence_lengths[:batch_size],
                fd_inter_tensor=fd_inter_tensor,
                batch_size=batch_size,
                is_prompts=False,
                use_cuda_graph=True,
                high_precision=False,
                kv_seq_len=sequence_lengths[:batch_size].max().item(),
                head_dim=head_dim,
                dtype=self.dtype,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens_ids[:batch_size],
                output_tensor[:batch_size],
                input_meta_data,
                k_caches=k_cache,
                v_caches=v_cache,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        t_capture_end = time.perf_counter()

        if self.verbose:
            self.logger.info(f"CUDA Graph capture time: {t_capture_end - t_capture_begin} s")

    def _verify_args(self) -> None:
        """Verify the input args"""
        if not isinstance(self.inference_config, InferenceConfig):
            raise TypeError("Invalid type of inference config provided.")
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"the model type must be nn.Module, but got {type(self.model)}")
        if not isinstance(self.tokenizer, (PreTrainedTokenizerFast, PreTrainedTokenizer)):
            raise TypeError(
                f"the tokenizer type must be PreTrainedTokenizer or PreTrainedTokenizerFast, but got {type(self.tokenizer)}"
            )
        if isinstance(self.model, ModelWrapper):
            model = self.model.module
        assert (
            model.__class__.__name__ in _supported_models.keys()
        ), f"Model {self.model.__class__.__name__} is not supported."

    def enable_spec_dec(
        self,
        drafter_model: nn.Module = None,
        n_spec_tokens: int = None,
        use_glide_drafter: bool = False,
    ) -> None:
        """Initialize drafter (if it has not yet), and enable Speculative Decoding for subsequent generations.

        Args:
            drafter_model (nn.Module): The drafter model (small model) used to speculate tokens.
                If provided, the previous drafter and drafter model, if exist, will be overwritten.
            n_spec_tokens (Optional[int]): The number of tokens to speculate in each round of speculating-verifying.
                If not provided, `max_n_spec_tokens` in InferenceConfig will be used.
            use_glide_drafter (bool): Whether to use glide model for speculative decoding. Defaults to False.
                If True, the drafter model will be replaced by a glide model.

        ```python
        ...
        engine = InferenceEngine(model, tokenizer, inference_config)

        engine.enable_spec_dec(drafter_model, n_spec_tokens=5)
        engine.generate(...)  # Speculative Decoding

        engine.disable_spec_dec()
        engine.generate(...)  # Normal generation

        engine.enable_spec_dec()
        engine.generate(...)  # Speculative-Decoding using previously set drafter model and number of spec tokens
        engine.clear_spec_dec()
        ```
        """

        if drafter_model is None and self.drafter is None:
            raise ValueError("Drafter not initialized. Please provide a Drafter Model")
        if n_spec_tokens is not None:
            assert 1 < n_spec_tokens <= self.inference_config.max_n_spec_tokens
            self.n_spec_tokens = n_spec_tokens
        if drafter_model is not None:
            assert isinstance(drafter_model, nn.Module)
            # overwrite the drafter, if exists
            self.clear_spec_dec()
            self.drafter_model = drafter_model
            self.drafter = Drafter(
                self.drafter_model,
                self.tokenizer,
                device=self.device,
                dtype=self.dtype,
            )

            # check if the provided drafter model is compatible with GLIDE structure
            # when `use_glide_drafter` is set to True
            if (
                use_glide_drafter
                and hasattr(drafter_model, "model")
                and hasattr(drafter_model.model, "layers")
                and hasattr(drafter_model.model.layers[0], "cross_attn")
            ):
                self.use_glide = use_glide_drafter
            elif use_glide_drafter:
                self.logger.warning(
                    f"`use_glide_drafter` is provided as {use_glide_drafter}, "
                    f"but the provided drafter model is not compatible with GLIDE structure."
                    f"Falling back to use the default drafter model (non-GLIDE)."
                )
        self.request_handler.set_spec_dec_mode(self.n_spec_tokens)
        # using speculative decoding for subsequent generations
        self.use_spec_dec = True

    def disable_spec_dec(self) -> None:
        """Disable using speculative decoding for subsequent generations."""
        self.request_handler.unset_spec_dec_mode()
        # set back to the maximum number of tokens to speculate
        self.n_spec_tokens = self.inference_config.max_n_spec_tokens
        self.use_glide = False
        self.use_spec_dec = False

    def clear_spec_dec(self) -> None:
        """Clear relatable structures of speculative decoding, if exist."""
        if self.use_spec_dec:
            self.disable_spec_dec()
        if self.drafter_model or self.drafter:
            self.drafter_model = None
            self.drafter = None
            torch.cuda.empty_cache()
        self.use_glide = False
        self.use_spec_dec = False

    def steps_spec_dec(self) -> List[Sequence]:
        """
        Run Speculative Decoding steps. This is like retrieving a single batch and launch inference
        with many steps of speculating by a drafter model as well as verifying by a main model.

        Returns:
            List[Sequence]: finished sequences generated by one step.
        """
        batch = self.request_handler.schedule()  # prefill batch
        assert batch.current_batch_size == 1, "Only support bsz 1 for speculative decoding for now."

        input_token_ids, output_tensor, input_meta_data = self.prepare_input(batch)

        if input_meta_data.use_cuda_graph:
            model_executable = self.graph_runners[input_meta_data.batch_size]
        else:
            model_executable = self.model

        # 1. Prefill small model (Drafter) - fill past kv cache for drafter model
        # NOTE For glide drafter models, we won't actually apply glide during prefill stage
        drafter_out = self.drafter.speculate(input_token_ids, 1, None)
        next_token_ids_spec = drafter_out.next_tokens
        drafter_past_key_values = drafter_out.past_key_values

        # 2. Prefill main model (Verifier) - fill past kv cache for main model
        logits = model_executable(input_token_ids, output_tensor, input_meta_data, self.k_cache, self.v_cache)
        next_tokens = search_tokens(self.generation_config, logits, batch_token_ids=batch.batch_token_ids)
        # append new inputs to the batch, temporarily
        batch.append_batch_tokens(next_tokens)
        self.request_handler.allocate_batch_spec_dec(batch, 1)
        already_allocated_kv_len = batch.seq_lengths[0].item()
        input_token_ids = batch.get_1D_inputs_spec_dec(1)

        finished_sequences = self.request_handler.update()

        while True:
            # HACK Retrieve the running batch
            #      Using RequestHandler.schedule here will re-allocate same kv cache for the batch
            batch = self.request_handler.running_bb  # running batch
            assert batch.current_batch_size == 1, "Only support bsz 1 for speculative decoding for now."

            # 3. Decoding - Drafter model speculates `n` tokens
            glide_input = None
            if self.use_glide:
                glide_input = GlideInput(
                    batch.get_block_table_tensor(),
                    self.k_cache[-1],  # use kv cahces of the last layer
                    self.v_cache[-1],
                    batch.get_sequence_lengths(),
                    n_spec_tokens=self.n_spec_tokens,
                )

            drafter_out = self.drafter.speculate(
                input_token_ids,
                self.n_spec_tokens,
                drafter_past_key_values,
                glide_input=glide_input,
            )
            next_token_ids_spec = drafter_out.next_tokens
            drafter_past_key_values = drafter_out.past_key_values
            drafter_spec_length = drafter_out.speculated_length

            for next_token_id_spec in next_token_ids_spec:
                self.request_handler.append_next_tokens(next_token_id_spec.unsqueeze(0))
            cur_length = batch.seq_lengths[0].item()
            if already_allocated_kv_len < cur_length:
                self.request_handler.allocate_batch_spec_dec(batch, n=cur_length - already_allocated_kv_len)
                already_allocated_kv_len = cur_length

            # 4. Decoding - Main model verifies `n` tokens in parallel
            if drafter_spec_length < batch.num_tokens_to_verify:
                batch.set_use_spec_dec(num_tokens_to_verify=drafter_spec_length)
            input_token_ids, output_tensor, input_meta_data = self.prepare_input(batch)
            logits = model_executable(input_token_ids, output_tensor, input_meta_data, self.k_cache, self.v_cache)

            next_tokens = search_tokens(self.generation_config, logits, batch_token_ids=batch.batch_token_ids)

            # 5. Compare and process the results
            diff_indexes = torch.nonzero(~(next_tokens[:-1] == next_token_ids_spec))
            n_matches = drafter_spec_length if diff_indexes.size(0) == 0 else diff_indexes[0][0].item()

            # revoke appended tokens for each Sequence in the current batch
            batch.revoke_batch_tokens(drafter_spec_length - n_matches)  # revoke drafted tokens

            # append the last correct token generated by the main model
            self.request_handler.append_next_tokens(next_tokens[n_matches].unsqueeze(0))

            # trim past key values of the drafter model
            drafter_past_key_values = Drafter.trim_kv_cache(
                drafter_past_key_values, drafter_spec_length - n_matches - 1
            )

            # prepare inputs for the next round of speculation
            n = 1 if n_matches < drafter_spec_length else 2
            input_token_ids = batch.get_1D_inputs_spec_dec(n)

            self.request_handler.update_batch_finished(batch, generation_config=self.generation_config)
            finished_sequences = self.request_handler.update()
            if len(finished_sequences) > 0:
                break

        # Reset back the number of speculated tokens of the batch,
        # this is used to handle the last round of speculation, in which case the number of speculated tokens
        # by the drafter is less than the number of speculated tokens set to the engine.
        batch.set_use_spec_dec(num_tokens_to_verify=self.n_spec_tokens)

        return finished_sequences

    def generate(
        self,
        request_ids: Union[List[int], int] = None,
        prompts: Union[List[str], str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
        return_token_ids: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Union[List[str], Tuple[List[str], List[List[int]]]]:
        """
        Executing the inference step.

        Args:
            request_ids (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (Union[List[int], torch.Tensor, np.ndarray], optional): token ids of input prompts. Defaults to None.
            return_token_ids (bool, optional): Whether to return output token ids. Defaults to False.
            generation_config (Optional[GenerationConfig], optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            Union[List[str], Tuple[List[str], List[List[int]]]]: Inference result returned by one generation.
        """

        gen_config_dict = generation_config.to_dict() if generation_config is not None else {}
        prompts = [prompts] if isinstance(prompts, str) else prompts
        request_ids = [request_ids] if isinstance(request_ids, int) else request_ids

        with torch.inference_mode():
            if prompts is not None or prompts_token_ids is not None:
                self.add_request(
                    request_ids=request_ids,
                    prompts=prompts,
                    prompts_token_ids=prompts_token_ids,
                    **gen_config_dict,
                )

            output_seqs_list = []
            total_tokens_list = []

            # intuition: If user provide a generation config, we should replace the existing one.
            if generation_config is not None:
                self.generation_config = generation_config
                self.generation_config_dict = gen_config_dict

            if self.use_spec_dec:
                assert self.drafter is not None, "Drafter Model is not initialized."
                while self.request_handler.check_unfinished_reqs():
                    output_seqs_list += self.steps_spec_dec()
            else:
                while self.request_handler.check_unfinished_reqs():
                    output_seqs_list += self.step()

            output_seqs_list = sorted(output_seqs_list, key=lambda x: int(x.request_id))

            for seq in output_seqs_list:
                total_tokens_list.append(seq.input_token_id + seq.output_token_id)

            output_str = self.tokenizer.batch_decode(total_tokens_list, skip_special_tokens=True)

            if return_token_ids:
                output_tokens_list = [seq.output_token_id for seq in output_seqs_list]
                return output_str, output_tokens_list
            else:
                return output_str

    @property
    def has_prompt_template(self) -> bool:
        """ """
        return self.inference_config.prompt_template is not None

    def format_prompt(self, prompts: Union[List[str], str]) -> Union[List[str], str]:
        """
        This method will format the input prompt according to the prompt template given to the InferenceConfig.
        """
        assert (
            self.has_prompt_template
        ), "Found the prompt_template is None. Please provide a valid prompt_template in InferenceConfig."

        if isinstance(prompts, (list, tuple)):
            return [self.inference_config.prompt_template.format(input_text=prompt) for prompt in prompts]
        elif isinstance(prompts, str):
            return self.inference_config.prompt_template.format(input_text=prompts)
        else:
            raise TypeError(f"Expected the input prompt to be one of list, tuple, or str, but got {type(prompts)}.")

    def add_request(
        self,
        request_ids: Union[List[int], int] = None,
        prompts: Union[List[str], str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Add requests.

        Args:
            request_ids (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
        """

        # apply the prompt template to the input prompts

        if self.has_prompt_template and prompts is not None:
            prompts = self.format_prompt(prompts)

        block_size = self.inference_config.block_size

        if request_ids is not None and not isinstance(request_ids, list):
            request_ids = [request_ids]

        if prompts is not None and not isinstance(prompts, list):
            prompts = [prompts]

        if prompts_token_ids is None:
            assert prompts, "When the prompts_token_ids is none, the input prompt list must be provided."
            prompts_token_ids = self.tokenizer.batch_encode_plus(prompts, padding=self.inference_config.pad_input)[
                "input_ids"
            ]

        # list of torch Tensor
        if isinstance(prompts_token_ids, list):
            if isinstance(prompts_token_ids[0], torch.Tensor):
                prompts_token_ids = [prompt_token_id.tolist() for prompt_token_id in prompts_token_ids]
        elif isinstance(prompts_token_ids, torch.Tensor) or isinstance(prompts_token_ids, np.ndarray):
            prompts_token_ids = prompts_token_ids.tolist()
        else:
            raise TypeError(
                f"The dtype of prompts_token_ids must be one of list, torch.Tensor, np.ndarray, but got {type(prompts_token_ids)}."
            )

        assert (
            len(prompts_token_ids[0]) <= self.inference_config.max_input_len
        ), f"The length of input prompts {len(prompts_token_ids[0])} must be less than max_input_len {self.inference_config.max_input_len}."

        prompts_num = len(prompts_token_ids)

        for i in range(prompts_num):
            if request_ids:
                assert isinstance(
                    request_ids[0], int
                ), f"The request_id type must be int, but got {type(request_ids[0])}"
                assert len(request_ids) == prompts_num
                request_id = request_ids[i]
            else:
                request_id = next(self.counter)
            if prompts == None:
                prompt = None
            else:
                prompt = prompts[i]

            max_length = kwargs.get("max_length", None)
            max_new_tokens = kwargs.get("max_new_tokens", None)
            if max_length is None and max_new_tokens is None:
                max_new_tokens = self.generation_config.max_new_tokens or self.inference_config.max_output_len
            elif max_length is not None:
                max_new_tokens = max_length - len(prompts_token_ids[i])

            if not self.inference_config.enable_streamingllm:
                assert (
                    self.inference_config.max_output_len >= max_new_tokens
                ), f"max_new_tokens={max_new_tokens} must be less than max_output_len={self.inference_config.max_output_len}."

            sequence = Sequence(
                request_id,
                prompt,
                prompts_token_ids[i],
                block_size,
                None,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                max_output_len=max_new_tokens,
                ignore_eos=self.inference_config.ignore_eos,
            )
            self.request_handler.add_sequence(sequence)

    def prepare_input(self, batch: BatchBucket) -> Tuple[torch.Tensor, torch.Tensor, InputMetaData]:
        input_ids = batch.get_1D_inputs()
        sequence_lengths = batch.get_sequence_lengths()

        if batch.is_prompts:
            n_tokens = sequence_lengths.sum().item()
        else:
            n_tokens = batch.current_batch_size
            if batch.use_spec_dec:
                n_tokens = batch.num_tokens_to_verify + 1
                assert n_tokens == input_ids.size(0)
                n_tokens = n_tokens * batch.current_batch_size
        output_tensor = torch.zeros(
            (n_tokens, batch.num_heads * batch.head_dim), dtype=batch.dtype, device=batch.device
        )

        batch_token_ids = None
        if (
            self.generation_config.repetition_penalty != 1.0
            or self.generation_config.no_repeat_ngram_size > 0
            or self.generation_config.forced_eos_token_id is not None
        ):
            batch_token_ids = batch.batch_token_ids

        # only when we have the graph for specific decoding batch size can we use the cuda graph for inference
        use_cuda_graph = False
        if self.use_cuda_graph and not batch.is_prompts and batch.current_batch_size in self.graph_runners.keys():
            use_cuda_graph = True

        input_meta_data = InputMetaData(
            block_tables=batch.get_block_table_tensor(),
            sequence_lengths=sequence_lengths,
            fd_inter_tensor=batch.fd_inter_tensor,
            batch_size=batch.current_batch_size,
            is_prompts=batch.is_prompts,
            use_cuda_kernel=self.inference_config.use_cuda_kernel,
            use_cuda_graph=use_cuda_graph,
            high_precision=self.high_precision,
            kv_seq_len=sequence_lengths.max().item(),
            head_dim=batch.head_dim,
            dtype=batch.dtype,
            use_spec_dec=batch.use_spec_dec,
            num_tokens_to_verify=batch.num_tokens_to_verify,
            batch_token_ids=batch_token_ids,
        )

        return input_ids, output_tensor, input_meta_data

    def step(self) -> List[str]:
        """
        In each step, do the follows:
            1. Run RequestHandler.schedule() and get the batch used for inference.
            2. Get the input, inputinfo and output placeholder from the batchbucket
            3. Run model to generate the next token
            4. Update waiting list and running list in RequestHandler and get finished sequences.
            5. Decode and return finished sequences.

        Returns:
            List[str]: Decoded finished sequences generated by one step.
        """

        batch = self.request_handler.schedule()

        input_token_ids, output_tensor, input_meta_data = self.prepare_input(batch)

        if input_meta_data.use_cuda_graph:
            model_executable = self.graph_runners[input_meta_data.batch_size]
        else:
            model_executable = self.model

        # TODO: padding_id is used for generating attn_mask and will be removed if nopad version is supported.
        logits = model_executable(input_token_ids, output_tensor, input_meta_data, self.k_cache, self.v_cache)
        if self.inference_config.pad_input:
            logits = logits[:, -1, :]

        if self.inference_config.enable_streamingllm:
            updated_block_ids = batch.streamingllm_update_batch(
                self.inference_config.start_token_size, self.inference_config.generated_token_size
            )
            self.request_handler.streamingllm_free_block_tables(updated_block_ids)

        next_tokens = search_tokens(
            self.generation_config, logits, input_meta_data.is_prompts, batch_token_ids=input_meta_data.batch_token_ids
        )
        self.request_handler.append_next_tokens(next_tokens)
        finished_sequences = self.request_handler.update()

        return finished_sequences
