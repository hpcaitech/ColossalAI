import os
from itertools import count
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.config import InferenceConfig
from colossalai.inference.modeling.policy import model_policy_map
from colossalai.inference.spec import Drafter, GlideInput
from colossalai.inference.struct import Sequence
from colossalai.logging import get_dist_logger
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.base_policy import Policy

from .request_handler import RequestHandler

__all__ = ["InferenceEngine"]

PP_AXIS, TP_AXIS = 0, 1

_supported_models = [
    "LlamaForCausalLM",
]


class InferenceEngine:

    """
    InferenceEngine which manages the inference process..

    Args:
        model (nn.Module): Path or nn.Module of this model.
        tokenizer Optional[(Union[PreTrainedTokenizer, PreTrainedTokenizerFast])]: Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        inference_config: InferenceConfig,
        verbose: bool = False,
        model_policy: Policy = None,
    ) -> None:
        self.inference_config = inference_config
        self.model_config = model.config
        self.model = model
        self.device = torch.device("cuda")
        self.dtype = inference_config.dtype
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._verify_args()

        self.generation_config = inference_config.to_generation_config(self.model_config)
        model.eval()
        model = model.to(self.dtype)
        model = model.to(self.device)

        # Model and relatable attrs of speculative decoding will be set by `enable_spec_dec`
        self.use_spec_dec = False
        self.drafter_model = None
        self.drafter = None
        self.use_glide = False
        self.n_spec_tokens = self.inference_config.max_n_spec_tokens

        if model_policy is None:
            if self.inference_config.pad_input:
                model_type = "padding_" + self.model_config.model_type
            else:
                model_type = "nopadding_" + self.model_config.model_type
            model_policy = model_policy_map[model_type]()

        pg_mesh = ProcessGroupMesh(inference_config.pp_size, inference_config.tp_size)

        self.model = self._shardformer(
            model,
            model_policy,
            None,
            pg_mesh.get_group_along_axis(TP_AXIS) if inference_config.pp_size * inference_config.tp_size > 1 else None,
        )

        self.verbose = verbose
        if verbose:
            self.logger = get_dist_logger(__name__)

        self.request_handler = RequestHandler(self.inference_config, self.model_config)
        self.k_cahce, self.v_cache = self.request_handler.get_kvcache()
        # DISCUSS maybe move this into batch info?

        self.counter = count()

        # Only for testing usage
        self._total_tokens_spec = 0
        self._total_tokens_hit = 0

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
        if self.model.__class__.__name__ not in _supported_models:
            raise ValueError(f"Model {self.model.__class__.__name__} is not supported.")

    def _shardformer(
        self,
        model: nn.Module,
        model_policy: Policy,
        stage_manager: PipelineStageManager = None,
        tp_group: ProcessGroupMesh = None,
    ) -> nn.Module:
        """
        Initialize ShardConfig and replace the model with shardformer.

        Args:
            model (nn.Module): Path or nn.Module of this model.
            model_policy (Policy): The policy to shardformer model which is determined by the model type.
            stage_manager (PipelineStageManager, optional): Used to manage pipeline stages. Defaults to None.
            tp_group (ProcessGroupMesh, optional): Used to manage the process TP group mesh. Defaults to None.

        Returns:
            nn.Module: The model optimized by Shardformer.
        """

        shardconfig = ShardConfig(
            tensor_parallel_process_group=tp_group,
            pipeline_stage_manager=stage_manager,
            enable_tensor_parallelism=(self.inference_config.tp_size > 1),
            enable_fused_normalization=False,
            enable_all_optimization=False,
            enable_flash_attention=False,
            enable_jit_fused=False,
            enable_sequence_parallelism=False,
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, model_policy)
        return shard_model

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
            self.use_glide = use_glide_drafter
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

    def glide_drafter_from_pretrained(self, model_path: Union[str, os.PathLike], config) -> nn.Module:
        """
        Load and prepare a pretrained glide model used as a drafter model, from the given path.

        Usage:
        ```python
        drafter_config = AutoConfig.from_pretrained(drafter_model_path)
        glide_config = GlideLlamaConfig(
            large_hidden_size=4096,
            large_num_attention_heads=32,
            **drafter_config.to_dict(),
        )
        # create a GLIDE drafte model
        drafter_model = engine.glide_drafter_from_pretrained(drafter_model_path, glide_config)
        ```

        Args:
            model_path (Union[str, os.PathLike]): The path to the pretrained glide model.
            config: Glide model config.

        Returns:
            nn.Module: The model ready to be used as a GLIDE drafter model.
        """
        drafter_model = AutoModelForCausalLM.from_pretrained(config)
        # For now, we try to support the same set of base models for glide models (drafter)
        # as those for main models (verifier)
        model_name = drafter_model.__class__.__name__
        if model_name not in _supported_models:
            raise ValueError(f"Model {model_name} is not supported yet as a glide drafter.")

        # get the policy corresponding to the drafter model from policy map
        model_type = drafter_model.config.model_type
        glide_type = f"glide_{model_type}"
        if glide_type not in model_policy_map:
            raise ValueError(f"GLIDE type {glide_type} is not supported yet. Please check the model type {model_type}")
        policy = model_policy_map[glide_type]

        # shard the drafter model add corresponding GLIDE layer
        self._shardformer(drafter_model, policy())

        # load params from the model path
        files = [f for f in os.listdir(model_path) if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".bin")]
        # assume only use a single checkpoint file for drafter model
        file_path = os.path.join(model_path, files[-1])
        state_dict = torch.load(file_path)
        drafter_model.load_state_dict(state_dict)

        return drafter_model

    def steps_spec_dec(self) -> List[Sequence]:
        """
        Run Speculative Decoding steps. This is like retrieving a single batch and launch inference
        with many steps of speculating by a drafter model as well as verifying by a main model.

        Returns:
            List[Sequence]: finished sequences generated by one step.
        """
        batch = self.request_handler.schedule()  # prefill batch

        assert batch.current_batch_size == 1, "Only support bsz 1 for speculative decoding for now."
        input_ids = batch.get_1D_inputs()  # bsz 1 for drafter model

        # 1. Prefill small model (Drafter) - fill past kv cache for drafter model
        # NOTE For glide drafter models, we won't actually apply glide during prefill stage
        drafter_out = self.drafter.speculate(input_ids, 1, None)
        next_token_ids_spec = drafter_out.next_tokens
        drafter_past_key_values = drafter_out.past_key_values

        # 2. Prefill main model (Verifier) - fill past kv cache for main model
        logits = self.model(batch, self.k_cahce, self.v_cache)
        next_tokens = self.request_handler.search_tokens(self.generation_config, logits)
        # append new inputs to the batch, temporarily
        batch.append_batch_tokens(next_tokens)
        self.request_handler.allocate_batch_spec_dec(batch, 1)
        already_allocated_kv_len = batch.seq_lengths[0].item()
        input_ids = batch.get_1D_inputs_spec_dec(1)

        finished_sequences = self.request_handler.update()

        total_tokens_spec = 0
        total_tokens_hit = 0

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
                    self.k_cahce[-1],  # use kv cahces of the last layer
                    self.v_cache[-1],
                    batch.get_sequence_lengths(),
                )

            drafter_out = self.drafter.speculate(
                input_ids,
                self.n_spec_tokens,
                drafter_past_key_values,
                glide_input=glide_input,
            )
            next_token_ids_spec = drafter_out.next_tokens
            drafter_past_key_values = drafter_out.past_key_values
            drafter_spec_length = drafter_out.speculated_length

            total_tokens_spec += drafter_spec_length

            for next_token_id_spec in next_token_ids_spec:
                self.request_handler.append_next_tokens(next_token_id_spec.unsqueeze(0))
            cur_length = batch.seq_lengths[0].item()
            if already_allocated_kv_len < cur_length:
                self.request_handler.allocate_batch_spec_dec(batch, n=cur_length - already_allocated_kv_len)
                already_allocated_kv_len = cur_length

            # 4. Decoding - Main model verifies `n` tokens in parallel
            if drafter_spec_length < batch.num_tokens_to_verify:
                batch.set_use_spec_dec(num_tokens_to_verify=drafter_spec_length)
            logits = self.model(batch, self.k_cahce, self.v_cache)
            next_tokens = self.request_handler.search_tokens(self.generation_config, logits)

            # 5. Compare and process the results
            diff_indexes = torch.nonzero(~(next_tokens[:-1] == next_token_ids_spec))
            n_matches = drafter_spec_length if diff_indexes.size(0) == 0 else diff_indexes[0][0].item()

            total_tokens_hit += n_matches

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
            input_ids = batch.get_1D_inputs_spec_dec(n)

            self.request_handler.update_batch_finished(batch, generation_config=self.generation_config)
            finished_sequences = self.request_handler.update()
            if len(finished_sequences) > 0:
                break

        self._total_tokens_spec += total_tokens_spec
        self._total_tokens_hit += total_tokens_hit
        print(
            f"  Total tokens speculated: {total_tokens_spec}, Total tokens hit: {total_tokens_hit}, Hit Ratio: {total_tokens_hit / total_tokens_spec}"
        )
        print(
            f"Global tokens speculated: {self._total_tokens_spec}, Global tokens hit: {self._total_tokens_hit}, Global Hit Ratio: {self._total_tokens_hit / self._total_tokens_spec}"
        )

        # Reset back the number of speculated tokens of the batch,
        # this is used to handle the last round of speculation, in which case the number of speculated tokens
        # by the drafter is less than the number of speculated tokens set to the engine.
        batch.set_use_spec_dec(num_tokens_to_verify=self.n_spec_tokens)

        return finished_sequences

    def generate(
        self,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
        request_ids: List[int] = None,
        return_token_ids: bool = False,
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Executing the inference step.

        Args:
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
            request_ids (List[int], optional): The request ID. Defaults to None.
            return_token_ids (bool): Whether to return output token ids. Defaults to False.
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """
        with torch.inference_mode():
            if prompts is not None or prompts_token_ids is not None:
                self.add_request(request_ids=request_ids, prompts=prompts, prompts_token_ids=prompts_token_ids)

            output_seqs_list = []
            total_tokens_list = []

            # intuition: If user provide a generation config, we should replace the existing one.
            if generation_config is not None:
                self.generation_config = generation_config

            if self.use_spec_dec:
                assert self.drafter is not None, "Drafter Model is not initialized."
                while self.request_handler.check_unfinished_seqs():
                    output_seqs_list += self.steps_spec_dec()
            else:
                while self.request_handler.check_unfinished_seqs():
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
            return self.inference_config.rompt_template.format(input_text=prompts)
        else:
            raise TypeError(f"Expected the input prompt to be one of list, tuple, or str, but got {type(prompts)}.")

    def add_request(
        self,
        request_ids: List[int] = None,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
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

        if prompts is not None and not isinstance(prompts, list):
            prompts = [prompts]

        if prompts_token_ids is None:
            assert prompts, "When the prompts_token_ids is none, the input prompt list must be provided."
            prompts_token_ids = self.tokenizer.batch_encode_plus(prompts, padding=self.inference_config.pad_input)[
                "input_ids"
            ]

        if isinstance(prompts_token_ids, list):
            pass
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
                if not isinstance(request_ids, list):
                    request_ids = [request_ids]
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

            sequence = Sequence(
                request_id,
                prompt,
                prompts_token_ids[i],
                block_size,
                None,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.inference_config.max_output_len,
            )
            self.request_handler.add_sequence(sequence)

    def step(self) -> List[str]:
        """
        In each step, do the follows:
            1. Run RequestHandler.schedule() and get the batch used for inference.
            2. Run model to generate the next token
            3. Update waiting list and running list in RequestHandler and get finished sequences.
            4. Decode and return finished sequences.

        Returns:
            List[str]: Decoded finished sequences generated by one step.
        """

        batch = self.request_handler.schedule()

        # TODO: padding_id is used for generating attn_mask and will be removed if nopad version is supported.
        logits = self.model(
            batch,
            self.k_cahce,
            self.v_cache,
        )

        if self.inference_config.pad_input:
            logits = logits[:, -1, :]
        next_tokens = self.request_handler.search_tokens(self.generation_config, logits)
        self.request_handler.append_next_tokens(next_tokens)

        finished_sequences = self.request_handler.update()

        return finished_sequences
