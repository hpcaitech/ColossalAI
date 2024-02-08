from itertools import count
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

from colossalai.cluster import ProcessGroupMesh
from colossalai.inference.config import InferenceConfig
from colossalai.inference.modeling.policy import model_policy_map
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
        tokenizer: [Union[PreTrainedTokenizer, PreTrainedTokenizerFast]],
        inference_config: InferenceConfig,
        verbose: bool = False,
        model_policy: Policy = None,
    ) -> None:
        assert inference_config, "Please provide inference_config."
        assert tokenizer, "Please provide a tokenizer, either a defined one or str"
        self.inference_config = inference_config
        self.model_config = model.config
        self.device = torch.device("cuda")
        self.dtype = inference_config.dtype
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = inference_config.to_generation_config(self.model_config)
        model = model.eval()
        model.to(self.dtype)

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

    def _verify_config(self) -> None:
        """
        Verify the input config
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"the model type must be nn.Module, but got {type(self.model)}")
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast) and not isinstance(
            self.tokenizer, PreTrainedTokenizer
        ):
            raise TypeError(
                f"the tokenizer type must be PreTrainedTokenizer or PreTrainedTokenizerFast, but got {type(self.tokenizer)}"
            )
        assert (
            self.model.__class__.__name__ in _supported_models
        ), f"Model {self.model.__class__.__name__} is not supported."

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
        return shard_model.cuda()

    def generate(
        self,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Executing the inference step.

        Args:
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """
        with torch.inference_mode():
            self.generation_config = generation_config
            if prompts is not None or prompts_token_ids is not None:
                self.add_request(prompts=prompts, prompts_token_ids=prompts_token_ids)

            output_seqs_list = []
            output_tokens_list = []

            # intuition: If user provide a generation config, we should replace the existing one.
            if generation_config is not None:
                self.generation_config = generation_config

            while self.request_handler.check_unfinished_seqs():
                output_seqs_list += self.step()

            output_seqs_list = sorted(output_seqs_list, key=lambda x: int(x.request_id))

            for seq in output_seqs_list:
                output_tokens_list.append(seq.input_token_id + seq.output_token_id)

            output_str = self.tokenizer.batch_decode(output_tokens_list, skip_special_tokens=True)

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
        requests_id: List[int] = None,
        prompts: List[str] = None,
        prompts_token_ids: Union[List[int], torch.Tensor, np.ndarray] = None,
    ) -> None:
        """
        Add requests.

        Args:
            requests_id (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
        """

        # apply the prompt template to the input prompts
        if self.has_prompt_template and prompts is not None:
            prompts = self.format_prompt(prompts)

        block_size = self.inference_config.block_size

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
            if requests_id:
                request_id = requests_id[i]
            else:
                request_id = next(self.counter)
            if prompts == None:
                prompt = None
            else:
                prompt = prompts[i]

            max_blocks_per_sequence = (
                self.inference_config.max_input_len
                + self.inference_config.max_output_len
                + self.inference_config.block_size
                - 1
            ) // self.inference_config.block_size
            block_table = torch.full([max_blocks_per_sequence], -1, device=self.device)
            sequence = Sequence(
                request_id,
                prompt,
                prompts_token_ids[i],
                block_size,
                None,
                block_table,
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
        self.request_handler.search_tokens(self.generation_config, logits)

        finished_sequences = self.request_handler.update()

        return finished_sequences
