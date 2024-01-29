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

PP_AXIS, TP_AXIS = 0, 1

_supported_models = [
    "LlamaForCausalLM",
]


class InferenceEngine:

    """
    InferenceEngine which manages the inference process..

    Args:
        model (nn.Module): Path or nn.Module of this model.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        inference_config: Optional["InferenceConfig"] = None,
        verbose: bool = False,
        model_policy: Policy = None,
    ) -> None:
        assert inference_config, "Please provide inference_config."
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_config = inference_config
        self.model_config = model.config
        self.device = torch.device("cuda")
        self.dtype = inference_config.dtype

        model = model.eval()
        model.to(self.dtype)

        if model_policy is None:
            model_policy = model_policy_map[self.model_config.model_type]()

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
            nn.Module: _description_
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
            extra_kwargs={"quant": self.inference_config.quant_mode},
        )
        shardformer = ShardFormer(shard_config=shardconfig)
        shard_model, _ = shardformer.optimize(model, model_policy)
        return shard_model.cuda()

    def generate(
        self,
        generation_config: GenerationConfig = None,
    ) -> List[str]:
        """
        Executing the inference step.

        Args:
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """

        self.generation_config = generation_config

        output_list = []

        while self.request_handler.check_unfinished_seqs():
            output_list += self.step()

        return output_list

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

        block_size = self.inference_config.block_size

        if prompts_token_ids is None:
            assert prompts, "When the prompts_token_ids is none, the input prompt list must be provided."
            prompts_token_ids = self.tokenizer.batch_encode_plus(prompts, padding=True)["input_ids"]

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

        output_list = []
        batch = self.request_handler.schedule()

        # TODO: padding_id is used for generating attn_mask and will be removed if nopad version is supported.
        logits = self.model(
            batch,
            self.k_cahce,
            self.v_cache,
        )

        logits = logits[:, -1, :]
        self.request_handler.search_tokens(self.generation_config, logits)
        finished_sequences = self.request_handler.update()

        # Decode completed sentences.
        # TODO : update decoding step
        for seq in finished_sequences:
            output_str = self.tokenizer.decode(seq.input_token_id + seq.output_token_id, skip_special_tokens=True)
            output_list.append(output_str)

        return output_list
