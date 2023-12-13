from itertools import count
from typing import List, Optional

import torch.nn as nn
from transformers import AutoConfig, GenerationConfig

from colossalai.logging import get_dist_logger

from ..kv_cache.kvcache_manager import KVCacheManager
from .config import InferenceConfig
from .get_tokenizer import get_tokenizer
from .inference_struct import Sequence
from .init_model import init_model
from .request_handler import RequestHandler


class InferenceEngine:

    """
        InferenceEngine which manages the inference process.
    .

        Args:
            inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
            verbose (bool): Determine whether or not to log the generation process.
    """

    def __init__(
        self,
        inference_config: Optional["InferenceConfig"] = None,
        verbose: bool = False,
    ) -> None:
        assert inference_config, "Please provide inference_config."

        self.tokenizer = get_tokenizer(
            inference_config.tokenizer,
            use_fast_tokenizer=inference_config.use_fast_tokenizer,
            trust_remote_code=inference_config.trust_remote_code,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.inference_config = inference_config

        self.verbose = verbose
        if verbose:
            self.logger = get_dist_logger(__name__)

        self._init_model_and_hf_config()
        self.cache_manager = KVCacheManager(self.inference_config, self.hf_model_config, verbose)
        self.requset_handler = RequestHandler(self.inference_config)

        self.counter = count()
        self._verify_config()

    def _init_model_and_hf_config(self):
        """
        Initialize model.
        """

        if self.verbose:
            self.logger.info("Start to initialize model")

        if isinstance(self.inference_config.model, str):
            self.model = init_model(self.inference_config, self.hf_model_config)
            self.hf_model_config = AutoConfig.from_pretrained(
                self.inference_config.model,
                trust_remote_code=self.inference_config.trust_remote_code,
                revision=self.inference_config.revision,
            )
        elif isinstance(self.inference_config.model, nn.Module):
            self.model = self.inference_config.model
            self.hf_model_config = self.model.config
        else:
            raise ValueError(
                f"The type of inference_config.model should be str or nn.Module, but get {type(self.inference_config.model)}"
            )

    def _verify_config(self):
        """
        Verify the configuration to avoid potential bugs.
        """

    def generate(
        self,
        generation_config: GenerationConfig = None,
    ) -> List[str]:
        """executing the inference step.

        Args:
            generation_config (GenerationConfig, optional): Huggingface GenerationConfig used for inference. Defaults to None.

        Returns:
            List[str]: Inference result returned by one generation.
        """

        self.generation_config = generation_config

        output_list = []

        while self.requset_handler.check_unfinished_seqs():
            output_list += self.step()

        return output_list

    def add_request(
        self,
        requests_id: List[int] = None,
        prompts: List[str] = None,
        prompts_token_ids: List[int] = None,
    ) -> None:
        """Add requests.

        Args:
            requests_id (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
        """

        block_size = self.inference_config.block_size

        if prompts_token_ids is None:
            assert prompts, "When the prompts_token_ids is none, the input prompt list must be provided."
            prompts_token_ids = []
            for prompt in prompts:
                prompts_token_ids.append(self.tokenizer.encode(prompt))

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
            sequence = Sequence(
                request_id,
                prompt,
                prompts_token_ids[i],
                block_size,
                None,
                None,
                self.tokenizer.eos_token_id,
                self.inference_config.max_output_len,
            )
            self.requset_handler.add_sequence(sequence)

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

        if self.verbose:
            self.logger.info("Running generation step")

        output_list = []
        self.requset_handler.schedule()

        # Uncomment if the development of RequestHandler is completed.
        # logits = self.model(batch)
        # self.requset_handler.search_tokens(logits, self.generation_config)

        finished_sequences = self.requset_handler.update()

        # Decode completed sentences.
        for seq in finished_sequences:
            if seq.prompt:
                output_str = self.tokenizer.decode(seq.output_token_id, skip_special_tokens=True)
                output_list.append(seq.prompt + output_str)
            else:
                output_str = self.tokenizer.decode(seq.input_token_id + seq.output_token_id, skip_special_tokens=True)
                output_list.append(output_str)

        return output_list
