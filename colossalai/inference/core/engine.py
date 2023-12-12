from logging import Logger
from typing import Optional

from transformers import AutoConfig

from colossalai.inference.config import InferenceConfig


class InferenceEngine:
    """
    InferenceEngine is the core component for Inference.

    It is responsible for launch the inference process, including:
        - Initialize model and distributed training environment(if needed)
        - Launch request_handler and corresponding kv cache manager
        - Receive requests and generate texts.
        - Log the generation process

    Args:
        tokenizer: Path of the tokenizer to use.
        inference_config: We provide a unified config api for that wrapped all the configs. You can use it to replace the below configs.
        verbose (bool): Determine whether or not to log the generation process.
    """

    def __init__(
        self,
        tokenizer: str = None,
        inference_config: Optional["InferenceConfig"] = None,
        verbose: bool = False,
    ) -> None:
        assert inference_config, "Please provide inference_config."

        self._init_model()
        # cache_config may need to be modified later.
        # self.request_handler = RequestHandler(cache_config)
        self.tokenizer = tokenizer
        self.hf_model_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code, revision=self.revision
        )
        if verbose:
            self.logger = Logger()

    def _init_model(self):
        """
        Initialize model and distributed training environment(if needed).
        May need to provide two different initialization methods:
            1. 用户自定义(from local path)
            2. 从checkpoint加载(hugging face)
        """

    def _verify_config(self):
        """
        Verify the configuration to avoid potential bugs.
        """

    def generate(self):
        pass

    def step(self):
        """
        In each step, do the follows:
            1. Run request_handler to update the kv cache and running input_ids
            2. Run model to generate the next token
            3. Check whether there is finied request and decode
        """
