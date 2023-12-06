from logging import Logger
from typing import Optional

from .request_handler import RequestHandler
from ..config import ColossalInferConfig
from transformers import AutoConfig, PretrainedConfig


class InferEngine:
    """
    InferEngine is the core component for Inference.

    It is responsible for launch the inference process, including:
        - Initialize model and distributed training environment(if needed)
        - Launch request_handler and corresponding kv cache manager
        - Receive requests and generate texts.
        - Log the generation process

    Args:
        tokenizer: Path of the tokenizer to use.
        colossal_config: We provide a unified config api for that wrapped all the configs. You can use it to replace the below configs.
        use_logger (bool): Determine whether or not to log the generation process.
    """

    def __init__(
        self,
        tokenizer: str = "",
        colossal_config: Optional["ColossalInferConfig"] = None,
        use_logger: bool = False,
    ) -> None:
        
        assert colossal_config, "Please provide colossal_config."

        self._init_model()
        self.request_handler = RequestHandler()
        self.tokenizer = tokenizer
        self.hf_model_config = self._get_hf_model_config()
        if use_logger:
            self.logger = Logger()

    def _init_model(self):
        """
        Initialize model and distributed training environment(if needed).
        May need to provide two different initialization methods:
            1. 用户自定义(from local path)
            2. 从checkpoint加载(hugging face)
        """
        
    def _get_hf_model_config(self) -> PretrainedConfig:
        """
        Get huggingface config.

        Returns:
            PretrainedConfig: The huggingface configuration object of imput model. 
        """
        return AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code, revision=self.revision
        )

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
