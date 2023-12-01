from logging import Logger
from typing import Optional

from .request_handler import RequestHandler


class InferEngine:
    """
    InferEngine is the core component for Inference.

    It is responsible for launch the inference process, including:
        - Initialize model and distributed training environment(if needed)
        - Launch request_handler and corresponding kv cache manager
        - Receive requests and generate texts.
        - Log the generation process

    Args:
        colossal_config: We provide a unified config api for that wrapped all the configs. You can use it to replace the below configs.
        model_config : The configuration for the model.
        parallel_config: The configuration for parallelize model.
        cache_config : Configuration for initialize and manage kv cache.
        tokenizer (Tokenizer): The tokenizer to be used for inference.
        use_logger (bool): Determine whether or not to log the generation process.
    """

    def __init__(
        self,
        model_config,
        cache_config,
        parallel_config,
        tokenizer,
        use_logger: bool = False,
        colossal_config: Optional["ColossalInferConfig"] = None,
    ) -> None:
        assert colossal_config or (
            model_config and cache_config and parallel_config
        ), "Please provide colossal_config or model_config, cache_config, parallel_config"
        if colossal_config:
            model_config, cache_config, parallel_config = colossal_config

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self._verify_config()

        self._init_model()
        self.request_handler = RequestHandler(cache_config)
        if use_logger:
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
