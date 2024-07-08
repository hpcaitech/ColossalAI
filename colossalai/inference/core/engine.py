from typing import List, Tuple, Type, Union

import numpy as np
import PIL.Image
import torch.nn as nn
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from colossalai.inference.config import InferenceConfig
from colossalai.inference.utils import ModelType, get_model_type
from colossalai.shardformer.policies.base_policy import Policy

__all__ = ["InferenceEngine"]


class InferenceEngine:
    """
    InferenceEngine which manages the inference process..

    Args:
        model_or_path (nn.Module or DiffusionPipeline or str): Path or nn.Module or DiffusionPipeline of this model.
        tokenizer Optional[(Union[PreTrainedTokenizer, PreTrainedTokenizerFast])]: Path of the tokenizer to use.
        inference_config (Optional[InferenceConfig], optional): Store the configuration information related to inference.
        verbose (bool): Determine whether or not to log the generation process.
        model_policy ("Policy"): the policy to shardformer model. It will be determined by the model type if not provided.
    """

    def __init__(
        self,
        model_or_path: Union[nn.Module, str, DiffusionPipeline],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        inference_config: InferenceConfig = None,
        verbose: bool = False,
        model_policy: Union[Policy, Type[Policy]] = None,
    ) -> None:
        self.__dict__["_initialized"] = False  # use __dict__ directly to avoid calling __setattr__
        self.model_type = get_model_type(model_or_path=model_or_path)
        self.engine = None
        if self.model_type == ModelType.LLM:
            from .llm_engine import LLMEngine

            self.engine = LLMEngine(
                model_or_path=model_or_path,
                tokenizer=tokenizer,
                inference_config=inference_config,
                verbose=verbose,
                model_policy=model_policy,
            )
        elif self.model_type == ModelType.DIFFUSION_MODEL:
            from .diffusion_engine import DiffusionEngine

            self.engine = DiffusionEngine(
                model_or_path=model_or_path,
                inference_config=inference_config,
                verbose=verbose,
                model_policy=model_policy,
            )
        elif self.model_type == ModelType.UNKNOWN:
            self.logger.error(f"Model Type either Difffusion or LLM!")

        self._initialized = True
        self._verify_args()

    def _verify_args(self) -> None:
        """Verify the input args"""
        assert self.engine is not None, "Please init Engine first"
        assert self._initialized, "Engine must be initialized"

    def generate(
        self,
        request_ids: Union[List[int], int] = None,
        prompts: Union[List[str], str] = None,
        *args,
        **kwargs,
    ) -> Union[List[Union[str, List[PIL.Image.Image], np.ndarray]], Tuple[List[str], List[List[int]]]]:
        """
        Executing the inference step.

        Args:
            request_ids (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
        """

        assert self.engine is not None, "Please init Engine first"
        return self.engine.generate(request_ids=request_ids, prompts=prompts, *args, **kwargs)

    def add_request(
        self,
        request_ids: Union[List[int], int] = None,
        prompts: Union[List[str], str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Add requests.

        Args:
            request_ids (List[int], optional): The request ID. Defaults to None.
            prompts (Union[List[str], optional): Input prompts. Defaults to None.
            prompts_token_ids (List[List[int]], optional): token ids of input prompts. Defaults to None.
            kwargs: for LLM, it could be max_length, max_new_tokens, etc
                    for diffusion, it could be prompt_2, prompt_3, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, negative_prompt_2, negative_prompt_3, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, clip_skip, which aligns with diffusers
        """
        assert self.engine is not None, "Please init Engine first"
        self.engine.add_request(request_ids=request_ids, prompts=prompts, *args, **kwargs)

    def step(self):
        assert self.engine is not None, "Please init Engine first"
        return self.engine.step()

    def __getattr__(self, name):
        """
        The Design logic of getattr, setattr:
        1. Since InferenceEngine is a wrapper for DiffusionEngine/LLMEngine, we hope to invoke all the member of DiffusionEngine/LLMEngine like we just call the member of InferenceEngine.
        2. When we call the __init__ of InferenceEngine, we don't want to setattr using self.__dict__["xxx"] = xxx, we want to use origin ways like self.xxx = xxx
        So we set the attribute `_initialized`. And after initialized, if we couldn't get the member from InferenceEngine, we will try to get the member from self.engine(DiffusionEngine/LLMEngine)
        """
        if self.__dict__.get("_initialized", False):
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                return getattr(self.engine, name)
        else:
            return self.__dict__[name]

    def __setattr__(self, name, value):
        if self.__dict__.get("_initialized", False):
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                setattr(self.engine, name, value)
        else:
            self.__dict__[name] = value
