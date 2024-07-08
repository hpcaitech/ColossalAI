"""
API and LLM warpper class for running LLMs locally

Usage:

import os
model_path = os.environ.get("ZH_MODEL_PATH")
model_name = "chatglm2"
colossal_api = ColossalAPI(model_name, model_path)
llm = ColossalLLM(n=1, api=colossal_api)
TEST_PROMPT_CHATGLM="续写文章：惊蛰一过，春寒加剧。先是料料峭峭，继而雨季开始，"
logger.info(llm(TEST_PROMPT_CHATGLM, max_new_tokens=100), verbose=True)

"""

from typing import Any, List, Mapping, Optional

import torch
from colossalqa.local.utils import get_response, post_http_request
from colossalqa.mylogging import get_logger
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger()


class ColossalAPI:
    """
    API for calling LLM.generate
    """

    __instances = dict()

    def __init__(self, model_type: str, model_path: str, ckpt_path: str = None) -> None:
        """
        Configure model
        """
        if model_type + model_path + (ckpt_path or "") in ColossalAPI.__instances:
            return
        else:
            ColossalAPI.__instances[model_type + model_path + (ckpt_path or "")] = self
        self.model_type = model_type
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)
            self.model.load_state_dict(state_dict)
        self.model.to(torch.cuda.current_device())

        # Configure tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.model.eval()

    @staticmethod
    def get_api(model_type: str, model_path: str, ckpt_path: str = None):
        if model_type + model_path + (ckpt_path or "") in ColossalAPI.__instances:
            return ColossalAPI.__instances[model_type + model_path + (ckpt_path or "")]
        else:
            return ColossalAPI(model_type, model_path, ckpt_path)

    def generate(self, input: str, **kwargs) -> str:
        """
        Generate response given the prompt
        Args:
            input: input string
            **kwargs: language model keyword type arguments, such as top_k, top_p, temperature, max_new_tokens...
        Returns:
            output: output string
        """
        if self.model_type in ["chatglm", "chatglm2"]:
            inputs = {
                k: v.to(torch.cuda.current_device()) for k, v in self.tokenizer(input, return_tensors="pt").items()
            }
        else:
            inputs = {
                "input_ids": self.tokenizer(input, return_tensors="pt")["input_ids"].to(torch.cuda.current_device())
            }

        output = self.model.generate(**inputs, **kwargs)
        output = output.cpu()
        prompt_len = inputs["input_ids"].size(1)
        response = output[0, prompt_len:]
        output = self.tokenizer.decode(response, skip_special_tokens=True)
        return output


class VllmAPI:
    def __init__(self, host: str = "localhost", port: int = 8077) -> None:
        # Configure api for model served through web
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}/generate"

    def generate(self, input: str, **kwargs):
        output = get_response(post_http_request(input, self.url, **kwargs))[0]
        return output[len(input) :]


class ColossalLLM(LLM):
    """
    Langchain LLM wrapper for a local LLM
    """

    n: int
    api: Any
    kwargs = {"max_new_tokens": 100}

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        logger.info(f"kwargs:{kwargs}\nstop:{stop}\nprompt:{prompt}", verbose=self.verbose)
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]

        generate_args = {k: kwargs[k] for k in kwargs if k not in ["stop", "n"]}
        out = self.api.generate(prompt, **generate_args)
        if isinstance(stop, list) and len(stop) != 0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        logger.info(f"{prompt}{out}", verbose=self.verbose)
        return out

    @property
    def _identifying_params(self) -> Mapping[str, int]:
        """Get the identifying parameters."""
        return {"n": self.n}

    def get_token_ids(self, text: str) -> List[int]:
        """Return the ordered ids of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of ids corresponding to the tokens in the text, in order they occur
                in the text.
        """
        # use the colossal llm's tokenizer instead of langchain's cached GPT2 tokenizer
        return self.api.tokenizer.encode(text)


class VllmLLM(LLM):
    """
    Langchain LLM wrapper for a local LLM
    """

    n: int
    api: Any
    kwargs = {"max_new_tokens": 100}

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        for k in self.kwargs:
            if k not in kwargs:
                kwargs[k] = self.kwargs[k]
        logger.info(f"kwargs:{kwargs}\nstop:{stop}\nprompt:{prompt}", verbose=self.verbose)
        generate_args = {k: kwargs[k] for k in kwargs if k in ["n", "max_tokens", "temperature", "stream"]}
        out = self.api.generate(prompt, **generate_args)
        if len(stop) != 0:
            for stopping_words in stop:
                if stopping_words in out:
                    out = out.split(stopping_words)[0]
        logger.info(f"{prompt}{out}", verbose=self.verbose)
        return out

    def set_host_port(self, host: str = "localhost", port: int = 8077, **kwargs) -> None:
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 100
        self.kwargs = kwargs
        self.api = VllmAPI(host=host, port=port)

    @property
    def _identifying_params(self) -> Mapping[str, int]:
        """Get the identifying parameters."""
        return {"n": self.n}
