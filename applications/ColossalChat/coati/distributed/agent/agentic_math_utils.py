# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A TIR(tool-integrated reasoning) math agent
```bash
python tir_math.py
```
"""
import os
import random

import ray
from qwen_agent.agents import TIRMathAgent
from qwen_agent.llm.base import register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.transformers_llm import Transformers
from qwen_agent.log import logger
from transformers import AutoTokenizer

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), "resource")

# We use the following two systems to distinguish between COT mode and TIR mode
TIR_SYSTEM = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""
COT_SYSTEM = """Please reason step by step, and put your final answer within \\boxed{}."""

from transformers import StoppingCriteria

tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs/share/data/model/Qwen2.5-Math-7B-Instruct", trust_remote_code=True)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last token is one of the stop tokens
        if input_ids[0, -1].item() in self.stop_token_ids:
            return True
        return False


class LocalLLMFromGenerationWorkers:
    """
    A class that wraps the Transformers model to support API-based text generation.
    """

    def __init__(self, generation_worker=None):
        self.device = "cpu"
        self.generation_worker = generation_worker

    def generate(self, **kwargs):
        rollouts = ray.get(self.generation_worker.generate.remote(**kwargs))
        return rollouts["input_ids"]


@register_llm("api_based_transformers")
class CustomTransformers(Transformers):
    """
    Transformers class that supports API-based text generation.
    """

    def __init__(self, cfg: dict, producer_idx, generation_workers=None):
        BaseFnCallModel.__init__(self, cfg)  # skip the super() init of Transformers to avoid loading hf model
        ############ Setup logic from Transformers.__init__ ###############
        if "model" not in cfg:
            raise ValueError("Please provide the model id or directory through `model` in cfg.")

        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast
        except ImportError as e:
            raise ImportError(
                "Could not import classes from transformers. " "Please install it with `pip install -U transformers`"
            ) from e

        self.hf_config = AutoConfig.from_pretrained(cfg["model"])
        arch = self.hf_config.architectures[0]
        if len(self.hf_config.architectures) > 1:
            logger.warning(
                f"The config for the transformers model type contains more than one architecture, choosing the first: {arch}"
            )

        # try loading a processor, if got a tokenizer, regarding the model as text-only
        processor = AutoProcessor.from_pretrained(cfg["model"])
        if isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            logger.info(f"Regarding the transformers model as text-only since its processor is a tokenizer.")
            self.tokenizer = processor
            self._support_multimodal_input = False
        else:
            self.processor = processor
            self.tokenizer = self.processor.tokenizer
            self._support_multimodal_input = True
        ################################################################
        self.generation_workers = generation_workers
        self.hf_models = [
            LocalLLMFromGenerationWorkers(generation_worker=generation_worker)
            for generation_worker in generation_workers
        ]
        self.producer_idx = producer_idx
        self.load_balancer_idx = producer_idx % len(self.generation_workers)

    @property
    def hf_model(self):
        # Simple round-robin load balancing
        model = self.hf_models[self.load_balancer_idx]
        return model

    def _chat_stream(
        self,
        messages,
        delta_stream: bool,
        generate_cfg: dict,
    ):
        # overwrite streaming because streamer is not serializable
        # determine load balancer idx based on producer load, refresh every generation
        load = [ray.get(generation_worker.get_producer_load.remote()) for generation_worker in self.generation_workers]
        min_load = min(load)
        candidates = [i for i, l in enumerate(load) if l == min_load]
        # random tie break
        self.load_balancer_idx = random.choice(candidates)
        response = self._chat_no_stream(messages=messages, generate_cfg=generate_cfg)
        # if self.producer_idx == 0:
        #     print(response)
        yield response


def init_agent_service():
    llm_cfg = {
        # Use the OpenAI-compatible model service provided by DashScope:
        "model": "/mnt/nfs/share/data/model/Qwen2.5-Math-7B-Instruct",
        "model_type": "transformers",
        "generate_cfg": {
            # Using the API's native tool call interface
            "top_k": 1,
        },
    }
    llm = CustomTransformers(llm_cfg)
    bot = TIRMathAgent(llm=llm, name="Qwen2.5-Math", system_message=TIR_SYSTEM)
    return bot


def app_tui():
    # Define the agent
    bot = init_agent_service()

    # Chat
    messages = []
    while True:
        # Query example: 斐波那契数列前10个数字
        query = input("user question: ")
        messages.append({"role": "user", "content": query})
        response = []
        for response in bot.run(messages):
            print("bot response:", response)
        messages.extend(response)


# if __name__ == '__main__':
#     # Test the TIR math agent locally
#     app_tui()
