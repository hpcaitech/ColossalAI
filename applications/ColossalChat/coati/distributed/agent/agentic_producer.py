import copy
import random
import re
from typing import Any, Dict
from uuid import uuid4

import ray
from coati.distributed.agent.base import BaseAgenticProducer
from transformers import AutoTokenizer

DEFAULT_SYSTEM_MESSAGE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively, i.e., <reason> reasoning process here </reason><answer> answer here </answer>."""


@ray.remote
class AgenticProducer(BaseAgenticProducer):
    """
    Asyncronous version of the producer that uses vLLM for generation.
    This class is designed to generate agentic response

    Please use the following SYSTEM message or a similar one for the agentic math model:
    '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The Assistant first thinks about the reasoning process in the mind and then provides the user with
    the answer. The reasoning process and answer are enclosed within <reason> </reason> and <answer>
    </answer> tags, respectively, i.e., <reason> reasoning process here </reason><answer> answer here </answer>.'''
    """

    def __init__(
        self,
        producer_idx,
        num_producers,
        num_consumer_procs,
        num_episodes,
        batch_size,
        train_dataset_config,
        model_config,
        generate_config,
        async_producers,
        tool_workers=[],
        tokenizer_config=None,
        agentic_config=None,
        microbatch_size=1,
        backend="transformers",
        num_generations: int = 8,
        consumer_plugin_config=None,
        eval_dataset_config=None,
        eval_interval=-1,  # disable evaluation
        grpo_config: Dict[str, Any] = None,
        eval_save_dir: str = "./eval",
        eval_generation_config={},
        project_name: str = None,
        run_name: str = None,
        wandb_group_name: str = None,
        log_rollout_interval: int = 20,
        rollout_log_file: str = "./rollout_log.jsonl",
        enable_profiling: bool = False,
        n_behind: int = 0,
    ):
        assert microbatch_size == 1  # microbatch_size must be 1 for agentic producer
        assert batch_size == 1  # batch_size must be 1 for agentic producer
        super().__init__(
            producer_idx,
            num_producers,
            num_consumer_procs,
            num_episodes,
            batch_size,
            train_dataset_config,
            model_config,
            generate_config,
            async_producers,
            tokenizer_config,
            microbatch_size,
            backend,
            num_generations,
            consumer_plugin_config,
            eval_dataset_config=eval_dataset_config,
            eval_interval=eval_interval,
            grpo_config=grpo_config,
            eval_save_dir=eval_save_dir,
            eval_generation_config=eval_generation_config,
            project_name=project_name,
            run_name=run_name,
            wandb_group_name=wandb_group_name,
            log_rollout_interval=log_rollout_interval,
            rollout_log_file=rollout_log_file,
            enable_profiling=enable_profiling,
            n_behind=n_behind,
        )
        self.tool_workers = tool_workers
        self.agentic_config = model_config if not agentic_config else agentic_config
        self.agentic_config.update({"model": model_config["path"]})
        tokenizer_path = None
        if tokenizer_config and "path" in tokenizer_config:
            tokenizer_path = tokenizer_config["path"]
        elif "path" in model_config:
            tokenizer_path = model_config["path"]
        assert tokenizer_path is not None, "Tokenizer path must be provided either in tokenizer_config or model_config."
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tools_schema = []
        self.tool_call_budget = self.agentic_config.get("tool_call_budget", 3)
        self.llm_call_budget = self.agentic_config.get("llm_call_budget", 10)
        self.async_llm_engine_map = {}
        self._get_tools()

    def _get_tools(self):
        """
        SYSTEM message for the agentic math model. Reference: r-start2 paper https://arxiv.org/pdf/2508.20722
        """
        tools = ray.get(self.tool_workers[0].list_tools.remote())
        tool_descriptions = {tool: ray.get(self.tool_workers[0].get_tool_description.remote(tool)) for tool in tools}
        tool_arg_schemas = {tool: ray.get(self.tool_workers[0].get_args_schema.remote(tool)) for tool in tools}
        self.tools = []
        for tool in tools:
            tool_schema = {"name": tool, "description": tool_descriptions[tool], "parameters": tool_arg_schemas[tool]}
            self.tools.append(tool_schema)

    def _build_prompt(
        self, messages, add_generation_prompt: bool = True, return_dict=True, return_tensors="pt"
    ) -> dict:
        """
        Build the prompt for the agentic math model.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors=return_tensors,
        )

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the agentic math model.

        Sample Assistant Response:
        The tool indicates that Singapore’s weather today is 31°C with partly cloudy skies and light showers. \\\\boxed{It is warm and slightly rainy in Singapore today.}<|im_end|>

        Sample Assistant Response with Tool Call:
        To answer this, I will check both the weather and the timezone for New York.\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "New York"}}\n</tool_call>\n<tool_call>\n{"name": "get_timezone", "arguments": {"location": "New York"}}\n</tool_call>

        Sample Ouput:
        {
            "role": "assistant",
            "content": "Let me check the current weather in Singapore by calling the weather tool.",
            "tool_calls": [
                {
                    "function": {
                    "name": "get_weather",
                    "arguments": {
                        "location": "New York"
                    }
                    }
                },
                {
                    "function": {
                    "name": "get_timezone",
                    "arguments": {
                        "location": "New York"
                    }
                    }
                }
            ]
        },
        {
            "role": "assistant",
            "content": "The tool indicates that Singapore’s weather today is 31°C with partly cloudy skies and light showers. \\\\boxed{It is warm and slightly rainy in Singapore today.}"
        }
        """
        # split by <im_end|>
        response_chunked = response.split("<|im_end|>")[0].strip()
        if "<tool_call>" in response_chunked:
            assistant_content = response_chunked.split("<tool_call>")[0].strip()
            tool_call_sections = response_chunked[response_chunked.find("<tool_call>") :].strip()
            # extract all tool calls
            tool_calls = []
            pattern = "<tool_call>(.*?)</tool_call>"
            matches = re.findall(pattern, tool_call_sections, re.DOTALL)
            for match in matches:
                try:
                    tool_call = eval(match.strip())
                    name = tool_call["name"]
                    arguments = tool_call["arguments"]
                    tool_calls.append({"function": {"name": name, "arguments": arguments}})
                except Exception as e:
                    print(f"Failed to parse tool call: {match.strip()}. Error: {e}")
                    tool_calls.append({"function": {"name": "return_parsing_error", "arguments": {}}})
        else:
            assistant_content = response_chunked
            tool_calls = []
        assistant_message = {"role": "assistant", "content": assistant_content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    def _select_tool_worker(self) -> ray.actor.ActorHandle:
        """
        Select a tool worker based on the current load.
        """
        loads = ray.get([worker.get_load.remote() for worker in self.tool_workers])
        min_load = min(loads)
        candidates = [i for i, l in enumerate(loads) if l == min_load]
        selected_idx = random.choice(candidates)  # random tie break
        ray.get(self.tool_workers[selected_idx].increase_load.remote())
        return self.tool_workers[selected_idx]

    def _select_async_producer(self, request_id) -> ray.actor.ActorHandle:
        """
        Select an async producer based on the current load.
        """
        # use the last used async producer if exists to reuse kv cache (as vllm use paged kv cache,
        # it will reuse most of the kv cache pages without recomputation)
        if request_id in self.async_llm_engine_map:
            return self.async_producers[self.async_llm_engine_map[request_id]]
        # otherwise select the least loaded async producer
        loads = ray.get([proc.get_producer_load.remote() for proc in self.async_producers])
        min_load = min(loads)
        candidates = [i for i, l in enumerate(loads) if l == min_load]
        selected_idx = random.choice(candidates)  # random tie break
        self.async_llm_engine_map[request_id] = selected_idx
        return self.async_producers[selected_idx]

    def _run_agentic_pipeline(self, messages):
        """
        Run the agentic pipeline to generate responses based on the input messages.
        """
        tool_call_count = 0
        llm_call_count = 0
        num_prompt_tokens = 0
        request_id = str(uuid4())
        logprobs = None
        while True:
            # tokenize the messages
            if llm_call_count > self.llm_call_budget:
                print(f"LLM call budget exceeded: {llm_call_count} > {self.llm_call_budget}. Stopping.")
                del self.async_llm_engine_map[request_id]
                while messages[-1]["role"] == "tool":
                    messages.pop()
                return messages, logprobs
            inputs = self._build_prompt(messages, return_dict=True, return_tensors="pt")
            if num_prompt_tokens == 0:
                num_prompt_tokens = inputs["input_ids"].size(-1)
            if inputs["input_ids"].size(-1) - num_prompt_tokens > self.generate_config["max_tokens"]:
                print(
                    f"Max tokens exceeded: Current have generated {inputs['input_ids'].size(-1) - num_prompt_tokens} tokens > {self.generate_config.get('max_tokens', 512)}. Stopping."
                )
                del self.async_llm_engine_map[request_id]
                while messages[-1]["role"] == "tool":
                    messages.pop()
                return messages, logprobs
            async_producer = self._select_async_producer(request_id=request_id)
            agentic_generate_config = copy.deepcopy(self.generate_config)
            agentic_generate_config["max_tokens"] = self.agentic_config.get("max_tokens", 2048)
            response = ray.get(
                async_producer.generate.remote(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    request_id=request_id,
                    **agentic_generate_config,
                )
            )
            llm_call_count += 1
            response_input_ids = response["input_ids"]
            logprobs = response["action_log_probs"]
            response_text = self.tokenizer.decode(
                response_input_ids[0][0][inputs["input_ids"].size(-1) :], skip_special_tokens=False
            )
            assistant_message = self._parse_response(response_text)
            messages.append(assistant_message)
            if "tool_calls" in assistant_message:
                if tool_call_count > self.tool_call_budget:
                    print(f"Tool call budget exceeded: {tool_call_count} > {self.tool_call_budget}. Stopping.")
                    del self.async_llm_engine_map[request_id]
                    return messages, logprobs
                tool_call_count += len(assistant_message["tool_calls"])
                handlers = []
                for tool_call in assistant_message["tool_calls"]:
                    # select a tool worker to execute the tool call
                    tool_worker = self._select_tool_worker()
                    handler = tool_worker.call.remote(tool_call["function"]["name"], tool_call["function"]["arguments"])
                    handlers.append(handler)
                tool_results = ray.get(handlers)
                for tool_call, tool_result in zip(assistant_message["tool_calls"], tool_results):
                    tool_message = {"role": "tool", "content": str(tool_result)}
                    messages.append(tool_message)
            else:
                # no further tool call, return the messages
                del self.async_llm_engine_map[request_id]
                return messages, logprobs
