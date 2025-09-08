# -------------------------------
# 1. Define the Python tool
# -------------------------------
import copy
import io
import random
import sys
from typing import Dict, List

import ray
import torch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tool_calling_llm import ToolCallingLLM
from transformers import AutoTokenizer

SYSTEM_PROMPT_TEMPLATE = """{task_description}. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

SYSTEM_PROMPT = PromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)


class Capturing(list):
    """Capture stdout prints inside exec()"""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


@tool
def python(code: str) -> str:
    """
    This function executes a string of Python code and returns the printed output.
    You need to print the output. Please import all libraries used in the code string.
    """
    local_vars = {}
    with Capturing() as output:
        exec(code, {}, local_vars)
    if output == []:
        return "Error: No output printed from the code. Please ensure you print the output."
    return "\n".join(output)


# -------------------------------
# 2. Define a Custom API LLM wrapper
# -------------------------------
class CustomOpenAIAPILLM:
    def __init__(self, cfg: dict, producer_idx, generation_workers=None):
        self.producer_idx = producer_idx
        self.generation_workers = generation_workers
        self.load_balancer_idx = producer_idx % len(self.generation_workers)
        assert "model" in cfg, "Please specify the model name in the config"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        self.role_mapping = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "human": "user",
            "tool": "tool",
        }

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        messages: list of {"role": "user"/"assistant"/"system", "content": "..."}
        """
        # load balancing
        load = [ray.get(generation_worker.get_producer_load.remote()) for generation_worker in self.generation_workers]
        min_load = min(load)
        candidates = [i for i, l in enumerate(load) if l == min_load]
        # random tie break
        self.load_balancer_idx = random.choice(candidates)
        generation_worker = self.generation_workers[self.load_balancer_idx]
        transformer_messages = []
        for message in messages:
            transformer_messages.append({"role": self.role_mapping[message.type], "content": message.content})
        input_ids = self.tokenizer.apply_chat_template(
            transformer_messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
        )
        attention_mask = torch.ones_like(input_ids)
        rollouts = ray.get(generation_worker.generate.remote(input_ids, attention_mask, **kwargs))
        response = self.tokenizer.batch_decode(
            rollouts["input_ids"][0][:, input_ids.size(-1) :], skip_special_tokens=True
        )[0]
        return response


class LangChainCustomLLM(ToolCallingLLM, BaseChatModel):
    client: CustomOpenAIAPILLM = None

    def __init__(self, client: CustomOpenAIAPILLM):
        super().__init__()
        self.client = client

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # content = self.client.invoke([m.dict() for m in messages])
        # chat_result = ChatResult(
        #     generations=[ChatGeneration(message=AIMessage(content=content))]
        # )
        print("messages:", messages)
        breakpoint()
        system_message, functions = self._generate_system_message_and_functions(kwargs)
        sample_params = {"stop": stop} if stop is not None else {}
        sample_params.update({k: v for k, v in kwargs.items() if k in ["temperature", "top_p", "top_k", "max_tokens"]})
        messages_ = copy.deepcopy(messages)
        messages_[0].content = messages_[0].content + "\n" + system_message.content
        response_message = self.client.invoke(  # type: ignore[safe-super]
            [system_message] + messages, **{"sample_params": sample_params}
        )
        breakpoint()
        response = self._process_response(AIMessage(content=response_message), functions)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "custom-api-llm"


# -------------------------------
# 3. Build a ReAct Agent with LangGraph
# -------------------------------
def build_agent():
    # Wrap custom API LLM in LangChain-compatible interface

    # Init LLM
    llm_client = CustomOpenAIAPILLM()
    llm = LangChainCustomLLM(llm_client)

    # Tools
    tools = [python]

    # Memory (optional)
    memory = MemorySaver()

    # Build ReAct agent
    agent = create_react_agent(llm, tools, checkpointer=memory)
    return agent


# -------------------------------
# 4. Run the agent on a math problem
# -------------------------------
if __name__ == "__main__":
    agent = build_agent()

    # Example math question
    user_input = "What is the least common multiple of 18 and 24? Use Python if needed."

    config = {"configurable": {"thread_id": "math-1"}}
    for event in agent.stream({"messages": [("user", user_input)]}, config):
        if "agent" in event:
            print("Agent event:", event["agent"]["messages"][-1].content)
        elif "tools" in event:
            print("Tool event:", event["tools"]["messages"][-1].content)

    final_state = agent.get_state(config)
    print("Final Answer:", final_state["messages"][-1].content)
