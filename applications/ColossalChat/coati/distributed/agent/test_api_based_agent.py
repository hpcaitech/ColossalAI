# -------------------------------
# 1. Define the Python tool
# -------------------------------
import io
import sys
from typing import Dict, List

import requests
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


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
class CustomAPILLM:
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {"role": "user"/"assistant"/"system", "content": "..."}
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "custom-model",  # depends on your API
            "messages": messages,
            "temperature": 0,
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Adjust according to your API response format
        return data["choices"][0]["message"]["content"]


# -------------------------------
# 3. Build a ReAct Agent with LangGraph
# -------------------------------
def build_agent():
    # Wrap custom API LLM in LangChain-compatible interface
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage

    class LangChainCustomLLM(BaseChatModel):
        client: CustomAPILLM = None

        def __init__(self, client: CustomAPILLM):
            super().__init__()
            self.client = client

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            content = self.client.invoke([m.dict() for m in messages])
            return self._create_chat_result([AIMessage(content=content)])

        @property
        def _llm_type(self) -> str:
            return "custom-api-llm"

    # Init LLM
    llm_client = CustomAPILLM(api_url="http://localhost:8000/v1/chat/completions")
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
