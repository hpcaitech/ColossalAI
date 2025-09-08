from typing import Any, Dict

import ray
from coati.distributed.agent.agentic import BaseAgenticProducer
from coati.distributed.agent.langgraph_math_agentic_utils import CustomOpenAIAPILLM, LangChainCustomLLM, python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@ray.remote
class LangGraphMathAgenticProducer(BaseAgenticProducer):
    """
    Asyncronous version of the producer that uses vLLM for generation.
    This class is designed to generate agentic response
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
        self.agentic_config = agentic_config
        self.agentic_config.pop("agentic_type", None)
        self.llm_client = CustomOpenAIAPILLM({"model": model_config["path"]}, producer_idx, self.async_producers)
        self.llm = LangChainCustomLLM(self.llm_client)
        # self.python_repl = PythonREPL()
        # repl_tool = Tool(
        #         name="python_repl",
        #         description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        #         func=self.python_repl.run,
        #     )
        # self.tools = [repl_tool]
        self.tools = [python]
        self.memory = MemorySaver()
        self.bot = create_react_agent(self.llm, self.tools, checkpointer=self.memory)

    def _run_agentic_pipeline(self, messages):
        """
        Run the agentic pipeline to generate responses based on the input messages using the LangGraph.
        """
        assert (
            len(messages) == 2 and messages[0]["role"] == "system" and messages[1]["role"] == "user"
        ), "Only support 1 system message and 1 user message as input."
        # inputs = messages
        for event in self.bot.stream(
            {"messages": [("system", messages[0]["content"]), ("user", "calculate the 1000th Fibonacci number")]},
            self.agentic_config,
        ):
            continue
        breakpoint()

        final_state = self.bot.get_state(self.agentic_config)
        transformer_messages = []
        for message in final_state[0]["messages"]:
            tool_calls = None
            if isinstance(message, SystemMessage):
                message.content
            elif isinstance(message, HumanMessage):
                message.content
            elif isinstance(message, AIMessage):
                message.content
                tool_calls = message.get("tool_calls", None)  # [{"type": "function", "function": tool_call}]
            elif isinstance(message, ToolMessage):
                message.content

        return transformer_messages
