from typing import Any, Dict

import ray
from coati.distributed.agent.base import BaseAgenticProducer
from coati.distributed.agent.qwen_math_agentic_utils import TIR_SYSTEM, CustomTransformers
from qwen_agent.agents import TIRMathAgent


@ray.remote
class QwenMathAgenticProducer(BaseAgenticProducer):
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
        self.agentic_config = model_config if not agentic_config else agentic_config
        self.agentic_config.update({"model": model_config["path"]})
        self.llm = CustomTransformers(self.agentic_config, self.producer_idx, generation_workers=self.async_producers)
        self.bot = TIRMathAgent(llm=self.llm, name=model_config["path"], system_message=TIR_SYSTEM)

    def _run_agentic_pipeline(self, messages):
        """
        Run the agentic pipeline to generate responses based on the input messages using the TIRMathAgent.
        """
        for response in self.bot.run(messages):
            continue
        messages.extend(response)
        # breakpoint()
        return messages
