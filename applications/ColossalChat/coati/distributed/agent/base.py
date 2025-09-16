import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import ray
import torch
from coati.distributed.producer import BaseProducer
from vllm import SamplingParams


class BaseAgenticProducer(BaseProducer):
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
            tokenizer_config,
            microbatch_size,
            backend,
            consumer_plugin_config,
            eval_dataset_config=eval_dataset_config,
            eval_interval=eval_interval,
            grpo_config=grpo_config,
            eval_save_dir=eval_save_dir,
            project_name=project_name,
            run_name=run_name,
            wandb_group_name=wandb_group_name,
            log_rollout_interval=log_rollout_interval,
            rollout_log_file=rollout_log_file,
            enable_profiling=enable_profiling,
            n_behind=n_behind,
            enable_agentic=True,
        )
        self.eval_generation_config = copy.deepcopy(generate_config)
        self.eval_generation_config["n"] = 1  # use 1 generation for evaluation
        self.eval_generation_config.update(eval_generation_config)
        self.eval_sample_params = SamplingParams(**self.eval_generation_config)
        self.async_producers = async_producers
        self.num_generations = num_generations
        self.generate_config = generate_config

    def _run_agentic_pipeline(self, messages):
        """
        Run the agentic pipeline to generate responses based on the input messages.
        This function should be implemented in subclasses.
        """
        raise NotImplementedError

    def _build_prompt(
        self, messages, add_generation_prompt: bool = True, return_dict=True, return_tensors="pt"
    ) -> dict:
        """
        Build the prompt from the input messages.
        This function should be implemented in subclasses.
        """
        raise NotImplementedError

    def rollout(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Rollout function to generate responses for the input, for example, using LLM or agentic pipeline.
        This function should be implemented in subclasses.
        """
        assert len(kwargs["messages"]) == 1, "Only support batch size of 1 for agentic producer"
        messages = kwargs["messages"][0]
        prompt_input_ids = self._build_prompt(
            messages, return_dict=True, return_tensors="pt", add_generation_prompt=True
        )["input_ids"]
        # add left padding
        prompt_length = prompt_input_ids.shape[1]
        max_prompt_length = self.train_dataset_config["max_length"]
        to_pad_left = max_prompt_length - prompt_length
        rollouts = {
            "input_ids": [],
            "attention_mask": [],
            "action_mask": [],
            "action_log_probs": [],
            "response_idx": [],
        }
        with ThreadPoolExecutor(max_workers=self.num_generations) as executor:
            results = list(
                executor.map(self._run_agentic_pipeline, [copy.deepcopy(messages) for _ in range(self.num_generations)])
            )

        for i in range(self.num_generations):
            _messages, logprobs = results[i]
            response_input_ids = self._build_prompt(
                _messages, return_dict=True, return_tensors="pt", add_generation_prompt=False
            )["input_ids"]
            # truncate if too long
            response_input_ids = response_input_ids[:, : self.grpo_config["max_length"] - to_pad_left]
            # add left right padding
            to_pad_right = self.grpo_config["max_length"] - response_input_ids.shape[1] - to_pad_left
            response_length = response_input_ids.shape[1] - prompt_length
            input_ids = torch.nn.functional.pad(
                response_input_ids, (to_pad_left, to_pad_right), "constant", value=self.tokenizer.pad_token_id
            )  # [1, max_length]
            attention_mask = torch.nn.functional.pad(
                torch.ones_like(response_input_ids), (to_pad_left, to_pad_right), "constant", value=0
            )  # [1, max_length]
            action_mask = torch.nn.functional.pad(
                torch.ones(size=(1, response_length)), (0, to_pad_right), "constant", value=0
            )  # [1, max_length-prompt_length]
            rollouts["attention_mask"].append(attention_mask)
            rollouts["action_mask"].append(action_mask)
            truncated_logprobs = logprobs[:, :, prompt_length : prompt_length + self.generate_config["max_tokens"]]
            logprobs_padded = torch.nn.functional.pad(
                truncated_logprobs,
                (0, self.generate_config["max_tokens"] - truncated_logprobs.size(-1)),
                "constant",
                value=0.0,
            )  # [1, max_new_tokens]
            rollouts["action_log_probs"].append(logprobs_padded[0])
            rollouts["response_idx"].append(
                torch.tensor(
                    [
                        [
                            self.train_dataset_config["max_length"],
                            self.train_dataset_config["max_length"] + response_length,
                        ]
                    ]
                )
            )  # [1, 2]
            rollouts["input_ids"].append(input_ids)
        rollouts = {k: torch.cat(v, dim=0).unsqueeze(0) for k, v in rollouts.items()}  # [num_generations, ...]
        rollouts["temperature"] = torch.tensor([self.agentic_config.get("temperature", 1.0)])
        if hasattr(self, "rollout_log_file") and self.producer_idx == 0 and not self.eval_mode:
            # for agentic producer, AsyncSimpleProducer is not the main producer, so we don't log rollouts
            if (
                self.consumer_global_step - self.latest_rollout_log_step >= self.log_rollout_interval
                or self.latest_rollout_log_step == -1
            ):
                new_record = (
                    json.dumps(
                        {
                            "train_step": self.consumer_global_step,
                            "rollout": self.tokenizer.batch_decode(
                                rollouts["input_ids"][:, 0], skip_special_tokens=True
                            ),
                        }
                    )
                    + "\n"
                )
                self.rollout_log_file.write(new_record)
                self.rollout_log_file.flush()
                self.latest_rollout_log_step = self.consumer_global_step

        if "gt_answer" in kwargs:
            rollouts["gt_answer"] = kwargs["gt_answer"]
        if "test_cases" in kwargs:
            rollouts["test_cases"] = kwargs["test_cases"]
        return rollouts

    def sync_model(self, episode, step) -> None:
        """
        sync model from consumer to self.async_producers
        AgenticProducer does not hold any model weights, so no need to sync model to self.async_producers
        """
        tasks = []
        for proc in self.async_producers:
            tasks.append(proc.async_sync_model.remote(episode, step, self.num_producers))
        ray.get(tasks)
        return

    def sync_data(self, data: Dict[str, torch.Tensor]) -> None:
        """
        sync data from self to consumer
        """
        tasks = []
        for idx, proc in enumerate(self.async_producers):
            if idx == self.producer_idx % len(self.async_producers):
                tasks.append(proc.async_sync_data.remote(data, self.num_producers))
            else:
                tasks.append(proc.async_sync_data.remote({}, self.num_producers))
        ray.get(tasks)
        return
