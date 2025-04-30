import copy
import os
from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
import tqdm
from coati.dataset.loader import RawConversationDataset
from coati.distributed.reward.reward_fn import boxed_math_reward_fn, math_reward_fn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from colossalai.utils import get_current_device

from .comm import ray_broadcast_tensor_dict
from .inference_backend import BACKEND_MAP
from .utils import pre_send, safe_write_jsonl

try:
    from vllm import SamplingParams
except ImportError:
    LLM = None


class BaseProducer:
    def __init__(
        self,
        producer_idx: int,
        num_producers: int,
        num_consumer_procs: int,
        num_episodes: int,
        batch_size: int,
        train_dataset_config: Dict[str, Any],
        dataloaders_config: Dict[str, Any],
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer_config: Optional[Dict[str, Any]] = None,
        microbatch_size: int = 1,
        backend: str = "transformers",
        consumer_plugin_config: Dict[str, Any] = None,
        eval_dataset_config=None,
        eval_interval=-1,  # disable evaluation
        evaluation_function_type="think_answer_tags",
        eval_save_dir: str = "./eval",
    ):
        self.producer_idx = producer_idx
        self.num_producers = num_producers
        self.num_consumer_procs = num_consumer_procs
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        assert batch_size % microbatch_size == 0
        self.num_microbatches = batch_size // microbatch_size

        self.train_dataset_config = train_dataset_config
        self.model_config = model_config
        self.generate_config = generate_config
        self.tokenizer_config = tokenizer_config
        self.consumer_plugin_config = consumer_plugin_config
        self.eval_interval = eval_interval
        self.eval_save_dir = eval_save_dir
        self.consumer_global_step = 0

        if os.path.exists(self.eval_save_dir):
            raise ValueError(f"Eval save dir {self.eval_save_dir} already exists. Please delete it or change the name.")

        # init tokenizer
        if tokenizer_config is None:
            tokenizer_path = model_config["path"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer_path = tokenizer_config.pop("path")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        self.tokenizer.padding_side = "left"

        # init dataloader
        train_dataset_path = train_dataset_config.pop("path")
        self.train_dataset = RawConversationDataset(self.tokenizer, train_dataset_path, **train_dataset_config)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=microbatch_size,
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=num_producers,
                rank=producer_idx,
                shuffle=True,
                drop_last=True,
                seed=42,
            ),
            num_workers=4,
            drop_last=True,
        )

        self.eval_dataset_config = eval_dataset_config
        if self.eval_dataset_config is not None:
            self.eval_dataloaders = {}
            for eval_task_name in self.eval_dataset_config:
                eval_dataset_path = eval_dataset_config[eval_task_name].pop("path")
                eval_dataset = RawConversationDataset(
                    self.tokenizer, eval_dataset_path, **eval_dataset_config[eval_task_name]
                )
                print(f"[P{self.producer_idx}] eval dataset {eval_task_name} size: {len(eval_dataset)}")
                self.eval_dataloaders[eval_task_name] = DataLoader(
                    eval_dataset,
                    batch_size=microbatch_size,
                    sampler=DistributedSampler(
                        eval_dataset,
                        num_replicas=num_producers,
                        rank=producer_idx,
                        shuffle=False,
                        drop_last=False,
                        seed=42,
                    ),
                )
            if evaluation_function_type == "think_answer_tags":
                self.evaluation_function = math_reward_fn
            elif evaluation_function_type == "boxed":
                self.evaluation_function = boxed_math_reward_fn
            else:
                raise ValueError(f"Unknown evaluation function type {evaluation_function_type}")
        else:
            raise ValueError("eval_dataset_config is not defined")
        self.device = get_current_device()

        # init backend
        if backend in BACKEND_MAP:
            self.backend_cls = BACKEND_MAP[backend]
        else:
            raise ValueError(f"Unexpected backend {backend}")

        self.consumer_pp_size = consumer_plugin_config.get("pp_size", 1)  # consumer pp size

    def setup(self) -> None:
        cc.init_collective_group(1 + self.num_consumer_procs, 0, group_name=f"sync_data_{self.producer_idx}")
        if self.consumer_pp_size > 1:
            for i in range(self.consumer_pp_size):
                cc.init_collective_group(self.num_producers + 1, self.producer_idx, group_name=f"sync_model_{i}")
        else:
            cc.init_collective_group(self.num_producers + 1, self.producer_idx, group_name="sync_model")
        cc.init_collective_group(1 + self.num_consumer_procs, 0, group_name=f"sync_eval_statistics_{self.producer_idx}")

    def rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def loop(self) -> None:
        num_update_per_episode = len(self.train_dataloader) // self.num_microbatches
        num_valid_microbatches = num_update_per_episode * self.num_microbatches

        print(
            f"[P{self.producer_idx}] num_valid_microbatches {num_valid_microbatches}, nmb: {self.num_microbatches}, dl: {len(self.train_dataloader)}"
        )
        for episode in range(self.num_episodes):
            self.train_dataloader.sampler.set_epoch(episode)
            for i, batch in enumerate(self.train_dataloader):
                if i >= num_valid_microbatches:
                    break
                if self.eval_interval > 0 and self.eval_dataset_config is not None:
                    if i % self.eval_interval == 0:
                        eval_statistics = {}
                        for eval_task_name in self.eval_dataloaders:
                            print(
                                f"[P{self.producer_idx}] Evaluate episode {episode} step {i} on task {eval_task_name}"
                            )
                            eval_results = []
                            eval_statistics[eval_task_name] = torch.zeros(2, device=self.device)
                            for eval_batch in tqdm.tqdm(
                                self.eval_dataloaders[eval_task_name], disable=self.producer_idx != 0
                            ):
                                eval_outputs = self.rollout(**eval_batch, sample_params=self.eval_sample_params)
                                eval_results = eval_results + [
                                    self.evaluation_function(
                                        eval_outputs["input_ids"][m][n],
                                        eval_outputs["gt_answer"][m][n],
                                        eval_outputs["response_idx"][m][n],
                                        tokenizer=self.tokenizer,
                                        eval_mode=True,
                                    )
                                    for m in range(eval_outputs["input_ids"].size(0))
                                    for n in range(eval_outputs["input_ids"].size(1))
                                ]
                            eval_statistics[eval_task_name][0] += len(
                                [res for res in eval_results if res["ans_valid"] == 1]
                            )
                            eval_statistics[eval_task_name][1] += len(eval_results)
                            # save eval results
                            result_file_name = os.path.join(
                                self.eval_save_dir,
                                f"{eval_task_name}_episode_{episode}_step_{self.consumer_global_step}.jsonl",
                            )
                            # delete the file if it exists
                            safe_write_jsonl(result_file_name, eval_results)
                        print(f"[P{self.producer_idx}] Send eval statistics episode {episode} step {i}")
                        ray_broadcast_tensor_dict(
                            eval_statistics,
                            src=0,
                            device=self.device,
                            group_name=f"sync_eval_statistics_{self.producer_idx}",
                        )
                outputs = self.rollout(**batch)

                print(f"[P{self.producer_idx}] Send data {[(k, v.shape) for k, v in outputs.items()]}")
                outputs["temperature"] = torch.tensor(
                    [self.model.generate_config["temperature"]] * outputs["input_ids"].size(0)
                ).to(outputs["input_ids"].device)
                outputs = pre_send(outputs)
                ray_broadcast_tensor_dict(
                    outputs, src=0, device=self.device, group_name=f"sync_data_{self.producer_idx}"
                )
                if (i + 1) % self.num_microbatches == 0 and (
                    episode != self.num_episodes - 1 or i != num_valid_microbatches - 1
                ):
                    if isinstance(self.model, BACKEND_MAP["vllm"]) and self.model.model_config.get(
                        "enable_sleep_mode", False
                    ):
                        self.model.llm.sleep()  # revict KV_cache to avoid OOM
                    # don't sync model for last iteration
                    torch.cuda.empty_cache()

                    if self.consumer_pp_size > 1:
                        for pp_idx in range(self.consumer_pp_size):
                            print(
                                f"[P{self.producer_idx}] Sync model PP stage {pp_idx} episode {episode} step {(i + 1) // self.num_microbatches - 1}"
                            )
                            state_dict = ray_broadcast_tensor_dict(
                                None, self.num_producers, device=self.device, group_name=f"sync_model_{pp_idx}"
                            )
                            self.load_state_dict(state_dict)
                    else:
                        print(
                            f"[P{self.producer_idx}] Sync model episode {episode} step {(i + 1) // self.num_microbatches - 1}"
                        )
                        state_dict = ray_broadcast_tensor_dict(
                            None, self.num_producers, device=self.device, group_name="sync_model"
                        )
                        if "consumer_global_step" in state_dict:
                            self.consumer_global_step = state_dict.pop("consumer_global_step").item()
                        self.load_state_dict(state_dict)
                    del state_dict
                    torch.cuda.empty_cache()
                    if isinstance(self.model, BACKEND_MAP["vllm"]) and self.model.model_config.get(
                        "enable_sleep_mode", False
                    ):
                        self.model.llm.wake_up()
                # linear annealing for 1 episode, temperature from initial to 0.9
                if episode <= 0:
                    ratio = 1 - (len(self.train_dataloader) - i) / len(self.train_dataloader)
                    if isinstance(self.model.generate_config.temperature, dict):
                        self.model.generate_config["temperature"] = (1 - ratio) * self.generate_config[
                            "temperature"
                        ] + ratio * 0.9
                    else:
                        self.model.generate_config.temperature = (1 - ratio) * self.generate_config[
                            "temperature"
                        ] + ratio * 0.9


@ray.remote
class SimpleProducer(BaseProducer):
    def __init__(
        self,
        producer_idx,
        num_producers,
        num_consumer_procs,
        num_episodes,
        batch_size,
        train_dataset_config,
        dataloaders_config,
        model_config,
        generate_config,
        tokenizer_config=None,
        microbatch_size=1,
        backend="transformers",
        num_generations: int = 8,
        consumer_plugin_config=None,
        eval_dataset_config=None,
        eval_interval=-1,  # disable evaluation
        evaluation_function_type="think_answer_tags",
        eval_save_dir: str = "./eval",
    ):
        super().__init__(
            producer_idx,
            num_producers,
            num_consumer_procs,
            num_episodes,
            batch_size,
            train_dataset_config,
            dataloaders_config,
            model_config,
            generate_config,
            tokenizer_config,
            microbatch_size,
            backend,
            consumer_plugin_config,
            eval_dataset_config=eval_dataset_config,
            eval_interval=eval_interval,
            evaluation_function_type=evaluation_function_type,
            eval_save_dir=eval_save_dir,
        )
        self.model = self.backend_cls(model_config, generate_config, self.tokenizer, num_generations)
        self.eval_generation_config = copy.deepcopy(self.model.generate_config)
        self.eval_generation_config["n"] = 1  # use 1 generation for evaluation
        self.eval_sample_params = SamplingParams(**self.eval_generation_config)

    @torch.no_grad()
    def rollout(self, input_ids, attention_mask, **kwargs):
        rollouts = self.model.generate(input_ids, attention_mask, **kwargs)
        # if self.producer_idx == 1:
        #     print("Rollout example:\n", self.tokenizer.decode(rollouts["input_ids"][0][0], skip_special_tokens=True))

        return rollouts

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
