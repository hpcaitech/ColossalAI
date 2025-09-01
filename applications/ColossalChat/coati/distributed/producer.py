import copy
import json
import os
from typing import Any, Dict, Optional

import ray
import ray.util.collective as cc
import torch
import tqdm
import wandb
from coati.dataset import StatefulDistributedSampler
from coati.dataset.loader import RawConversationDataset, collate_fn_grpo
from coati.distributed.profiling_utils import CustomProfiler
from coati.distributed.reward.reward_fn import boxed_math_reward_fn, code_reward_fn, math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from coati.utils import load_checkpoint
from ray.util.collective import allreduce
from ray.util.collective.types import Backend, ReduceOp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from colossalai.utils import get_current_device

from .comm import ray_broadcast_tensor_dict
from .inference_backend import BACKEND_MAP
from .utils import pre_send, safe_append_to_jsonl_file

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
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer_config: Optional[Dict[str, Any]] = None,
        microbatch_size: int = 1,
        backend: str = "transformers",
        consumer_plugin_config: Dict[str, Any] = None,
        eval_dataset_config=None,
        eval_interval=-1,  # disable evaluation
        grpo_config: Dict[str, Any] = None,
        eval_save_dir: str = "./eval",
        project_name: str = None,
        run_name: str = None,
        wandb_group_name: str = None,
        log_rollout_interval: int = 20,
        rollout_log_file: str = "./rollout_log.jsonl",
        enable_profiling: bool = False,
        n_behind: int = 0,
    ):
        self.producer_idx = producer_idx
        self.num_producers = num_producers
        self.num_consumer_procs = num_consumer_procs
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        assert batch_size % microbatch_size == 0
        self.num_microbatches = batch_size // microbatch_size
        self.latest_eval_step = -1
        self.profiler = CustomProfiler(f"P{self.producer_idx}", disabled=not enable_profiling)

        self.train_dataset_config = train_dataset_config
        self.checkpoint_path = model_config.pop("checkpoint_path", None)
        self.model_config = model_config
        self.generate_config = generate_config
        self.tokenizer_config = tokenizer_config
        self.consumer_plugin_config = consumer_plugin_config
        self.eval_interval = eval_interval
        self.eval_save_dir = eval_save_dir
        self.consumer_global_step = 0
        self.eval_mode = False
        self.log_rollout_interval = log_rollout_interval
        self.latest_rollout_log_step = -1
        self.grpo_config = grpo_config
        self.n_behind = n_behind
        reward_model_kwargs = {
            k: v
            for k, v in grpo_config.items()
            if k in ["soft_over_length_punishment", "max_new_tokens", "cache_length", "code_verifier_api_url"]
        }
        self.response_format_tags = grpo_config.get("response_format_tags", None)
        if producer_idx == 0:
            if os.path.exists(rollout_log_file):
                raise ValueError(
                    f"Rollout log file {rollout_log_file} already exists. Please delete it or change the name."
                )
            else:
                os.makedirs(os.path.dirname(rollout_log_file), exist_ok=True)
                self.rollout_log_file = open(rollout_log_file, "w", encoding="utf8")
        if self.producer_idx == 0:
            self.wandb_run = wandb.init(
                project=project_name,
                sync_tensorboard=False,
                dir="./wandb",
                name=run_name + "_eval",
                group=wandb_group_name,
            )

        if os.path.exists(self.eval_save_dir) and self.eval_interval > 0:
            raise ValueError(f"Eval save dir {self.eval_save_dir} already exists. Please delete it or change the name.")

        # init tokenizer
        if tokenizer_config is None:
            tokenizer_path = model_config["path"]
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer_path = tokenizer_config.pop("path")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # init dataloader
        train_dataset_path = train_dataset_config.pop("path")
        self.train_dataset = RawConversationDataset(self.tokenizer, train_dataset_path, **train_dataset_config)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=microbatch_size,
            sampler=StatefulDistributedSampler(
                self.train_dataset,
                num_replicas=num_producers,
                rank=producer_idx,
                shuffle=True,
                drop_last=True,
                seed=42,
            ),
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn_grpo,
        )
        if self.checkpoint_path is not None:
            # resume training from checkpoint
            start_epoch, start_step, sampler_start_idx = load_checkpoint(self.checkpoint_path, None, None, None, None)
            self.train_dataloader.sampler.set_start_index(sampler_start_idx)
            print(
                f"[P{self.producer_idx}] Resume training from checkpoint {self.checkpoint_path}, start epoch {start_epoch}, start step {start_step}, sampler start index {sampler_start_idx}"
            )
        if grpo_config["reward_fn_type"] == "think_answer_tags":
            self.evaluation_function = math_reward_fn
        elif grpo_config["reward_fn_type"] == "boxed":
            self.evaluation_function = boxed_math_reward_fn
        elif grpo_config["reward_fn_type"] == "code":
            self.evaluation_function = code_reward_fn
        else:
            raise ValueError(f"Unknown evaluation function type {grpo_config['reward_fn_type']}")

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
                    collate_fn=collate_fn_grpo,
                )
        else:
            print("No eval dataset provided, skip eval")
        self.device = get_current_device()
        self.reward_model = VerifiableReward(
            reward_fns=[self.evaluation_function],  # multiple reward functions can be added here
            tokenizer=self.tokenizer,
            tags=self.response_format_tags,
            **reward_model_kwargs,
        )

        # init backend
        if backend in BACKEND_MAP:
            self.backend_cls = BACKEND_MAP[backend]
        else:
            raise ValueError(f"Unexpected backend {backend}")

        self.consumer_pp_size = consumer_plugin_config.get("pp_size", 1)  # consumer pp size

    def setup(self) -> None:
        cc.init_collective_group(
            world_size=self.num_producers,
            rank=self.producer_idx,
            backend=Backend.NCCL,
            group_name="producer_group",
        )
        cc.init_collective_group(1 + self.num_consumer_procs, 0, group_name=f"sync_data_{self.producer_idx}")
        if self.consumer_pp_size > 1:
            for i in range(self.consumer_pp_size):
                cc.init_collective_group(self.num_producers + 1, self.producer_idx, group_name=f"sync_model_{i}")
        else:
            cc.init_collective_group(self.num_producers + 1, self.producer_idx, group_name="sync_model")

    def rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def loop(self) -> None:

        torch.cuda.empty_cache()
        self.profiler.enter("sync_model")
        if self.consumer_pp_size > 1:
            for pp_idx in range(self.consumer_pp_size):
                state_dict = ray_broadcast_tensor_dict(
                    None, self.num_producers, device=self.device, group_name=f"sync_model_{pp_idx}"
                )
                if "consumer_global_step" in state_dict:
                    self.consumer_global_step = state_dict.pop("consumer_global_step").item()
                self.load_state_dict(state_dict)
        else:
            state_dict = ray_broadcast_tensor_dict(
                None, self.num_producers, device=self.device, group_name="sync_model"
            )
            if "consumer_global_step" in state_dict:
                self.consumer_global_step = state_dict.pop("consumer_global_step").item()
            self.load_state_dict(state_dict)
        self.profiler.exit("sync_model")
        print(f"[P{self.producer_idx}] Sync initial model done.")
        del state_dict
        torch.cuda.empty_cache()

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
                    if (
                        self.consumer_global_step - self.latest_eval_step >= self.eval_interval
                        and self.consumer_global_step > self.latest_eval_step
                    ) or self.latest_eval_step == -1:
                        to_log_msg = {}
                        self.eval_mode = True
                        for eval_task_name in self.eval_dataloaders:
                            if self.producer_idx == 0:
                                print(
                                    f"[P{self.producer_idx}] Evaluate model at training step {self.consumer_global_step} on task {eval_task_name}"
                                )
                            eval_results = []
                            eval_statistics_tensor = torch.zeros((2,), dtype=torch.float32).to(self.device)
                            for eval_batch in tqdm.tqdm(
                                self.eval_dataloaders[eval_task_name], disable=self.producer_idx != 0
                            ):
                                eval_outputs = self.rollout(**eval_batch, sample_params=self.eval_sample_params)
                                eval_results = eval_results + [
                                    self.evaluation_function(
                                        eval_outputs["input_ids"][m][n],
                                        eval_outputs[
                                            (
                                                "test_cases"
                                                if self.grpo_config["reward_fn_type"] == "code"
                                                else "gt_answer"
                                            )
                                        ][m],
                                        eval_outputs["response_idx"][m][n],
                                        tokenizer=self.tokenizer,
                                        eval_mode=True,
                                        tags=self.response_format_tags,
                                    )
                                    for m in range(eval_outputs["input_ids"].size(0))
                                    for n in range(eval_outputs["input_ids"].size(1))
                                ]
                            eval_statistics_tensor[0] += sum([max(0, res["ans_valid"]) for res in eval_results])
                            eval_statistics_tensor[1] += len(eval_results)
                            allreduce(eval_statistics_tensor, op=ReduceOp.SUM, group_name="producer_group")
                            to_log_msg[f"eval/{eval_task_name}"] = (
                                eval_statistics_tensor[0].item() / eval_statistics_tensor[1].item()
                            )
                            if self.producer_idx == 0:
                                print(
                                    f"[P{self.producer_idx}]: Accuracy on {eval_task_name}: {to_log_msg[f'eval/{eval_task_name}']}"
                                )
                            # save eval results
                            safe_append_to_jsonl_file(
                                os.path.join(
                                    self.eval_save_dir,
                                    f"{eval_task_name}_training_step_{self.consumer_global_step}.jsonl",
                                ),
                                eval_results,
                            )

                        if self.producer_idx == 0:
                            self.wandb_run.log(to_log_msg, step=self.consumer_global_step)
                        self.eval_mode = False
                        self.latest_eval_step = self.consumer_global_step
                self.profiler.enter("rollout")
                outputs = self.rollout(**batch)
                self.profiler.exit("rollout")
                outputs["temperature"] = torch.tensor(
                    [self.model.generate_config["temperature"]] * outputs["input_ids"].size(0)
                ).to(outputs["input_ids"].device)
                bs, num_gen = outputs["input_ids"].size(0), outputs["input_ids"].size(1)
                self.profiler.enter("calculate_reward")
                if self.grpo_config["reward_fn_type"] == "code":
                    test_cases = []
                    for prompt_id in range(bs):
                        test_cases.extend([outputs["test_cases"][prompt_id]] * num_gen)
                    reward_model_output = self.reward_model(
                        outputs["input_ids"].view((-1, outputs["input_ids"].size(-1))),
                        test_cases=test_cases,
                        response_idx=outputs["response_idx"].view((-1, 2)),
                    )
                else:
                    gt_answer = []
                    for prompt_id in range(bs):
                        gt_answer.extend([outputs["gt_answer"][prompt_id]] * num_gen)
                    reward_model_output = self.reward_model(
                        outputs["input_ids"].view((-1, outputs["input_ids"].size(-1))),
                        gt_answer=gt_answer,
                        response_idx=outputs["response_idx"].view((-1, 2)),
                    )
                outputs["reward"] = (
                    torch.tensor([value[0] for value in reward_model_output])
                    .to(outputs["input_ids"].device)
                    .view((bs, num_gen, 1))
                )
                outputs["format_acc"] = (
                    torch.tensor([value[1] for value in reward_model_output])
                    .to(outputs["input_ids"].device)
                    .view((bs, num_gen, 1))
                )
                outputs["ans_acc"] = (
                    torch.tensor([value[2] for value in reward_model_output])
                    .to(outputs["input_ids"].device)
                    .view((bs, num_gen, 1))
                )
                if "gt_answer" in outputs:
                    outputs.pop("gt_answer")
                if "test_cases" in outputs:
                    outputs.pop("test_cases")
                self.profiler.exit("calculate_reward")

                print(f"[P{self.producer_idx}] Send data {[(k, v.shape) for k, v in outputs.items()]}")
                outputs = pre_send(outputs)
                self.profiler.enter("send_broadcast_data")
                ray_broadcast_tensor_dict(
                    outputs, src=0, device=self.device, group_name=f"sync_data_{self.producer_idx}"
                )
                self.profiler.exit("send_broadcast_data")
                if (
                    (i + 1) % self.num_microbatches == 0
                    and (episode != self.num_episodes - 1 or i != num_valid_microbatches - 1)
                    and (episode != 0 or (i + 1) > self.n_behind * self.num_microbatches)
                ):
                    if isinstance(self.model, BACKEND_MAP["vllm"]) and self.model.model_config.get(
                        "enable_sleep_mode", False
                    ):
                        self.model.llm.sleep()  # revict KV_cache to avoid OOM
                    # don't sync model for last iteration
                    torch.cuda.empty_cache()
                    self.profiler.enter("sync_model")
                    if self.consumer_pp_size > 1:
                        for pp_idx in range(self.consumer_pp_size):
                            print(
                                f"[P{self.producer_idx}] Sync model PP stage {pp_idx} episode {episode} step {(i + 1) // self.num_microbatches - 1}"
                            )
                            state_dict = ray_broadcast_tensor_dict(
                                None, self.num_producers, device=self.device, group_name=f"sync_model_{pp_idx}"
                            )
                            if "consumer_global_step" in state_dict:
                                self.consumer_global_step = state_dict.pop("consumer_global_step").item()
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
                    self.profiler.exit("sync_model")
                    del state_dict
                    torch.cuda.empty_cache()
                    if isinstance(self.model, BACKEND_MAP["vllm"]) and self.model.model_config.get(
                        "enable_sleep_mode", False
                    ):
                        self.model.llm.wake_up()
                # linear annealing for 1 episode, temperature from initial to 0.9
                if episode <= 0:
                    ratio = 1 - (len(self.train_dataloader) - i) / len(self.train_dataloader)
                    self.model.generate_config["temperature"] = (1 - ratio) * self.generate_config[
                        "temperature"
                    ] + ratio * 0.9
                    if isinstance(self.model, BACKEND_MAP["vllm"]):
                        self.model.sample_params.temperature = (1 - ratio) * self.generate_config[
                            "temperature"
                        ] + ratio * 0.9

    def __del__(self):
        self.profiler.close()


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
        model_config,
        generate_config,
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
        )
        self.model = self.backend_cls(model_config, generate_config, self.tokenizer, num_generations)
        self.eval_generation_config = copy.deepcopy(self.model.generate_config)
        self.eval_generation_config["n"] = 1  # use 1 generation for evaluation
        self.eval_generation_config.update(eval_generation_config)
        self.eval_sample_params = SamplingParams(**self.eval_generation_config)

    @torch.no_grad()
    def rollout(self, input_ids, attention_mask, **kwargs):
        rollouts = self.model.generate(input_ids, attention_mask, **kwargs)
        if self.producer_idx == 0 and not self.eval_mode:
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
        return rollouts

    def __del__(self):
        if self.producer_idx == 0:
            self.wandb_run.finish()
        if hasattr(self, "rollout_log_file"):
            self.rollout_log_file.close()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
