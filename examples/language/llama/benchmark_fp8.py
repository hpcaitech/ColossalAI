# CUDA_VISIBLE_DEVICES=6 python bench.py -c 100m -b 1 -s 5 -i 2 -x -g --fp8
import argparse
import time
from typing import Optional

import torch
import transformer_engine.pytorch as te
from data_utils import RandomDataset
from model_utils import format_numel_str, get_model_numel
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from colossalai.accelerator import get_accelerator
from colossalai.shardformer.modeling.llama import TELlamaModelForCausalLM


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


def get_profile_context(enable_flag, warmup_steps, active_steps, save_dir):
    class DummyProfiler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    if enable_flag:
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
    else:
        return DummyProfiler()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time.time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time.time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0


class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        model_numel: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        enable_grad_checkpoint: bool = False,
        ignore_steps: int = 0,
    ) -> None:
        self.model_numel = model_numel
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_steps = ignore_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0
        self.flop_megatron = 0
        self.flop: int = 0

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return
        get_accelerator().synchronize()
        self.timer.start()

    def on_step_end(self, input_ids: Tensor, **kwargs) -> None:
        if self.disable:
            return
        get_accelerator().synchronize()
        self.timer.end()

        batch_size, seq_len = input_ids.shape

        self.num_samples += batch_size
        checkpoint_activations_factor = 3 + int(self.enable_grad_checkpoint)
        self.flop_megatron += (
            24 * checkpoint_activations_factor * batch_size * seq_len * self.num_layers * (self.hidden_size**2)
        ) * (
            1.0 + (seq_len / (6.0 * self.hidden_size)) + (self.vocab_size / (16.0 * self.num_layers * self.hidden_size))
        )
        self.flop += batch_size * seq_len * self.model_numel * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_duration = self.timer.duration
        avg_throughput = self.num_samples / (avg_duration + 1e-12)
        avg_tflops_per_gpu_megatron = self.flop_megatron / 1e12 / (avg_duration + 1e-12)
        avg_tflops_per_gpu = self.flop / 1e12 / (avg_duration + 1e-12)
        print(f"num_samples: {self.num_samples}, flop_megatron: {self.flop_megatron}, flop: {self.flop}")
        print(
            f"Throughput: {avg_throughput:.2f} samples/sec, TFLOPS per GPU by Megatron: {avg_tflops_per_gpu_megatron:.2f}, TFLOPS per GPU: {avg_tflops_per_gpu:.2f}"
        )


MODEL_CONFIGS = {
    "100m": LlamaConfig(
        max_position_embeddings=4096,
        num_hidden_layers=4,
        num_attention_heads=64 * 4,
        intermediate_size=2048,
        hidden_size=1024,
        use_cache=False,
        output_attentions=False,
    )
}


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="100m", help="Model configuration")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-s", "--num_steps", type=int, default=1, help="Number of steps to run")
    parser.add_argument("-i", "--ignore_steps", type=int, default=1, help="Number of steps to ignore")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "-w", "--warmup_ratio", type=float, default=0.8, help="warm up ratio of non-model data. Only for gemini-auto"
    )
    parser.add_argument("-m", "--memory_limit", type=int, help="Gemini memory limit in mb")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use xformers")
    parser.add_argument("--custom-ckpt", action="store_true", help="Customize checkpoint", default=False)
    parser.add_argument("--profile", action="store_true", help="Profile the code", default=False)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--fp8", action="store_true", default=False, help="Enable FP8")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.config]
    torch.cuda.manual_seed(42)

    dataset = RandomDataset(
        num_samples=args.batch_size * args.num_steps, max_length=args.max_length, vocab_size=config.vocab_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    model = LlamaForCausalLM(config).cuda().to(torch.bfloat16)
    if args.fp8:
        model = TELlamaModelForCausalLM.from_hf_model(model)
    optimizer = torch.optim.Adam(model.parameters())

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    model_numel = get_model_numel(model)
    print(f"Model params: {format_numel_str(model_numel)}")
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.config.vocab_size,
        args.grad_checkpoint,
        args.ignore_steps,
    )

    model.train()
    with get_profile_context(
        args.profile,
        args.ignore_steps,
        len(dataloader) - 1,
        save_dir=f"profile/{time.strftime('%H:%M', time.localtime())}-llama",
    ) as prof:
        for step, batch in enumerate(tqdm(dataloader, desc="Step")):
            performance_evaluator.on_step_start(step)
            with te.fp8_autocast(enabled=args.fp8):
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            performance_evaluator.on_step_end(batch["input_ids"])
            prof.step()
    performance_evaluator.on_fit_end()
    print(f"Max CUDA memory usage: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
