import argparse

import ray
import torch
from coati.distributed.launch import launch_distributed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("-d", "--dataset", type=str, default="data.jsonl")
    parser.add_argument("-t", "--num-trainers", type=int, default=2)
    parser.add_argument("-i", "--num-inferencer", type=int, default=2)
    parser.add_argument("-g", "--num-generations", type=int, default=8, help="Number of generations per prompt.")
    parser.add_argument("-p", "--project", type=str, default="GRPO", help="Project name.")
    parser.add_argument(
        "-ibs",
        "--inference-batch-size",
        type=int,
        default=64,
        help="Number of prompts to generate per inference step. It should be divisible by tbs, and the weights on the inference backend will be synced every ibs/tbs training steps of the policy model.",
    )
    parser.add_argument(
        "-imbs",
        "--inference-microbatch-size",
        type=int,
        default=8,
        help="Effective batch size for the inference backend to run generation. Please select based on memory constraint.",
    )
    parser.add_argument(
        "-tbs",
        "--train-batch-size",
        type=int,
        default=32,
        help="Number of unique prompts to update policy model per step per dp group. Gradient is accumulated across tbs * dp_size unique prompts, equivalently tbs * g * dp_size samples",
    )
    parser.add_argument(
        "-tMbs",
        "--train-minibatch-size",
        type=int,
        default=1,
        help="Number of unique prompts in each training batch per dp group. The inference backend must generate tMbs * g * dp_size samples before forwarding. Satisfy tMbs * g >= tmbs",
    )
    parser.add_argument(
        "-tmbs",
        "--train-microbatch-size",
        type=int,
        default=2,
        help="Effective batch size per dp group for forwarding and backwarding. Please select based on the availiable memory.",
    )
    parser.add_argument("-b", "--backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("-a", "--algo", type=str, default="GRPO", choices=["Simple", "GRPO", "EvalGRPO"])
    parser.add_argument("-s", "--system-prompt", type=str, default=None, help="System prompt for data construction.")
    args = parser.parse_args()

    assert args.train_minibatch_size > 0, "Train mini batch size must be greater than 0"
    assert (
        args.train_minibatch_size * args.num_generations >= args.train_microbatch_size
        and args.train_microbatch_size > 0
    ), "Train micro batch size must be greater than 0 less than train mini batch size * num generations"

    ray.init(address="local", namespace="ray-example")

    inference_model_config = dict(path=args.model)
    train_model_config = dict(path=args.model, use_flash_attention_2=True, use_cache=False)
    generate_config = dict(top_k=50, top_p=0.75, temperature=0.9)

    if args.backend == "transformers":
        inference_model_config.update(
            dict(
                use_flash_attention_2=True,
                torch_dtype=torch.bfloat16,
            )
        )
        generate_config.update(
            dict(
                max_length=1024 + 512,
                do_sample=True,
                max_new_tokens=None,
                early_stopping=False,
                stop_strings=["</answer>"],
            )
        )
    elif args.backend == "vllm":
        inference_model_config.update(dict(gpu_memory_utilization=0.7, enforce_eager=True, enable_chunked_prefill=True))
        generate_config.update(
            dict(
                max_tokens=2048,
                ignore_eos=True,
                include_stop_str_in_output=True,
                stop=["</answer>"],
            )
        )
    else:
        inference_model_config.update(
            dict(
                mem_fraction_static=0.6,
            )
        )
        generate_config.update(
            dict(
                max_new_tokens=256,
                ignore_eos=True,
            )
        )

    launch_distributed(
        num_producers=args.num_inferencer,
        num_proc_per_producer=1,
        num_consumer_procs=args.num_trainers,
        num_episodes=1,
        inference_batch_size=args.inference_batch_size,
        inference_microbatch_size=args.inference_microbatch_size,
        train_batch_size=args.train_batch_size,
        train_minibatch_size=args.train_minibatch_size,
        train_microbatch_size=args.train_microbatch_size,
        dataset_config={"path": args.dataset, "max_length": 300, "system_prompt": args.system_prompt},
        dataloaders_config={},
        inference_model_config=inference_model_config,
        generate_config=generate_config,
        num_generations=args.num_generations,
        train_model_config=train_model_config,
        plugin_config={},  # Default setting: zero.
        # plugin_config={
        #     "pp_size": 2,
        #     "tp_size": 2,
        #     "microbatch_size": args.train_microbatch_size // 2,
        #     "zero_stage": 0,
        #     "max_norm": 1.0,
        # },  # for pp
        inference_backend=args.backend,
        master_addr="localhost",
        master_port=29506,
        core_algo=args.algo,
        project_name=args.project,
    )
