import argparse
import os

import ray
import torch
from coati.distributed.launch import launch_distributed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("-d", "--dataset", type=str, default="data.jsonl")
    parser.add_argument("-p", "--project", type=str, default="GRPO", help="Project name.")

    # Distributed training parameters
    parser.add_argument("-t", "--num-trainers", type=int, default=2)
    parser.add_argument("-i", "--num-inferencer", type=int, default=2)
    parser.add_argument("-g", "--num-generations", type=int, default=8, help="Number of generations per prompt.")
    parser.add_argument(
        "-ibs",
        "--inference-batch-size",
        type=int,
        default=None,
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
        default=None,
        help="Number of unique prompts in each training batch per dp group. The inference backend must generate tMbs * g * dp_size samples before forwarding. Satisfy tMbs * g >= tmbs",
    )
    parser.add_argument(
        "-tmbs",
        "--train-microbatch-size",
        type=int,
        default=2,
        help="Effective batch size per dp group for forwarding and backwarding. Please select based on the availiable memory.",
    )
    parser.add_argument(
        "--ray_dir", type=str, default=None, help="Custom temperary directory for storing ray cluster data, Optional"
    )
    parser.add_argument(
        "--master_address", type=str, default=None, help="Master address for multi-node distributed training, Optional"
    )
    parser.add_argument(
        "--master_port", type=int, default=29506, help="Master port for multi-node distributed training, Optional"
    )

    # Sampling parameters
    parser.add_argument("-b", "--backend", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("-temp", "--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument(
        "-topk",
        "--top-k",
        type=int,
        default=None,
        help="Top k for sampling. Please check the generation arguments documentation for your backend.",
    )
    parser.add_argument(
        "-topp",
        "--top-p",
        type=float,
        default=1.0,
        help="Top p for sampling. Please check the generation arguments documentation for your backend.",
    )
    parser.add_argument("-s", "--system-prompt", type=str, default=None, help="System prompt for data construction.")
    parser.add_argument("-mnt", "--max-new-tokens", type=int, default=1024 * 4 - 512, help="Max length for generation.")
    parser.add_argument("-mpt", "--max-prompt-tokens", type=int, default=512, help="Max length for prompt.")

    # GRPO parameters
    parser.add_argument("-a", "--algo", type=str, default="GRPO", choices=["DAPO", "GRPO"])
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-6, help="Learning rate for GRPO.")
    parser.add_argument("-kl", "--kl-coeff", type=float, default=0.01, help="KL penalty coefficient for GRPO.")

    # Logging/Checkpointing parameters
    parser.add_argument("-si", "--save-interval", type=int, default=100, help="Interval for saving checkpoints.")
    parser.add_argument("-sd", "--save-dir", type=str, default="./model", help="Directory for saving checkpoints.")

    args = parser.parse_args()

    if args.train_minibatch_size is None:
        # Default settings: Using train batch size as mini batch size
        args.train_minibatch_size = args.train_batch_size
    if args.inference_batch_size is None:
        # Default settings: Using train batch size as inference batch size, sync every inference model every train step
        args.inference_batch_size = args.train_batch_size
    assert (
        args.train_minibatch_size * args.num_generations >= args.train_microbatch_size
        and args.train_microbatch_size > 0
    ), "Train micro batch size must be greater than 0 less than train mini batch size * num generations"
    assert (
        args.train_minibatch_size <= args.train_batch_size
    ), "Train mini batch size must be less than or equals to train batch size"

    if args.master_address is None:
        # Default settings: Using single machine
        ray.init(address="local", namespace="ray-example")
    else:
        # For ray distributed multi-machine training, Please change _node_ip_address to your IP address of your master node
        ray.init(_node_ip_address=args.master_address, namespace="ray-example", _temp_dir=args.ray_dir)

    if args.top_k is None:
        if args.backend == "transformers":
            args.top_k = 50
        elif args.backend == "vllm":
            args.top_k = -1

    inference_model_config = dict(path=args.model)
    train_model_config = dict(path=args.model, use_flash_attention_2=True, use_cache=False)
    generate_config = dict(top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

    if args.backend == "transformers":
        inference_model_config.update(
            dict(
                use_flash_attention_2=True,
                torch_dtype=torch.bfloat16,
            )
        )
        generate_config.update(
            dict(
                max_length=args.max_new_tokens + args.max_prompt_tokens,
                do_sample=True,
                max_new_tokens=None,
                early_stopping=False,
                stop_strings=["</answer>"],
            )
        )
    elif args.backend == "vllm":
        inference_model_config.update(
            dict(
                gpu_memory_utilization=0.7,
                enforce_eager=True,
                enable_chunked_prefill=True,
                max_model_len=args.max_new_tokens + args.max_prompt_tokens,
                tensor_parallel_size=1,
            )
        )
        generate_config.update(
            dict(
                max_tokens=args.max_new_tokens,  # max new tokens
                ignore_eos=True,
                include_stop_str_in_output=True,
                stop=["</answer>"],
            )
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if args.algo == "GRPO":
        # Default Settings
        grpo_config = {
            "lr": args.learning_rate,
            "train_microbatch_size": args.train_microbatch_size,
            "beta": args.kl_coeff,  # KL penalty coefficient
            "loss_variation": "sample_level",
        }
    elif args.algo == "DAPO":
        # DAPO variant settings
        grpo_config = {
            "filter_range": [0.01, 0.99],  # only filter out all zero batch and all one batch
            "lr": args.learning_rate,
            "train_microbatch_size": args.train_microbatch_size,
            "dynamic_batching": True,
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.28,
            "skip_threshold": 20.0,
            "beta": 0,  # no KL penalty for DAPO
            "loss_variation": "token_level",
            "soft_over_length_punishment": True,
            "max_length": args.max_new_tokens + args.max_prompt_tokens,
            "cache_length": min(1024, int(args.max_new_tokens / 4)),
            "filter_truncated_response": True,
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    launch_distributed(
        num_producers=args.num_inferencer,
        num_proc_per_producer=inference_model_config.get("tensor_parallel_size", 1),
        num_consumer_procs=args.num_trainers,
        num_episodes=1,
        inference_batch_size=args.inference_batch_size,
        inference_microbatch_size=args.inference_microbatch_size,
        train_batch_size=args.train_batch_size,
        train_minibatch_size=args.train_minibatch_size,
        train_microbatch_size=args.train_microbatch_size,
        dataset_config={
            "path": args.dataset,
            "max_length": args.max_prompt_tokens,
            "system_prompt": args.system_prompt,
        },
        dataloaders_config={},
        inference_model_config=inference_model_config,
        generate_config=generate_config,
        num_generations=args.num_generations,
        train_model_config=train_model_config,
        grpo_config=grpo_config,
        plugin_config={
            "zero_stage": 2,
        },  # for zero
        # currently not support tp/pp
        # plugin_config={
        #     "tp_size": 2,
        #     "microbatch_size": args.train_microbatch_size // 2,
        #     "zero_stage": 0,
        #     "max_norm": 1.0,
        # },  # for pp
        inference_backend=args.backend,
        master_addr="localhost",
        master_port=args.master_port,
        core_algo=args.algo,
        project_name=args.project,
        save_interval=args.save_interval,
        save_dir=os.path.join(args.save_dir, args.project.replace(" ", "_")),
    )
