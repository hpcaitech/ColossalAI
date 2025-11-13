import argparse
import json
import os

import ray
import torch
from coati.distributed.launch_zero_bubble import launch_distributed

DEFAUT_SYSTEM_PROMPT = {
    "think_answer_tags": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a math problem that involves reasoning. After thinking, when you finally reach a conclusion, clearly output the final answer without explanation within the <answer> </answer> tags, i.e., <answer> 123 </answer>.\n\n",
    "boxed": "Please reason step by step, and put your final answer within \\boxed{}.",
    "code": "You are a helpful assistant.",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the tokenizer. If not provided, will use the model path.",
    )
    parser.add_argument("-d", "--dataset", type=str, default="data.jsonl")
    parser.add_argument(
        "-ed",
        "--eval-dataset",
        type=str,
        default=None,
        help="Evaluation dataset for each task, please use json format to specify the dataset for each task. \
        For example: {'task1':'data_eval_task1.jsonl', 'task2':'data_eval_task2.jsonl'}, the jsonl file should be in the same format as the training dataset. \
        The key is the task name, and the value is the path to the jsonl file",
    )
    parser.add_argument("-p", "--project", type=str, default="GRPO", help="Project name.")
    parser.add_argument("-e", "--num-episodes", type=int, default=1, help="Number of episodes to train.")

    # Distributed training parameters
    parser.add_argument("-t", "--num-trainers", type=int, default=2)
    parser.add_argument("-i", "--num-inferencer", type=int, default=2)
    parser.add_argument("-g", "--num-generations", type=int, default=8, help="Number of generations per prompt.")
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
        default=8,
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
        "-tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the trainer (consumer). Please check the generation arguments documentation for your backend.",
    )
    parser.add_argument(
        "-pp",
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size for the trainer (consumer). Please check the generation arguments documentation for your backend.",
    )
    parser.add_argument(
        "-zero",
        "--zero-stage",
        type=int,
        default=0,
        help="Zero stage for the trainer (consumer). Please check the generation arguments documentation for your backend.",
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
    parser.add_argument(
        "-ptp",
        "--producer-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the producer. Please check the generation arguments documentation for your backend.",
    )

    # GRPO parameters
    parser.add_argument("-a", "--algo", type=str, default="GRPO", choices=["DAPO", "GRPO"])
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-6, help="Learning rate for GRPO.")
    parser.add_argument("-kl", "--kl-coeff", type=float, default=0.01, help="KL penalty coefficient for GRPO.")
    parser.add_argument(
        "-rt",
        "--reward-type",
        type=str,
        default="think_answer_tags",
        choices=["think_answer_tags", "boxed", "code"],
        help="Reward type for GRPO.",
    )
    parser.add_argument(
        "-ei",
        "--eval-interval",
        type=int,
        default=100,
        help="Interval for evaluation. Evaluate every ei training steps.",
    )
    parser.add_argument(
        "-cbsl",
        "--data_actor_buffer_size_limit",
        type=int,
        default=-1,
        help="The approximate number of samples to keep in the consumer buffer. After this limit is reached, the producer will stop generating new samples and prioritize model sync until the consumer has processed some samples",
    )

    # Logging/Checkpointing parameters
    parser.add_argument("-si", "--save-interval", type=int, default=100, help="Interval for saving checkpoints.")
    parser.add_argument("-sd", "--save-dir", type=str, default="./model", help="Directory for saving checkpoints.")
    parser.add_argument(
        "-esd", "--eval-save-dir", type=str, default="./eval", help="Directory for saving evaluation results."
    )
    parser.add_argument(
        "-rsd", "--rollout-save-dir", type=str, default="./rollouts", help="Directory for saving rollout loggings."
    )
    parser.add_argument(
        "--enable_profiling", action="store_true", default=False, help="Enable profiling for the training process."
    )
    args = parser.parse_args()
    print(args)

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
        args.train_minibatch_size <= args.train_batch_size and args.train_batch_size % args.train_minibatch_size == 0
    ), "Train mini batch size must be less than or equals to train batch size and train batch size must be divisible by train mini batch size"

    if args.master_address is None:
        # Default settings: Using single machine
        ray.init(
            address="local",
            namespace="ray-example",
            runtime_env={
                "env_vars": {
                    # "RAY_DEBUG_POST_MORTEM": "1"  # enable post-mortem debugging with ray
                    "TOKENIZERS_PARALLELISM": "false"
                },
            },
        )
    else:
        # For ray distributed multi-machine training, Please change _node_ip_address to your IP address of your master node
        ray.init(
            _node_ip_address=args.master_address,
            namespace="ray-example",
            _temp_dir=args.ray_dir,
            runtime_env={
                "env_vars": {
                    # "RAY_DEBUG_POST_MORTEM": "1"  # enable post-mortem debugging with ray
                    "TOKENIZERS_PARALLELISM": "false"
                },
            },
        )

    if args.top_k is None:
        if args.backend == "transformers":
            args.top_k = 50
        elif args.backend == "vllm":
            args.top_k = -1

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid deadlock

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
                early_stopping=False if args.reward_type == "think_answer_tags" else True,
                stop_strings=["</answer>"] if args.reward_type == "think_answer_tags" else None,
            )
        )
        eval_generation_config = {"temperature": 0.6}  # used to update generation config for evaluation
    elif args.backend == "vllm":
        inference_model_config.update(
            dict(
                gpu_memory_utilization=0.7,
                enforce_eager=True,
                enable_chunked_prefill=True,
                max_model_len=args.max_new_tokens + args.max_prompt_tokens,
                tensor_parallel_size=args.producer_tensor_parallel_size,
            )
        )
        generate_config.update(
            dict(
                max_tokens=args.max_new_tokens,  # max new tokens
                ignore_eos=True if args.reward_type == "think_answer_tags" else False,
                include_stop_str_in_output=True,
                stop=["</answer>"] if args.reward_type == "think_answer_tags" else None,
            )
        )
        eval_generation_config = {"temperature": 0.6}  # used to update generation config for evaluation
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if args.algo == "GRPO":
        # Default Settings
        grpo_config = {
            "lr": args.learning_rate,
            "train_microbatch_size": args.train_microbatch_size,
            "num_minibatch_during_rollout": 1,  # number of mini batches to pop out from buffer and used for training during rollout of the producer after it syncs the model. Hint, set to a proper value close to the number of mini batches for training that takes roughly the same time as the rollout of the producer. A value that is too large or too small will cause bubble time on the trainer or the producer.
            "beta": args.kl_coeff,  # KL penalty coefficient
            "loss_variation": "sample_level",
            "reward_fn_type": args.reward_type,
            "max_length": args.max_new_tokens + args.max_prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "response_format_tags": (
                {
                    "think_start": {"text": "<think>", "num_occur": 1},
                    "think_end": {"text": "</think>", "num_occur": 1},
                    "answer_start": {"text": "<answer>", "num_occur": 1},
                    "answer_end": {"text": "</answer>", "num_occur": 1},
                }
                if args.reward_type == "think_answer_tags"
                else None
            ),
        }
    elif args.algo == "DAPO":
        # DAPO variant settings
        grpo_config = {
            "filter_range": [0.01, 0.7],  # only filter out all zero batch and all one batch
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
            "max_new_tokens": args.max_new_tokens,
            "cache_length": min(1024, int(args.max_new_tokens / 4)),
            "filter_truncated_response": True,
            "reward_fn_type": args.reward_type,
            "response_format_tags": (
                {
                    "think_start": {"text": "<think>", "num_occur": 1},
                    "think_end": {"text": "</think>", "num_occur": 1},
                    "answer_start": {"text": "<answer>", "num_occur": 1},
                    "answer_end": {"text": "</answer>", "num_occur": 1},
                }
                if args.reward_type == "think_answer_tags"
                else None
            ),
        }
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    if args.system_prompt is None:
        # Default system prompt
        args.system_prompt = DEFAUT_SYSTEM_PROMPT[args.reward_type]

    launch_distributed(
        num_producers=args.num_inferencer,
        num_proc_per_producer=inference_model_config.get("tensor_parallel_size", args.producer_tensor_parallel_size),
        num_consumer_procs=args.num_trainers,
        num_episodes=args.num_episodes,
        inference_batch_size=args.inference_batch_size,
        inference_microbatch_size=args.inference_microbatch_size,
        train_batch_size=args.train_batch_size,
        train_minibatch_size=args.train_minibatch_size,
        train_dataset_config={
            "path": args.dataset,
            "max_length": args.max_prompt_tokens,
            "system_prompt": args.system_prompt,
        },
        inference_model_config=inference_model_config,
        generate_config=generate_config,
        num_generations=args.num_generations,
        train_model_config=train_model_config,
        grpo_config=grpo_config,
        plugin_config={
            "tp_size": args.tensor_parallel_size,
            "pp_size": args.pipeline_parallel_size,
            "microbatch_size": max(
                1, args.train_microbatch_size // args.pipeline_parallel_size
            ),  # microbatch size should be set to train_microbatch_size // pp_size
            "zero_stage": args.zero_stage,
            "max_norm": 1.0,
            # "num_layers_per_stage": [18, 10],  # Example for 28 layers model with pp_size=2, set manually according to your model architecture
        },  # for pp, tp
        tokenizer_config={"path": args.tokenizer_path} if args.tokenizer_path else {"path": args.model},
        inference_backend=args.backend,
        master_addr="localhost",
        master_port=args.master_port,
        core_algo=args.algo,
        project_name=args.project,
        save_interval=args.save_interval,
        save_dir=os.path.join(args.save_dir, args.project.replace(" ", "_")),
        eval_dataset_config=(
            {
                k: {"path": v, "max_length": args.max_prompt_tokens, "system_prompt": args.system_prompt}
                for k, v in json.loads(args.eval_dataset).items()
            }
            if args.eval_dataset
            else None
        ),
        eval_interval=args.eval_interval,
        eval_save_dir=os.path.join(args.eval_save_dir, args.project.replace(" ", "_")),
        eval_generation_config=eval_generation_config,
        log_rollout_interval=20,
        rollout_save_dir=args.rollout_save_dir,
        enable_profiling=args.enable_profiling,
        data_actor_buffer_size_limit=args.data_actor_buffer_size_limit,
    )
