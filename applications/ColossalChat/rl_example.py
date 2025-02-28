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
    parser.add_argument("-ibs", "--inference-batch-size", type=int, default=64)
    parser.add_argument("-imbs", "--inference-microbatch-size", type=int, default=16)
    parser.add_argument("-tbs", "--train-batch-size", type=int, default=16)
    parser.add_argument("-tmbs", "--train-microbatch-size", type=int, default=2)
    parser.add_argument("-b", "--backend", type=str, default="transformers")
    args = parser.parse_args()

    ray.init(address="local", namespace="ray-example")

    inference_model_config = dict(path=args.model)
    train_model_config = dict(path=args.model)
    generate_config = dict(
        top_k=50,
        top_p=0.8,
    )

    if args.backend == "transformers":
        inference_model_config.update(
            dict(
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        )
        train_model_config.update(
            dict(
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                use_cache=False,
            )
        )
        generate_config.update(
            dict(
                max_length=512,
                do_sample=True,
                max_new_tokens=None,
                early_stopping=False,
            )
        )
    elif args.backend == "vllm":
        inference_model_config.update(
            dict(
                gpu_memory_utilization=0.7,
            )
        )
        generate_config.update(
            dict(
                max_tokens=2048,
                ignore_eos=True,
                include_stop_str_in_output=True,
                stop=["</answer>"],
                temperature=0.2,
                top_p=0.95,
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
        train_microbatch_size=args.train_microbatch_size,
        dataset_config={"path": args.dataset, "max_length": 256},
        dataloaders_config={},
        inference_model_config=inference_model_config,
        generate_config=generate_config,
        train_model_config=train_model_config,
        plugin_config={},
        inference_backend=args.backend,
        master_addr="localhost",
        master_port=29504,
    )
