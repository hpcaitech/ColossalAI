import argparse

import ray
import torch
from coati.distributed.launch_ppo import launch_distributed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("-rm", "--reward-model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("-d", "--dataset", type=str, default="data.jsonl")
    parser.add_argument("-t", "--num-trainers", type=int, default=2)
    parser.add_argument("-i", "--num-inferencer", type=int, default=2)
    parser.add_argument("-ibs", "--inference-batch-size", type=int, default=64)
    parser.add_argument("-imbs", "--inference-microbatch-size", type=int, default=4)
    parser.add_argument("-tbs", "--train-batch-size", type=int, default=64)
    parser.add_argument("-tmbs", "--train-microbatch-size", type=int, default=2)
    parser.add_argument("-b", "--backend", type=str, default="transformers")
    args = parser.parse_args()

    ray.init(address="local", namespace="ray-example")

    inference_model_config = dict(path=args.model)
    train_model_config = dict(
        path=args.model, rm_path=args.reward_model, input_len=300, kl_coef=0.09, gamma=1.0, lamda=0.95
    )
    generate_config = dict(top_k=10, top_p=0.8, temperature=0.9)

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
                max_length=1000,
                do_sample=True,
                max_new_tokens=None,
                early_stopping=False,
            )
        )
    elif args.backend == "vllm":
        inference_model_config.update(
            dict(
                gpu_memory_utilization=0.6,
            )
        )
        generate_config.update(
            dict(
                max_tokens=700,
                ignore_eos=True,
                # stop_token_ids = [6231]  # '_end
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
                max_new_tokens=700,
                ignore_eos=True,
            )
        )
    dataset_config = dict(
        path=args.dataset,
        max_length=300,
        # chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        system_message="You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <|im_start|>assistant\n<think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a math problem that involves reasoning. After thinking, when you finally reach a conclusion, clearly output the final answer without explanation within the <answer> </answer> tags, then end with the <im_end> tag. Your final answer should be a integer without unit, currency mark, thousands separator or other text. i.e., <answer> 123 </answer>\n",
        add_generation_prompt=True,
    )

    launch_distributed(
        num_producers=args.num_inferencer,
        num_proc_per_producer=1,
        num_consumer_procs=args.num_trainers,
        num_episodes=3,
        inference_batch_size=args.inference_batch_size,
        inference_microbatch_size=args.inference_microbatch_size,
        train_batch_size=args.train_batch_size,
        train_microbatch_size=args.train_microbatch_size,
        dataset_config=dataset_config,
        dataloaders_config={},
        inference_model_config=inference_model_config,
        generate_config=generate_config,
        train_model_config=train_model_config,
        plugin_config={},
        inference_backend=args.backend,
        master_addr="localhost",
        master_port=29505,
    )
