import copy
import os
import uuid
from typing import Any, Dict, Optional

import ray

from .consumer import SimpleConsumer
from .grpo_consumer import GRPOConsumer
from .producer import SimpleProducer

ALGO_MAP = {
    "Simple": SimpleConsumer,
    "GRPO": GRPOConsumer,
    "DAPO": GRPOConsumer,
    "REINFORCE_PPB": GRPOConsumer,
    "RLOO": GRPOConsumer,
}


def get_jsonl_size_fast(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()
        lines = [line for line in lines if line.strip()]
        return len(lines)


def get_dp_size_fast(n_procs: int, plugin_config: Dict[str, Any]) -> int:
    tp_size = plugin_config.get("tp_size", 1)
    pp_size = plugin_config.get("pp_size", 1)
    ep_size = plugin_config.get("ep_size", 1)
    sp_size = plugin_config.get("sp_size", 1)
    return n_procs // (tp_size * pp_size * ep_size * sp_size)


def launch_distributed(
    num_producers: int,
    num_proc_per_producer: int,
    num_consumer_procs: int,
    num_episodes: int,
    inference_batch_size: int,
    inference_microbatch_size: int,
    train_batch_size: int,
    train_minibatch_size: int,
    train_dataset_config: Dict[str, Any],
    inference_model_config: Dict[str, Any],
    generate_config: Dict[str, Any],
    train_model_config: Dict[str, Any],
    grpo_config: Dict[str, Any],
    plugin_config: Dict[str, Any],
    tokenizer_config: Optional[Dict[str, Any]] = None,
    inference_backend: str = "transformers",
    num_generations: int = 8,
    master_addr: str = "localhost",
    master_port: int = 29500,
    core_algo: str = "GRPO",
    project_name: Optional[str] = None,
    save_interval: int = 100,
    save_dir: str = "./model",
    eval_dataset_config: Optional[Dict[str, Any]] = None,
    eval_interval: int = 100,
    eval_save_dir: Optional[str] = None,
    eval_generation_config: Optional[Dict[str, Any]] = None,
    log_rollout_interval: int = 20,
    rollout_save_dir: str = "./rollout",
    enable_profiling: bool = False,
    n_behind: int = 0,
):
    if core_algo not in ALGO_MAP:
        raise NotImplementedError(f"{core_algo} is not supported yet.")
    else:
        core_consumer = ALGO_MAP.get(core_algo, SimpleConsumer)

    train_dp_size = get_dp_size_fast(num_consumer_procs, plugin_config)

    assert (inference_batch_size * num_producers) % (train_batch_size * train_dp_size) == 0

    dataset_path = train_dataset_config["path"]
    num_samples = get_jsonl_size_fast(dataset_path)
    global_inference_batch_size = inference_batch_size * num_producers
    num_update_per_episode = num_samples // global_inference_batch_size
    num_recv_per_update = inference_batch_size // inference_microbatch_size

    run_name = f"{inference_backend}_bs_{train_batch_size * train_dp_size}_temp_{generate_config['temperature']:.01f}_top_p_{generate_config['top_p']:.02f}"
    wandb_group_name = str(uuid.uuid4())
    rollout_log_file = os.path.join(
        rollout_save_dir,
        f"{project_name.replace(' ','_')}_run_{wandb_group_name}.jsonl",
    )

    # Attention: Ray use complex schedualing method that consider various factors including load-balancing.
    # when requesting resources, it is not guaranteed that the resource comes from a node with lower node it
    # this go against the design principle of our implementation, and we need to manually force the schedualing,
    # allocating the producer to nodes with lower node id and the consumer to the resouces from nodes with higher
    # node id. See the reference here: https://docs.ray.io/en/latest/ray-core/scheduling/index.html#nodeaffinityschedulingstrategy
    nodes = ray.nodes()
    node_info = {
        node["NodeID"]: {
            "num_gpus": node["Resources"].get("GPU", 0),
            "address": node["NodeManagerAddress"],
        }  # Default to 0 if no GPUs are available
        for node in nodes
    }
    gpu_to_node_id = []
    gpu_to_ip_address = []
    for node_id in node_info:
        for idx in range(int(node_info[node_id]["num_gpus"])):
            gpu_to_node_id.append(node_id)
            gpu_to_ip_address.append(node_info[node_id]["address"])
    print(node_info)

    producer_procs = []
    for i in range(num_producers):
        node_id = gpu_to_node_id[0]
        producer_ip_address = gpu_to_ip_address[0]
        for _ in range(num_proc_per_producer):
            gpu_to_node_id.pop(0)
            gpu_to_ip_address.pop(0)
        print(f"Schedual Producer P[{i}] which requires {num_proc_per_producer} GPUs on node {producer_ip_address}")
        producer = SimpleProducer.options(num_gpus=num_proc_per_producer).remote(
            producer_idx=i,
            num_producers=num_producers,
            num_consumer_procs=num_consumer_procs,
            num_episodes=num_episodes,
            batch_size=inference_batch_size,
            train_dataset_config=train_dataset_config,
            model_config=inference_model_config,
            generate_config=generate_config,
            tokenizer_config=tokenizer_config,
            microbatch_size=inference_microbatch_size,
            backend=inference_backend,
            num_generations=num_generations,
            consumer_plugin_config=plugin_config,
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
        producer_procs.append(producer)
    ray.get([p.setup.remote() for p in producer_procs])
    generate_config_consumer = copy.deepcopy(generate_config)
    generate_config_consumer.update(
        dict(
            backend=inference_backend,
        )
    )
    consumer_master_ip_address = gpu_to_ip_address[0]
    print(f"Use {consumer_master_ip_address} as master address for torch DDP.")
    consumer_procs = []
    for i in range(num_consumer_procs):
        node_id = gpu_to_node_id[0]
        consumer_ip_address = gpu_to_ip_address[0]
        gpu_to_node_id.pop(0)
        gpu_to_ip_address.pop(0)
        print(f"Schedual Consumer T[{i}] which requires 1 GPUs on node {consumer_ip_address}")
        consumer = core_consumer.options(num_gpus=1).remote(
            num_producers=num_producers,
            num_episodes=num_episodes,
            rank=i,
            world_size=num_consumer_procs,
            master_addr=consumer_master_ip_address,
            master_port=master_port,
            num_update_per_episode=num_update_per_episode,
            num_recv_per_update=num_recv_per_update,
            batch_size=train_batch_size,
            model_config=train_model_config,
            plugin_config=plugin_config,
            minibatch_size=train_minibatch_size,
            generate_config=generate_config_consumer,
            grpo_config=grpo_config,
            num_generations=num_generations,
            save_interval=save_interval,
            save_dir=save_dir,
            project_name=project_name,
            run_name=run_name,
            wandb_group_name=wandb_group_name,
            enable_profiling=enable_profiling,
            n_behind=n_behind,
        )
        consumer_procs.append(consumer)
    ray.get([p.setup.remote() for p in consumer_procs])
    ray.get([p.loop.remote() for p in (producer_procs + consumer_procs)])
