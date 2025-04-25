import copy
from typing import Any, Dict, Optional

import ray

from .consumer import SimpleConsumer
from .grpo_consumer import GRPOConsumer
from .producer import SimpleProducer

ALGO_MAP = {"Simple": SimpleConsumer, "GRPO": GRPOConsumer, "DAPO": GRPOConsumer}


def get_jsonl_size_fast(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()
        lines = [line for line in lines if line.strip()]
        return len(lines) - 1


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
    train_microbatch_size: int,
    train_minibatch_size: int,
    dataset_config: Dict[str, Any],
    dataloaders_config: Dict[str, Any],
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
):

    if core_algo not in ALGO_MAP:
        raise NotImplementedError(f"{core_algo} is not supported yet.")
    else:
        core_consumer = ALGO_MAP.get(core_algo, SimpleConsumer)

    train_dp_size = get_dp_size_fast(num_producers, plugin_config)
    assert (inference_batch_size * num_producers) % (train_batch_size * train_dp_size) == 0

    dataset_path = dataset_config["path"]
    num_samples = get_jsonl_size_fast(dataset_path)
    global_inference_batch_size = inference_batch_size * num_producers
    num_update_per_episode = num_samples // global_inference_batch_size
    num_recv_per_update = inference_batch_size // inference_microbatch_size

    procs = []
    for i in range(num_producers):
        producer = SimpleProducer.options(num_gpus=num_proc_per_producer).remote(
            producer_idx=i,
            num_producers=num_producers,
            num_consumer_procs=num_consumer_procs,
            num_episodes=num_episodes,
            batch_size=inference_batch_size,
            dataset_config=dataset_config,
            dataloaders_config=dataloaders_config,
            model_config=inference_model_config,
            generate_config=generate_config,
            tokenizer_config=tokenizer_config,
            microbatch_size=inference_microbatch_size,
            backend=inference_backend,
            num_generations=num_generations,
        )
        procs.append(producer)
    generate_config_consumer = copy.deepcopy(generate_config)
    generate_config_consumer.update(
        dict(
            backend=inference_backend,
        )
    )
    for i in range(num_consumer_procs):
        consumer = core_consumer.options(num_gpus=1).remote(
            num_producers=num_producers,
            num_episodes=num_episodes,
            rank=i,
            world_size=num_consumer_procs,
            master_addr=master_addr,
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
            project_name=project_name,
            save_interval=save_interval,
            save_dir=save_dir,
        )
        procs.append(consumer)
    ray.get([p.setup.remote() for p in procs])
    ray.get([p.loop.remote() for p in procs])
