import copy
import os
import uuid
from typing import Any, Dict, Optional

import ray

from .comm import SharedVariableActor
from .zero_bubble.distributor import Distributor
from .zero_bubble.grpo_consumer import GRPOConsumer
from .zero_bubble.producer import SimpleProducer

ALGO_MAP = {"GRPO": GRPOConsumer, "DAPO": GRPOConsumer}


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
    data_actor_buffer_size_limit: int = 0,
):
    if core_algo not in ALGO_MAP:
        raise NotImplementedError(f"{core_algo} is not supported yet.")
    else:
        core_consumer = ALGO_MAP.get(core_algo, GRPOConsumer)

    train_dp_size = get_dp_size_fast(num_consumer_procs, plugin_config)
    assert (inference_batch_size * num_producers) % (train_batch_size * train_dp_size) == 0
    if data_actor_buffer_size_limit <= 0:
        # use 2 times the train_minibatch_size as the default buffer size limit
        data_actor_buffer_size_limit = train_minibatch_size * train_dp_size * 2

    dataset_path = train_dataset_config["path"]
    train_dataset_size = get_jsonl_size_fast(dataset_path)
    global_inference_batch_size = inference_batch_size * num_producers
    train_dataset_size = (train_dataset_size // global_inference_batch_size) * global_inference_batch_size

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

    # every producer is associated with a data worker, data worker is responsible for moving data from the producer to all consumer
    shared_sync_data_actor = SharedVariableActor.remote(num_consumer_procs, data_actor_buffer_size_limit)
    # all producer and the consumer 0 share the same model actor, model actor only provide signal for model synchronization
    shared_signal_actor = SharedVariableActor.remote()

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
        producer = SimpleProducer.options(num_gpus=num_proc_per_producer, num_cpus=4).remote(
            shared_sync_data_actor=shared_sync_data_actor,
            shared_signal_actor=shared_signal_actor,
            producer_idx=i,
            num_producers=num_producers,
            num_consumer_procs=num_consumer_procs,
            num_episodes=num_episodes,
            batch_size=inference_batch_size,
            train_dataset_config=train_dataset_config,
            model_config=inference_model_config,
            generate_config=generate_config,
            tokenizer_config=copy.deepcopy(tokenizer_config),
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
        )
        producer_procs.append(producer)
    # ray.get([p.setup.remote() for p in producer_procs])
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
        consumer = core_consumer.options(num_gpus=1, num_cpus=4).remote(
            shared_sync_data_actor=shared_sync_data_actor,
            shared_signal_actor=shared_signal_actor,
            num_producers=num_producers,
            num_episodes=num_episodes,
            rank=i,
            world_size=num_consumer_procs,
            master_addr=consumer_master_ip_address,
            master_port=master_port,
            train_dataset_size=train_dataset_size,
            batch_size=train_batch_size,
            model_config=train_model_config,
            plugin_config=plugin_config,
            minibatch_size=train_minibatch_size,
            tokenizer_config=copy.deepcopy(tokenizer_config),
            generate_config=generate_config_consumer,
            grpo_config=grpo_config,
            num_generations=num_generations,
            save_interval=save_interval,
            save_dir=save_dir,
            project_name=project_name,
            run_name=run_name,
            wandb_group_name=wandb_group_name,
            enable_profiling=enable_profiling,
        )
        consumer_procs.append(consumer)

    distributor_procs = []
    for i in range(num_producers):
        distributor_procs.append(
            Distributor.options(num_cpus=2).remote(
                i,
                plugin_config.get("pp_size", 1),
                num_producers,
                shared_signal_actor,
                enable_profiling=enable_profiling,
            )
        )
    print("=================== All processes are created, starting setup torch DDP ===================", flush=True)
    ray.get([p.setup.remote() for p in consumer_procs])
    print(
        "=================== All processes are setup, starting initialize communication groups ===================",
        flush=True,
    )
    remote_refs = []
    # Initialize consumer communication group
    for i, p in enumerate(consumer_procs):
        remote_refs.append(p.init_collective_group.remote(num_consumer_procs, i, "gloo", f"consumer_pg"))
    ray.get(remote_refs)
    remote_refs = []
    # Initialize producer communication group
    for i, p in enumerate(producer_procs):
        remote_refs.append(p.init_collective_group.remote(num_producers, i, "nccl", f"producer_pg"))
    ray.get(remote_refs)
    remote_refs = []
    # Initialize distributor communication group
    for i, p in enumerate(distributor_procs):
        remote_refs.append(p.init_collective_group.remote(num_producers, i, "gloo", f"distributor_pg"))
    ray.get(remote_refs)
    remote_refs = []
    # Initialize sync model communication group between consumer and sync model actor
    # As per tested, gloo do not support nested initialization, so we need to initialize all participants in the same group in the same ray.get call.
    consumer_pp = plugin_config.get("pp_size", 1)
    for i, p in enumerate(consumer_procs):
        consumer_ddp_config = ray.get(p.get_ddp_config.remote())
        if consumer_pp > 1:
            if consumer_ddp_config["tp_rank"] == 0 and consumer_ddp_config["dp_rank"] == 0:
                pp_rank = consumer_ddp_config["pp_rank"]
                remote_refs.append(
                    p.init_collective_group.remote(
                        num_producers + 1,
                        0,
                        backend="gloo",
                        group_name=f"sync_model_consumer_pp_{pp_rank}",
                        gloo_timeout=3000000,
                    )
                )
                for distributor_id, p_distributor in enumerate(distributor_procs):
                    remote_refs.append(
                        p_distributor.init_collective_group.remote(
                            num_producers + 1,
                            1 + distributor_id,
                            backend="gloo",
                            group_name=f"sync_model_consumer_pp_{pp_rank}",
                            gloo_timeout=3000000,
                        )
                    )
                ray.get(remote_refs)
                remote_refs = []
        else:
            if i == 0:
                remote_refs.append(
                    p.init_collective_group.remote(
                        num_producers + 1, 0, backend="gloo", group_name=f"sync_model_consumer", gloo_timeout=3000000
                    )
                )
                for distributor_id, p_distributor in enumerate(distributor_procs):
                    remote_refs.append(
                        p_distributor.init_collective_group.remote(
                            num_producers + 1,
                            1 + distributor_id,
                            backend="gloo",
                            group_name=f"sync_model_consumer",
                            gloo_timeout=3000000,
                        )
                    )
                ray.get(remote_refs)
                remote_refs = []
    # Initialize sync model communication group between producer and sync model actor
    for i, p in enumerate(producer_procs):
        if consumer_pp > 1:
            for pp_rank in range(consumer_pp):
                remote_refs.append(
                    p.init_collective_group.remote(
                        2, 0, backend="gloo", group_name=f"sync_model_producer_{i}_pp_{pp_rank}", gloo_timeout=3000000
                    )
                )
                remote_refs.append(
                    distributor_procs[i].init_collective_group.remote(
                        2, 1, backend="gloo", group_name=f"sync_model_producer_{i}_pp_{pp_rank}", gloo_timeout=3000000
                    )
                )
                ray.get(remote_refs)
                remote_refs = []
        else:
            remote_refs.append(
                p.init_collective_group.remote(
                    2, 0, backend="gloo", group_name=f"sync_model_producer_{i}", gloo_timeout=3000000
                )
            )
            remote_refs.append(
                distributor_procs[i].init_collective_group.remote(
                    2, 1, backend="gloo", group_name=f"sync_model_producer_{i}", gloo_timeout=3000000
                )
            )
            ray.get(remote_refs)
            remote_refs = []
    print("=================== All processes are set up, starting loop ===================", flush=True)
    ray.get([p.loop.remote() for p in (producer_procs + consumer_procs + distributor_procs)])
