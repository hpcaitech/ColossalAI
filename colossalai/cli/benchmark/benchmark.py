from functools import partial
from typing import Dict, List

import click
import torch.multiprocessing as mp

import colossalai
from colossalai.cli.benchmark.utils import find_all_configs, get_batch_data, profile_model
from colossalai.context import Config
from colossalai.context.random import reset_seeds
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils import MultiTimer, free_port

from .models import MLP


def run_benchmark(args: Config) -> None:
    """
    Run benchmarking with torch.multiprocessing.
    """

    # sanity checks
    if args.gpus is None:
        click.echo("Error: --num_gpus is not given")
        exit()
    if args.gpus <= 1:
        click.echo("Warning: tensor parallel will be activated with at least 2 devices.")

    click.echo("=== Benchmarking Parameters ===")
    for k, v in args.items():
        click.echo(f'{k}: {v}')
    click.echo('')

    config_list = find_all_configs(args.gpus)

    avail_ports = [free_port() for _ in range(len(config_list))]
    run_func = partial(run_dist_profiling,
                       world_size=args.gpus,
                       port_list=avail_ports,
                       config_list=config_list,
                       hyperparams=args)
    mp.spawn(run_func, nprocs=args.gpus)


def run_dist_profiling(rank: int, world_size: int, port_list: List[int], config_list: List[Dict],
                       hyperparams: Config) -> None:
    """
    A function executed for profiling, this function should be spawn by torch.multiprocessing.

    Args:
        rank (int): rank of the process
        world_size (int): the number of processes
        port_list (List[int]): a list of free ports for initializing distributed networks
        config_list (List[Dict]): a list of configuration
        hyperparams (Config): the hyperparameters given by the user

    """

    # disable logging for clean output
    disable_existing_loggers()
    logger = get_dist_logger()
    logger.set_level('WARNING')

    for config, port in zip(config_list, port_list):
        colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
        timer = MultiTimer()

        # 1D parallel should be skipped if in_features or out_features is not able to be divided exactly by 1D parallel size.
        if config.parallel.tensor.mode == '1d' and hyperparams.dimension % config.parallel.tensor.size != 0:
            click.echo(
                "1D parallel will be skipped because in_features or out_features is not able to be divided exactly by 1D parallel size."
            )
            continue

        if hyperparams.model == 'mlp':
            model = MLP(dim=hyperparams.dimension, layers=hyperparams.layers)
        else:
            if gpc.get_global_rank() == 0:
                click.echo("Error: Invalid argument for --model")
                exit()

        data_func = partial(get_batch_data,
                            dim=hyperparams.dimension,
                            batch_size=hyperparams.batch_size,
                            seq_length=hyperparams.seq_len,
                            mode=config.parallel.tensor.mode)

        fwd_time, bwd_time, max_allocated, max_cached = profile_model(model=model,
                                                                      warmup_steps=hyperparams.warmup_steps,
                                                                      profile_steps=hyperparams.profile_steps,
                                                                      data_func=data_func,
                                                                      timer=timer)

        gpc.destroy()
        reset_seeds()

        if gpc.get_global_rank() == 0:
            config_str = ', '.join([f'{k}: {v}' for k, v in config.parallel.tensor.items()])
            click.echo(f"=== {config_str} ===")
            click.echo(f"Average forward time: {fwd_time}")
            click.echo(f"Average backward time: {bwd_time}")
            click.echo(f"Max allocated GPU memory: {max_allocated}")
            click.echo(f"Max cached GPU memory: {max_cached}\n")
