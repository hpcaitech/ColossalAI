import click

from colossalai.context import Config

from .benchmark import run_benchmark
from .utils import *

__all__ = ['benchmark']


@click.command()
@click.option("-g", "--gpus", type=int, default=None, help="Total number of devices to use.")
@click.option("-b", "--batch_size", type=int, default=8, help="Batch size of the input tensor.")
@click.option("-s", "--seq_len", type=int, default=512, help="Sequence length of the input tensor.")
@click.option("-d", "--dimension", type=int, default=1024, help="Hidden dimension of the input tensor.")
@click.option("-w", "--warmup_steps", type=int, default=10, help="The number of warmup steps.")
@click.option("-p", "--profile_steps", type=int, default=50, help="The number of profiling steps.")
@click.option("-l", "--layers", type=int, default=2)
@click.option("-m",
              "--model",
              type=click.Choice(['mlp'], case_sensitive=False),
              default='mlp',
              help="Select the model to benchmark, currently only supports MLP")
def benchmark(gpus: int, batch_size: int, seq_len: int, dimension: int, warmup_steps: int, profile_steps: int,
              layers: int, model: str):
    args_dict = locals()
    args = Config(args_dict)
    run_benchmark(args)
