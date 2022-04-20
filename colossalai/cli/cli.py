import click
from .launcher import run
from colossalai.cli.benchmark.utils import BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM, ITER_TIMES
from colossalai.cli.benchmark.run import launch as col_benchmark


class Arguments():

    def __init__(self, arg_dict):
        for k, v in arg_dict.items():
            self.__dict__[k] = v


@click.group()
def cli():
    pass


@click.command()
@click.option("--num_gpus", type=int, default=-1)
@click.option("--bs", type=int, default=BATCH_SIZE)
@click.option("--seq_len", type=int, default=SEQ_LENGTH)
@click.option("--hid_dim", type=int, default=HIDDEN_DIM)
@click.option("--num_steps", type=int, default=ITER_TIMES)
def benchmark(num_gpus, bs, seq_len, hid_dim, num_steps):
    args_dict = locals()
    args = Arguments(args_dict)
    col_benchmark(args)


cli.add_command(run)
cli.add_command(benchmark)

if __name__ == '__main__':
    cli()
