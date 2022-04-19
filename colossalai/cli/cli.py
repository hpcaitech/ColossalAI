import click
from colossalai.cli.launcher.run import main as col_launch
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
@click.option("--num_gpus",
                type=int,
                default=-1)
@click.option("--bs",
                type=int,
                default=BATCH_SIZE)
@click.option("--seq_len",
                type=int,
                default=SEQ_LENGTH)
@click.option("--hid_dim",
                type=int,
                default=HIDDEN_DIM)
@click.option("--num_steps",
                type=int,
                default=ITER_TIMES)
def benchmark(num_gpus, bs, seq_len, hid_dim, num_steps):
    args_dict = locals()
    args = Arguments(args_dict)
    col_benchmark(args)

@click.command()
@click.option("--hostfile", 
                type=str,
                default="")
@click.option("--include",
              type=str,
              default="")
@click.option("--exclude",
                type=str,
                default="")
@click.option("--num_nodes",
                type=int,
                default=-1)
@click.option("--num_gpus",
                type=int,
                default=-1)
@click.option("--master_port",
                type=int,
                default=29500)
@click.option("--master_addr",
                type=str,
                default="127.0.0.1")
@click.option("--launcher",
                type=str,
                default="torch")
@click.option("--launcher_args",
                type=str,
                default="")
@click.argument("user_script",
                type=str)
@click.argument('user_args', nargs=-1)
def launch(hostfile, num_nodes, num_gpus, include, exclude, master_addr, master_port, 
           launcher, launcher_args, user_script, user_args):
    args_dict = locals()
    args = Arguments(args_dict)
    args.user_args = list(args.user_args)
    col_launch(args)

cli.add_command(launch)
cli.add_command(benchmark)

if __name__ == '__main__':
    cli()
