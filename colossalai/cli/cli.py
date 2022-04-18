import click
from colossalai.cli.launcher.run import main as col_launch

class Arguments():
    def __init__(self, dict):
        for k, v in dict.items():
            self.__dict__[k] = v

@click.group()
def cli():
    pass

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

if __name__ == '__main__':
    cli()
