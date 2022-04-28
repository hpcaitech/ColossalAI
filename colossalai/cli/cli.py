import click
from .launcher import run
from .check import check
from .benchmark import benchmark


class Arguments():

    def __init__(self, arg_dict):
        for k, v in arg_dict.items():
            self.__dict__[k] = v


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(check)
cli.add_command(benchmark)

if __name__ == '__main__':
    cli()
