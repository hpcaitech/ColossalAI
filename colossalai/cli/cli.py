import click

from .check import check
from .launcher import run


class Arguments:
    def __init__(self, arg_dict):
        for k, v in arg_dict.items():
            self.__dict__[k] = v


@click.group()
def cli():
    pass


cli.add_command(run)
cli.add_command(check)

if __name__ == "__main__":
    cli()
