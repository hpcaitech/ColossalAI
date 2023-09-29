import click

from .check_installation import check_installation

__all__ = ["check"]


@click.command(help="Check if Colossal-AI is correct based on the given option")
@click.option("-i", "--installation", is_flag=True, help="Check if Colossal-AI is built correctly")
def check(installation):
    if installation:
        check_installation()
        return
    click.echo("No option is given")
