import click

from colossalai.context import Config

from .run import launch_multi_processes


@click.command(
    help="Launch distributed training on a single node or multiple nodes",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-H",
    "-host",
    "--host",
    type=str,
    default=None,
    help="the list of hostnames to launch in the format <host1>,<host2>",
)
@click.option(
    "--hostfile",
    type=str,
    default=None,
    help="Hostfile path that defines the device pool available to the job, each line in the file is a hostname",
)
@click.option(
    "--include",
    type=str,
    default=None,
    help="Specify computing devices to use during execution. String format is <host1>,<host2>,"
    " only effective when used with --hostfile.",
)
@click.option(
    "--exclude",
    type=str,
    default=None,
    help="Specify computing devices to NOT use during execution. Mutually exclusive with --include. Formatting is the same as --include,"
    " only effective when used with --hostfile.",
)
@click.option(
    "--num_nodes",
    type=int,
    default=-1,
    help="Total number of worker nodes to use, only effective when used with --hostfile.",
)
@click.option("--nproc_per_node", type=int, default=None, help="Number of GPUs to use on each node.")
@click.option(
    "--master_port",
    type=int,
    default=29500,
    help="(optional) Port used by PyTorch distributed for communication during distributed training.",
)
@click.option(
    "--master_addr",
    type=str,
    default="127.0.0.1",
    help="(optional) IP address of node 0, will be inferred via 'hostname -I' if not specified.",
)
@click.option(
    "--extra_launch_args",
    type=str,
    default=None,
    help="Set additional torch distributed launcher arguments such as --standalone. The format is --extra_launch_args arg1=1,arg2=2. "
    "This will be converted to --arg1=1 --arg2=2 during execution",
)
@click.option("--ssh-port", type=int, default=None, help="(optional) the port used for ssh connection")
@click.argument("user_script", type=str)
@click.argument("user_args", nargs=-1)
def run(
    host: str,
    hostfile: str,
    num_nodes: int,
    nproc_per_node: int,
    include: str,
    exclude: str,
    master_addr: str,
    master_port: int,
    extra_launch_args: str,
    ssh_port: int,
    user_script: str,
    user_args: str,
) -> None:
    """
    To launch multiple processes on a single node or multiple nodes via command line.

    Usage::
        # run with 4 GPUs on the current node use default port 29500
        colossalai run --nprocs_per_node 4 train.py

        # run with 2 GPUs on the current node at port 29550
        colossalai run --nprocs_per_node 4 --master_port 29550 train.py

        # run on two nodes
        colossalai run --host <host1>,<host2> --master_addr host1  --nprocs_per_node 4 train.py

        # run with hostfile
        colossalai run --hostfile <file_path> --master_addr <host>  --nprocs_per_node 4 train.py

        # run with hostfile with only included hosts
        colossalai run --hostfile <file_path> --master_addr host1 --include host1,host2  --nprocs_per_node 4 train.py

        # run with hostfile excluding the hosts selected
        colossalai run --hostfile <file_path> --master_addr host1 --exclude host2  --nprocs_per_node 4 train.py
    """
    if not user_script.endswith(".py"):
        click.echo(f"Error: invalid Python file {user_script}. Did you use a wrong option? Try colossalai run --help")
        exit()

    args_dict = locals()
    args = Config(args_dict)
    args.user_args = list(args.user_args)
    launch_multi_processes(args)
