import click
import sys
import os
import torch
from colossalai.context import Config
from .multinode_runner import MultiNodeRunner
from .hostinfo import HostInfo, HostInfoList
from typing import List
from packaging import version

# Constants that define our syntax
NODE_SEP = ','


def fetch_hostfile(hostfile_path: str, ssh_port: int) -> HostInfoList:
    if not os.path.isfile(hostfile_path):
        click.echo(f"Error: Unable to find the hostfile, no such file: {hostfile_path}")
        exit()

    # e.g., worker-0:16
    with open(hostfile_path, 'r') as fd:
        device_pool = HostInfoList()

        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            hostname = line.strip()
            hostinfo = HostInfo(hostname=hostname, port=ssh_port)

            if device_pool.has(hostname):
                click.echo(f"Error: found duplicate host {hostname} in the hostfile")
                exit()
            device_pool.append(hostinfo)
    return device_pool


def parse_device_filter(device_pool: HostInfoList, include_str=None, exclude_str=None) -> HostInfoList:
    '''Parse an inclusion or exclusion string and filter a hostfile dictionary.

    Examples:
        include_str="worker-0,worker-1" will execute jobs only on worker-0 and worker-1.
        exclude_str="worker-1" will use all available devices except worker-1.
    '''

    # Ensure include/exclude are mutually exclusive
    if include_str and exclude_str:
        click.echo("--include and --exclude are mutually exclusive, only one can be used")
        exit()

    # no-op
    if include_str is None and exclude_str is None:
        return device_pool

    # Either build from scratch or remove items
    if include_str:
        parse_str = include_str
        filtered_hosts = HostInfoList()
    elif exclude_str:
        parse_str = exclude_str
        filtered_hosts = device_pool

    # foreach node in the list
    for node_config in parse_str.split(NODE_SEP):
        hostname = node_config
        hostinfo = device_pool.get_hostinfo(hostname)
        # sanity check hostname
        if not device_pool.has(hostname):
            click.echo(f"Error: Hostname '{hostname}' not found in hostfile")
            exit()

        if include_str:
            filtered_hosts.append(hostinfo)
        elif exclude_str:
            filtered_hosts.remove(hostname)

    return filtered_hosts


def get_launch_command(master_addr: str,
                       master_port: int,
                       nproc_per_node: int,
                       user_script: str,
                       user_args: List[str],
                       node_rank: int = 0,
                       num_nodes: int = 1):
    if version.parse(torch.__version__) < version.parse("1.10"):
        cmd = [
            sys.executable, "-u", "-m", "torch.distributed.launch", f"--nproc_per_node={nproc_per_node}",
            f"--master_addr={master_addr}", f"--master_port={master_port}", f"--nnodes={num_nodes}",
            f"--node_rank={node_rank}"
        ]
    else:
        cmd = [
            "torchrun", f"--nproc_per_node={nproc_per_node}", f"--master_addr={master_addr}",
            f"--master_port={master_port}", f"--nnodes={num_nodes}", f"--node_rank={node_rank}"
        ]

    cmd += [user_script] + user_args
    cmd = ' '.join(cmd)
    return cmd


def launch_multi_processes(args):
    """
    Launch multiple processes on a single node or multiple nodes.

    The overall logic can be summarized as the pseudo code below:
    
    if hostfile given:
        hostinfo = parse_hostfile(hostfile)
        hostinfo = include_or_exclude_hosts(hostinfo)
        launch_on_multi_nodes(hostinfo)
    elif hosts given:
        hostinfo = parse_hosts(hosts)
        launch_on_multi_nodes(hostinfo)
    else:
        launch_on_current_node()
    """
    assert isinstance(args, Config)

    # cannot accept hosts and hostfile at the same time
    if args.host and args.hostfile:
        click.echo("Error: hostfile and hosts are mutually exclusive, only one is required")

    # check if hostfile is given
    if args.hostfile:
        device_pool = fetch_hostfile(args.hostfile, ssh_port=args.ssh_port)
        active_device_pool = parse_device_filter(device_pool, args.include, args.exclude)

        if args.num_nodes > 0:
            # only keep the first num_nodes to execute jobs
            updated_active_device_pool = HostInfoList()
            for count, hostinfo in enumerate(active_device_pool):
                if args.num_nodes == count:
                    break
                updated_active_device_pool.append(hostinfo)
            active_device_pool = updated_active_device_pool

        if args.nproc_per_node > 0:
            # only keep the first nproc_per_node GPUs
            for hostinfo in active_device_pool:
                if hostinfo.num_slots < args.nproc_per_node:
                    click.echo(
                        f"Error: The number of available GPUs on {hostinfo.hostname} is smaller than the argument nproc_per_node"
                    )
                    exit()
                hostinfo.slots = hostinfo.slots[:args.nproc_per_node]

    else:
        active_device_pool = None

    env = os.environ.copy()

    # use hosts if hostfile is not given
    if args.host and active_device_pool is None:
        active_device_pool = HostInfoList()
        host_list = args.host.strip().split(',')
        for hostname in host_list:
            hostinfo = HostInfo(hostname=hostname, port=args.ssh_port)
            active_device_pool.append(hostinfo)

    if not active_device_pool:
        # run on local node if not hosts or hostfile is given
        # add local node to host info list
        active_device_pool = HostInfoList()
        localhost_info = HostInfo(hostname='127.0.0.1', port=args.ssh_port)
        active_device_pool.append(localhost_info)

        # use all gpus by default if nproc_per_node is not given for single-node run
        if args.nproc_per_node == -1:
            args.nproc_per_node = torch.cuda.device_count()
            click.echo("Warning: nproc_per_node is not given, use all available GPUs instead")

    else:
        # run on multi-node
        if args.nproc_per_node < 1:
            click.echo("Error: nproc_per_node is not specified or smaller than 1 for the multi-node run")
            exit()

    # some machines may not use 22 as the default port,
    # you can set the port number on your own
    if args.ssh_port:
        for hostinfo in active_device_pool:
            hostinfo.port = args.ssh_port

    runner = MultiNodeRunner()
    curr_path = os.path.abspath('.')

    # collect current path env
    env = dict()
    for k, v in os.environ.items():
        if v and '\n' not in v:
            env[k] = v

    # establish remote connection
    runner.connect(host_info_list=active_device_pool, workdir=curr_path, env=env)

    for node_id, hostinfo in enumerate(active_device_pool):
        cmd = get_launch_command(master_addr=args.master_addr,
                                 master_port=args.master_port,
                                 nproc_per_node=args.nproc_per_node,
                                 user_script=args.user_script,
                                 user_args=args.user_args,
                                 node_rank=node_id,
                                 num_nodes=len(active_device_pool))
        runner.send_to_remote(hostinfo=hostinfo, cmd=cmd)

    runner.recv_from_all()
    runner.stop_all()
    runner.recv_from_all()
