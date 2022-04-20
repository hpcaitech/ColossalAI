import click
import subprocess
import collections
import sys
import os
import torch
from colossalai.context import Config
from .multinode_runner import PDSHRunner
from copy import deepcopy


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        click.echo(f"Error: Unable to find the hostfile, no such file: {hostfile_path}")
        exit()

    # e.g., worker-0:16
    with open(hostfile_path, 'r') as fd:
        device_pool = collections.OrderedDict()
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            try:
                hostname, slot_count = line.split(":")
                slot_count = int(slot_count)
            except ValueError as err:
                click.echo(f"Error: Hostfile is not formatted correctly, expected <hostname>:<slot>, but found {line}")
                exit()

            if hostname in device_pool:
                click.echo(f"Error: found duplicate host {hostname} in the hostfile")
                exit()
            device_pool[hostname] = slot_count
    return device_pool


def _stable_remove_duplicates(data):
    # Create a new list in the same order as original but with duplicates
    # removed, should never be more than ~16 elements so simple is best
    new_list = []
    for x in data:
        if x not in new_list:
            new_list.append(x)
    return new_list


def parse_device_filter(host_info, include_str=None, exclude_str=None):
    '''Parse an inclusion or exclusion string and filter a hostfile dictionary.

    Examples:
        include_str="worker-0@worker-1:0,2" will use all slots on worker-0 and
          slots [0, 2] on worker-1.
        exclude_str="worker-1:0" will use all available devices except
          slot 0 on worker-1.
    '''

    # Constants that define our syntax
    NODE_SEP = '@'
    SLOT_LIST_START = ':'
    SLOT_SEP = ','

    # Ensure include/exclude are mutually exclusive
    if include_str and exclude_str:
        click.echo("--include and --exclude are mutually exclusive, only one can be used")
        exit()

    # no-op
    if include_str is None and exclude_str is None:
        return host_info

    # Either build from scratch or remove items
    filtered_hosts = dict()
    if include_str:
        parse_str = include_str
    elif exclude_str:
        filtered_hosts = deepcopy(host_info)
        parse_str = exclude_str

    # foreach node in the list
    for node_config in parse_str.split(NODE_SEP):
        # Node can either be alone or node:slot,slot,slot
        if SLOT_LIST_START in node_config:
            hostname, slots = node_config.split(SLOT_LIST_START)
            slots = [int(x) for x in slots.split(SLOT_SEP)]

            # sanity checks
            if hostname not in host_info:
                click.echo(f"Hostname '{hostname}' not found in hostfile")
                exit()
            for slot in slots:
                if slot not in host_info[hostname]:
                    click.echo(f"No slot '{slot}' specified on host '{hostname}'")

            # If include string, build the list from here
            if include_str:
                filtered_hosts[hostname] = slots
            elif exclude_str:
                for slot in slots:
                    click.echo(f'- removing {slot} from {hostname}')
                    filtered_hosts[hostname].remove(slot)

        # User just specified the whole node
        else:
            hostname = node_config
            # sanity check hostname
            if hostname not in host_info:
                click.echo(f"Hostname '{hostname}' not found in hostfile")
                exit()

            if include_str:
                filtered_hosts[hostname] = host_info[hostname]
            elif exclude_str:
                filtered_hosts[hostname] = []

    # Post-processing to remove duplicates and empty nodes
    del_keys = []
    for hostname in filtered_hosts:
        # Remove duplicates
        filtered_hosts[hostname] = _stable_remove_duplicates(filtered_hosts[hostname])
        # Remove empty hosts
        if len(filtered_hosts[hostname]) == 0:
            del_keys.append(hostname)

    # remove unneeded hosts
    for name in del_keys:
        del filtered_hosts[name]

    # Lastly, go over filtered_hosts and convert to a OrderedDict() to ensure
    # we map ranks to nodes correctly by maintaining host_info ordering.
    ordered_hosts = collections.OrderedDict()
    for host in host_info:
        if host in filtered_hosts:
            ordered_hosts[host] = filtered_hosts[host]

    return ordered_hosts


def parse_inclusion_exclusion(device_pool, inclusion, exclusion):
    active_devices = collections.OrderedDict()
    for hostname, slots in device_pool.items():
        active_devices[hostname] = list(range(slots))

    return parse_device_filter(active_devices, include_str=inclusion, exclude_str=exclusion)


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
        device_pool = fetch_hostfile(args.hostfile)
    else:
        device_pool = None

    # filter and only keep the ones needed
    active_devices = None
    if device_pool:
        active_devices = parse_inclusion_exclusion(device_pool, args.include, args.exclude)

        if args.num_nodes > 0:
            # only keep the first num_nodes to execute jobs
            updated_active_devices = collections.OrderedDict()
            for count, hostname in enumerate(active_devices.keys()):
                if args.num_nodes == count:
                    break
                updated_active_devices[hostname] = active_devices[hostname]
            active_devices = updated_active_devices

        if args.nproc_per_node > 0:
            # only keep the first
            updated_active_devices = collections.OrderedDict()
            for hostname, active_devices in active_devices.items():
                if len(active_devices) < args.nproc_per_node:
                    click.echo(
                        f"Error: The number of available GPUs on {hostname} is smaller than the argument nproc_per_node"
                    )
                    exit()
                updated_active_devices[hostname] = active_devices[args.nproc_per_node]
            active_devices = updated_active_devices

    env = os.environ.copy()

    # use hosts if hostfile is not given
    if args.host and active_devices is None:
        hostinfo = collections.OrderedDict()
        host_list = args.host.strip().split(',')
        for hostname in host_list:
            hostinfo[hostname] = args.nproc_per_node
        active_devices = hostinfo

    # run on local node if not hosts or hostfile is given
    if not active_devices:
        if args.nproc_per_node == -1 or args.nproc_per_node > torch.cuda.device_count():
            nproc_per_node = torch.cuda.device_count()
        else:
            nproc_per_node = args.nproc_per_node

        if torch.__version__ <= "1.9":
            cmd = [
                sys.executable, "-u", "-m", "torch.distributed.launch", f"--nproc_per_node={nproc_per_node}",
                f"--master_addr={args.master_addr}", f"--master_port={args.master_port}"
            ] + [args.user_script] + args.user_args
        else:
            cmd = [
                "torchrun", f"--nproc_per_node={nproc_per_node}", f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}"
            ] + [args.user_script] + args.user_args
    else:
        runner = PDSHRunner(args)

        curr_path = os.path.abspath('.')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = curr_path + ":" + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = curr_path

        cmd = runner.get_cmd(env, active_devices, args)

    result = subprocess.Popen(cmd, env=env)
    result.wait()
    if result.returncode > 0:
        sys.exit(result.returncode)
