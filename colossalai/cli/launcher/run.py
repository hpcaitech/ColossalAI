import argparse
from argparse import ArgumentParser, REMAINDER
import subprocess
import collections
import sys
import os
import torch
from colossalai.logging import get_dist_logger
from .multinode_runner import PDSHRunner, OpenMPIRunner, SLURMRunner

def build_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="colossal distributed training launcher")

    parser.add_argument("-H",
                        "--hostfile",
                        type=str,
                        default="",
                        help="Hostfile path that defines the "
                        "device pool available to the job (e.g., "
                        "worker-name:number of slots)")

    parser.add_argument("-i",
                        "--include",
                        type=str,
                        default="",
                        help="Specify computing devices to use during execution."
                        "String format is NODE_SPEC@NODE_SPEC"
                        "where NODE_SPEC=<worker-name>:<list-of-slots>")

    parser.add_argument("-e",
                        "--exclude",
                        type=str,
                        default="",
                        help="Specify computing devices to NOT use during execution."
                        "Mutually exclusive with --include. Formatting"
                        "is the same as --include.")

    parser.add_argument("--num_nodes",
                        type=int,
                        default=-1,
                        help="Total number of worker nodes to use.")

    parser.add_argument("--num_gpus",
                        type=int,
                        default=-1,
                        help="Number of GPUs to use on each node.")

    parser.add_argument("--master_port",
                        default=29500,
                        type=int,
                        help="(optional) Port used by PyTorch distributed for "
                        "communication during distributed training.")

    parser.add_argument("--master_addr",
                        default="127.0.0.1",
                        type=str,
                        help="(optional) IP address of node 0, will be "
                        "inferred via 'hostname -I' if not specified.")

    parser.add_argument("--launcher",
                        default="torch",
                        type=str,
                        help="(optional) choose launcher backend for multi-node "
                        "training. Options currently include PDSH, OpenMPI, SLURM.")

    parser.add_argument("--launcher_args",
                        default="",
                        type=str,
                        help="(optional) pass launcher specific arguments as a "
                        "single quoted argument.")

    parser.add_argument("user_script",
                        type=str,
                        help="User script to launch, followed by any required "
                        "arguments.")
    
    parser.add_argument('user_args', nargs=argparse.REMAINDER)

    return parser

def fetch_hostfile(hostfile_path):
    logger = get_dist_logger()
    if not os.path.isfile(hostfile_path):
        logger.warning("Unable to find hostfile, will proceed with training "
                       "with local resources only")
        return None

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
                logger.error("Hostfile is not formatted correctly, unable to "
                             "proceed with training.")
                raise err
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

def parse_device_filter(host_info, include_str="", exclude_str=""):
    '''Parse an inclusion or exclusion string and filter a hostfile dictionary.

    Examples:
        include_str="worker-0@worker-1:0,2" will use all slots on worker-0 and
          slots [0, 2] on worker-1.
        exclude_str="worker-1:0" will use all available devices except
          slot 0 on worker-1.
    '''

    logger = get_dist_logger()

    # Constants that define our syntax
    NODE_SEP = '@'
    SLOT_LIST_START = ':'
    SLOT_SEP = ','

    # Ensure include/exclude are mutually exclusive
    if (include_str != "") and (exclude_str != ""):
        raise ValueError('include_str and exclude_str are mutually exclusive.')

    # no-op
    if (include_str == "") and (exclude_str == ""):
        return host_info

    # Either build from scratch or remove items
    filtered_hosts = dict()
    if include_str:
        parse_str = include_str
    if exclude_str != "":
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
                raise ValueError(f"Hostname '{hostname}' not found in hostfile")
            for slot in slots:
                if slot not in host_info[hostname]:
                    raise ValueError(f"No slot '{slot}' specified on host '{hostname}'")

            # If include string, build the list from here
            if include_str:
                filtered_hosts[hostname] = slots
            elif exclude_str:
                for slot in slots:
                    logger.info(f'removing {slot} from {hostname}')
                    filtered_hosts[hostname].remove(slot)

        # User just specified the whole node
        else:
            hostname = node_config
            # sanity check hostname
            if hostname not in host_info:
                raise ValueError(f"Hostname '{hostname}' not found in hostfile")

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

    return parse_device_filter(active_devices,
                                 include_str=inclusion,
                                 exclude_str=exclusion)

def main(args=None):
    logger = get_dist_logger()
    assert args is not None, "args should not be None."
    
    device_pool = fetch_hostfile(args.hostfile)

    active_devices = None
    if device_pool:
        active_devices = parse_inclusion_exclusion(device_pool,
                                                 args.include,
                                                 args.exclude)
        if args.num_nodes > 0:
            updated_active_devices = collections.OrderedDict()
            for count, hostname in enumerate(active_devices.keys()):
                if args.num_nodes == count:
                    break
                updated_active_devices[hostname] = active_devices[hostname]
            active_devices = updated_active_devices

        if args.num_gpus > 0:
            updated_active_devices = collections.OrderedDict()
            for hostname in active_devices.keys():
                updated_active_devices[hostname] = list(range(args.num_gpus))
            active_devices = updated_active_devices

    env = os.environ.copy()

    if not active_devices:
        if args.num_gpus == -1 or args.num_gpus > torch.cuda.device_count():
            nproc_per_node = torch.cuda.device_count()
        else:
            nproc_per_node = args.num_gpus
        if torch.__version__ <= "1.09":
            cmd = [sys.executable, "-u", "-m", 
                    "torch.distributed.launch",
                    f"--nproc_per_node={nproc_per_node}",
                    f"--master_addr={args.master_addr}",
                    f"--master_port={args.master_port}"] + [args.user_script] + args.user_args
        else:
            cmd = ["torchrun",
                    f"--nproc_per_node={nproc_per_node}",
                    f"--master_addr={args.master_addr}",
                    f"--master_port={args.master_port}"] + [args.user_script] + args.user_args
    else:
        if args.launcher == "torch":
            runner = PDSHRunner(args)
        elif args.launcher == "mpi":
            runner = OpenMPIRunner(args, device_pool)
        elif args.launcher == "slurm":
            runner = SLURMRunner(args, device_pool)
        else:
            raise NotImplementedError(f"Unknown launcher {args.launcher}")

        if not runner.backend_exists():
            raise RuntimeError(f"launcher '{args.launcher}' not installed.")

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


if __name__ == "__main__":
    main()
