import os
import sys
import shutil
import subprocess
import warnings
from shlex import quote
from abc import ABC, abstractmethod

from colossalai.logging import get_dist_logger

logger = get_dist_logger()

class MultiNodeRunner(ABC):
    def __init__(self, args):
        self.args = args
        self.user_arguments = self.args.user_args
        self.user_script = args.user_script
        self.exports = {}

    @abstractmethod
    def backend_exists(self):
        """Return whether the corresponding backend exists"""

    @abstractmethod
    def get_cmd(self, environment, active_devices):
        """Return the command to execute on node"""

    def add_export(self, key, var):
        self.exports[key.strip()] = var.strip()

    @property
    def name(self):
        """Return the name of the backend"""
        return self.__class__.__name__


class PDSHRunner(MultiNodeRunner):
    def __init__(self, args):
        super().__init__(args)

    def backend_exists(self):
        return shutil.which('pdsh')

    @property
    def name(self):
        return "pdsh"

    def parse_user_args(self):
        return list(
            map(lambda x: x if x.startswith("-") else f"'{x}'",
                self.args.user_args))

    def get_cmd(self, environment, active_devices, args):
        environment['PDSH_RCMD_TYPE'] = 'ssh'

        active_workers = ",".join(active_devices.keys())
        logger.info("Running on the following workers: %s" % active_workers)

        pdsh_cmd_args = ['pdsh', '-f', str(1024), '-w', active_workers]

        exports = ""
        for key, val in self.exports.items():
            exports += f"export {key}={quote(val)}; "

        # https://linux.die.net/man/1/pdsh
        # %n will be replaced by pdsh command
        colossal_launch = [
            exports,
            f"cd {os.path.abspath('.')};",
            sys.executable, "-u", "-m", 
            "torch.distributed.launch",
            f"--nproc_per_node={args.num_gpus}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}"
        ]
        return pdsh_cmd_args + colossal_launch + [self.user_script] + self.user_arguments

class OpenMPIRunner(MultiNodeRunner):
    def __init__(self, args, device_pool):
        super().__init__(args)
        self.device_pool = device_pool

    def backend_exists(self):
        return shutil.which('ompi_info')

    @property
    def name(self):
        return "openmpi"

    def get_cmd(self, environment, active_devices):
        total_process_count = sum(self.device_pool.values())

        mpirun_cmd = [
            'mpirun',
            '-n',
            f'{total_process_count}',
            '-hostfile',
            f'{self.args.hostfile}'
        ]

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-x', f'{k}={quote(v)}']

        python_exec = []
        python_exec = [sys.executable, "-u", "-m"]

        return mpirun_cmd + export_cmd + python_exec + [self.user_script
                                                        ] + self.user_arguments

class SLURMRunner(MultiNodeRunner):
    def __init__(self, args):
        super().__init__(args)

    def backend_exists(self):
        return shutil.which('slurm_info')

    @property
    def name(self):
        return "slurm"

    def get_cmd(self, environment, active_devices, args):

        assert "-p" in args.launcher_args
        srun_args = args.launcher_args.strip().split()
        assert len(srun_args) >= 2, "we need more info about which partition to use."
        partition_name = srun_args(srun_args.index("-p")+1)
        slurm_cmd = [
            'srun',
            "-p",
            f"{partition_name}",
            "--nodes",
            f"{args.num_nodes}",
            "--tasks",
            f"{args.num_gpus}"
        ]

        export_cmd = []
        for k, v in self.exports.items():
            export_cmd += ['-x', f'{k}={quote(v)}']

        python_exec = []
        python_exec = [sys.executable, "-u", "-m"]

        return slurm_cmd + export_cmd + python_exec + [self.user_script
                                                        ] + self.user_arguments
