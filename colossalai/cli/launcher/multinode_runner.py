import os
import sys
import shutil
from shlex import quote
from abc import ABC, abstractmethod

from colossalai.logging import get_dist_logger


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
        return list(map(lambda x: x if x.startswith("-") else f"'{x}'", self.args.user_args))

    def get_cmd(self, environment, active_devices, args):
        environment['PDSH_RCMD_TYPE'] = 'ssh'

        active_workers = ",".join(active_devices.keys())
        print("Running on the following workers: %s" % active_workers)

        pdsh_cmd_args = ['pdsh', '-f', str(1024), '-w', active_workers]

        exports = ""
        for key, val in self.exports.items():
            exports += f"export {key}={quote(val)}; "

        # https://linux.die.net/man/1/pdsh
        # %n will be replaced by pdsh command
        colossal_launch = [
            exports, f"cd {os.path.abspath('.')};", sys.executable, "-u", "-m", "torch.distributed.launch",
            f"--nproc_per_node={args.nproc_per_node}", f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}"
        ]
        return pdsh_cmd_args + colossal_launch + [self.user_script] + self.user_arguments
