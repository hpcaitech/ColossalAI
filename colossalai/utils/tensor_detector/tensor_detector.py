import gc
import inspect
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn

LINE_WIDTH = 108
LINE = "-" * LINE_WIDTH + "\n"


class TensorDetector:
    def __init__(
        self, show_info: bool = True, log: str = None, include_cpu: bool = False, module: Optional[nn.Module] = None
    ):
        """This class is a detector to detect tensor on different devices.

        Args:
            show_info (bool, optional): whether to print the info on screen, default True.
            log (str, optional): the file name to save the log. Defaults to None.
            include_cpu (bool, optional): whether to detect tensor on cpu, default False.
            module (Optional[:class:`nn.Module`]): when sending an ``nn.Module`` object,
                the detector can name the tensors detected better.
        """
        self.show_info = show_info
        self.log = log
        self.include_cpu = include_cpu
        self.tensor_info = defaultdict(list)
        self.saved_tensor_info = defaultdict(list)
        self.order = []
        self.detected = []
        self.devices = []
        self.info = ""

        self.module = module
        if isinstance(module, nn.Module):
            # if module is an instance of nn.Module, we can name the parameter with its real name
            for name, param in module.named_parameters():
                self.tensor_info[id(param)].append(name)
                self.tensor_info[id(param)].append(param.device)
                self.tensor_info[id(param)].append(param.shape)
                self.tensor_info[id(param)].append(param.requires_grad)
                self.tensor_info[id(param)].append(param.dtype)
                self.tensor_info[id(param)].append(self.get_tensor_mem(param))

    def get_tensor_mem(self, tensor):
        # calculate the memory occupied by a tensor
        memory_size = tensor.element_size() * tensor.storage().size()
        if (tensor.is_leaf or tensor.retains_grad) and tensor.grad is not None:
            grad_memory_size = tensor.grad.element_size() * tensor.grad.storage().size()
            memory_size += grad_memory_size
        return self.mem_format(memory_size)

    def mem_format(self, real_memory_size):
        # format the tensor memory into a reasonable magnitude
        if real_memory_size >= 2**30:
            return str(real_memory_size / (2**30)) + " GB"
        if real_memory_size >= 2**20:
            return str(real_memory_size / (2**20)) + " MB"
        if real_memory_size >= 2**10:
            return str(real_memory_size / (2**10)) + " KB"
        return str(real_memory_size) + " B"

    def collect_tensors_state(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                # skip cpu tensor when include_cpu is false and the tensor we have collected before
                if (not self.include_cpu) and obj.device == torch.device("cpu"):
                    continue
                self.detected.append(id(obj))
                # skip parameters we had added in __init__ when module is an instance of nn.Module for the first epoch
                if id(obj) not in self.tensor_info:
                    name = type(obj).__name__
                    # after backward, we want to update the records, to show you the change
                    if isinstance(self.module, nn.Module) and name == "Parameter":
                        if obj.grad is not None:
                            # with grad attached
                            for par_name, param in self.module.named_parameters():
                                if param.requires_grad and param.grad.equal(obj.grad):
                                    name = par_name + " (with grad)"
                        else:
                            # with no grad attached
                            # there will be no new parameters created during running
                            # so it must be in saved_tensor_info
                            continue
                    # we can also marked common tensors as tensor(with grad)
                    if name == "Tensor" and (obj.is_leaf or obj.retains_grad):
                        if obj.grad is not None:
                            name = name + " (with grad)"
                    # in fact, common tensor have no grad
                    # unless you set retain_grad()
                    if id(obj) in self.saved_tensor_info.keys() and name == self.saved_tensor_info[id(obj)][0]:
                        continue

                    self.tensor_info[id(obj)].append(name)
                    self.tensor_info[id(obj)].append(obj.device)
                    self.tensor_info[id(obj)].append(obj.shape)
                    self.tensor_info[id(obj)].append(obj.requires_grad)
                    self.tensor_info[id(obj)].append(obj.dtype)
                    self.tensor_info[id(obj)].append(self.get_tensor_mem(obj))
                # recorded the order we got the tensor
                # by this we can guess the tensor easily
                # it will record every tensor updated this turn
                self.order.append(id(obj))
                # recorded all different devices
                if obj.device not in self.devices:
                    self.devices.append(obj.device)

    def print_tensors_state(self):
        template_format = "{:3s}{:<30s}{:>10s}{:>20s}{:>10s}{:>20s}{:>15s}"
        self.info += LINE
        self.info += template_format.format("  ", "Tensor", "device", "shape", "grad", "dtype", "Mem")
        self.info += "\n"
        self.info += LINE

        # if a tensor updates this turn, and was recorded before
        # it should be updated in the saved_tensor_info as well
        outdated = [x for x in self.saved_tensor_info.keys() if x in self.order]
        minus = [x for x in self.saved_tensor_info.keys() if x not in self.detected]
        minus = outdated + minus
        if len(self.order) > 0:
            for tensor_id in self.order:
                self.info += template_format.format(
                    "+",
                    str(self.tensor_info[tensor_id][0]),
                    str(self.tensor_info[tensor_id][1]),
                    str(tuple(self.tensor_info[tensor_id][2])),
                    str(self.tensor_info[tensor_id][3]),
                    str(self.tensor_info[tensor_id][4]),
                    str(self.tensor_info[tensor_id][5]),
                )
                self.info += "\n"
        if len(self.order) > 0 and len(minus) > 0:
            self.info += "\n"
        if len(minus) > 0:
            for tensor_id in minus:
                self.info += template_format.format(
                    "-",
                    str(self.saved_tensor_info[tensor_id][0]),
                    str(self.saved_tensor_info[tensor_id][1]),
                    str(tuple(self.saved_tensor_info[tensor_id][2])),
                    str(self.saved_tensor_info[tensor_id][3]),
                    str(self.saved_tensor_info[tensor_id][4]),
                    str(self.saved_tensor_info[tensor_id][5]),
                )
                self.info += "\n"
                # deleted the updated tensor
                self.saved_tensor_info.pop(tensor_id)

        # trace where is the detect()
        locate_info = inspect.stack()[2]
        locate_msg = '"' + locate_info.filename + '" line ' + str(locate_info.lineno)

        self.info += LINE
        self.info += f"Detect Location: {locate_msg}\n"
        for device in self.devices:
            if device == torch.device("cpu"):
                continue
            gpu_mem_alloc = self.mem_format(torch.cuda.memory_allocated(device))
            self.info += f"Total GPU Memory Allocated on {device} is {gpu_mem_alloc}\n"
        self.info += LINE
        self.info += "\n\n"
        if self.show_info:
            print(self.info)
        if self.log is not None:
            with open(self.log + ".log", "a") as f:
                f.write(self.info)

    def detect(self, include_cpu=False):
        self.include_cpu = include_cpu
        self.collect_tensors_state()
        self.print_tensors_state()
        self.saved_tensor_info.update(self.tensor_info)
        self.tensor_info.clear()
        self.order = []
        self.detected = []
        self.info = ""

    def close(self):
        self.saved_tensor_info.clear()
        self.module = None
