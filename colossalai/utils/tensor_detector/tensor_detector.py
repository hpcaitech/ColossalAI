# This tool was inspired by https://github.com/Stonesjtu/pytorch_memlab/blob/master/pytorch_memlab/mem_reporter.py
# and https://github.com/Oldpan/Pytorch-Memory-Utils

import gc
import math
import inspect
import torch
import torch.nn as nn
from typing import Optional
from collections import defaultdict


LINE_WIDTH = 108

class TensorDetector():
    def __init__(self,
                 include_cpu: bool = False,
                 module: Optional[nn.Module] = None
                 ):
        """This class is an dectector to detect tensor on different devices.
        
        :param include_cpu: whether to detect tensor on cpu, default False
        :type include_cpu: bool
        :param module: when sending an `nn.Module` it, the detector can name the tensors detected better
        :type module: Optional[nn.Module]

        """
        self.include_cpu = include_cpu 
        self.tensors = defaultdict(list)
        self.order = []
        self.devices = []
        self.pre_order = []
        self.module = module

        if isinstance(module, nn.Module):
            # if module is an instance of nn.Module, we can name the parameter with its real name
            for name, param in module.named_parameters():
                self.tensors[id(param)].append(name)
                self.tensors[id(param)].append(param.device)
                self.tensors[id(param)].append(param.shape)
                self.tensors[id(param)].append(param.requires_grad)
                self.tensors[id(param)].append(param.dtype)
                self.tensors[id(param)].append(self.get_tensor_mem(param))
                

    def get_tensor_mem(self, tensor):
        # calculate the memory occupied by a tensor
        memory_size = tensor.element_size() * tensor.storage().size()
        if (tensor.is_leaf or tensor.retains_grad) and tensor.grad is not None:
            grad_memory_size = tensor.grad.element_size() * tensor.grad.storage().size()
            memory_size += grad_memory_size
        return self.mem_format(memory_size)


    def mem_format(self, real_memory_size):
        # format the tensor memory into a reasonal magnitude
        if real_memory_size >= 2 ** 30:
            return str(real_memory_size / (2 ** 30)) + ' GB'
        if real_memory_size >= 2 ** 20:
            return str(real_memory_size / (2 ** 20)) + ' MB'
        if real_memory_size >= 2 ** 10:
            return str(real_memory_size / (2 ** 10)) + ' KB'
        return str(real_memory_size) + ' B' 
        

    def collect_tensors_state(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                # skip cpu tensor when include_cpu is false and the tensor we have collected before
                if (not self.include_cpu) and obj.device == torch.device('cpu') or id(obj) in self.pre_order:
                    continue
                # skip paramters we had added in __init__ when module is an instance of nn.Module
                if id(obj) not in self.tensors:
                    name = type(obj).__name__
                    # after update the parameters, the ids changed, but we could still mark them.
                    if isinstance(self.module, nn.Module) and name == 'Parameter' and obj.grad is not None: 
                        for par_name, param in self.module.named_parameters():
                            if param.requires_grad and param.grad.equal(obj.grad):
                                name = par_name + ' (with grad)'
                    # in the case of common tensor, we can't
                    # but we can still marked it as tensor(with grad)
                    # actually, we can, by recording the data and comparing
                    # that requires much more memory
                    if name == 'Tensor' and (obj.is_leaf or obj.retains_grad):
                        if obj.grad is not None:
                            name = name + ' (with grad)'

                    self.tensors[id(obj)].append(name)
                    self.tensors[id(obj)].append(obj.device)
                    self.tensors[id(obj)].append(obj.shape)
                    self.tensors[id(obj)].append(obj.requires_grad)
                    self.tensors[id(obj)].append(obj.dtype)
                    self.tensors[id(obj)].append(self.get_tensor_mem(obj))
                # recorded the order we got the tensor
                # by this we can guess the tensor easily
                self.order.append(id(obj))
                # recorded all different devices
                if obj.device not in self.devices:
                    self.devices.append(obj.device)

    
    def print_tensors_state(self):
        template_format = '{:3s}{:<30s}{:>10s}{:>20s}{:>10s}{:>20s}{:>15s}'
        print('-' * LINE_WIDTH)
        print(template_format.format('  ', 'Tensor', 'device', 'shape', 'grad', 'dtype', 'Mem'))
        print('-' * LINE_WIDTH)


        for tensor_id in self.order:
            print(template_format.format('+',
                                        str(self.tensors[tensor_id][0]),
                                        str(self.tensors[tensor_id][1]),
                                        str(tuple(self.tensors[tensor_id][2])),
                                        str(self.tensors[tensor_id][3]),
                                        str(self.tensors[tensor_id][4]),
                                        str(self.tensors[tensor_id][5])))
        
        # trace where is the detect()
        locate_info = inspect.stack()[2]
        locate_msg = '"' +  locate_info.filename + '" line ' + str(locate_info.lineno)
        print('-' * LINE_WIDTH)
        print(f"Detect Location: {locate_msg}")
        for device in self.devices:
            if device == torch.device('cpu'):
                continue
            gpu_mem_alloc = self.mem_format(torch.cuda.memory_allocated(device))
            print(f"Totle GPU Memery Allocated on {device} is {gpu_mem_alloc}")
        print('-' * LINE_WIDTH)
        print('\n\n')
    
    def detect(self):
        self.collect_tensors_state()
        self.print_tensors_state()
        self.tensors.clear()
        self.pre_order = self.order.copy()
        self.order = []
