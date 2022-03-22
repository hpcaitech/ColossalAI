import functools
from typing import Optional

import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.memory_tracer.model_data_memtracer import \
    GLOBAL_MODEL_DATA_TRACER
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model._zero3_utils import cast_tensor_to_fp16
from colossalai.zero.sharded_param import ShardedParamV2
from torch.distributed import ProcessGroup

# Inserts _post_init_method at the end of init method


# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):

    def __init__(self):
        pass

    def __enter__(self):
        r"""
        Enter the context scope.
        """

        def preprocess_after(f):

            @functools.wraps(f)
            def wrapper(module: torch.nn.Module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        # The function is called during init subclass.
        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        # Excution self._post_init_method after the default init function.
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _enable_class(subclass)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

        self._pre_context_exec()

    def __exit__(self, exc_type, exc_value, traceback):

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = (torch.nn.modules.module.Module._old_init_subclass)

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _pre_context_exec(self):
        pass

    def _post_context_exec(self):
        pass


class ZeroInitContext(InsertPostInitMethodToModuleSubClasses):
    r"""
    A context to initialize model.
    1. Convert the model to fp16.
    2. The paramaters of the module are adapted to type ShardedParameter.
    3. Shard the param and grad according to flags.

    target_device: the device where param data after exiting the context
    shard_strategy: shard strategy instance
    shard_param: is param sharded after exiting the context
    shard_grad: is param sharded after exiting the context

    rm_torch_payload_on_the_fly:
    True: remove tensor payload on param.data after module init finished.
    False: remove tensor payload on param.data afther the context exist.
            This is used when you add some logic to operate tensors in __init__ of module.
            See torchvision resnet18.
    """

    def __init__(self,
                 convert_fp16: bool,
                 target_device: torch.device,
                 shard_strategy: BaseShardStrategy,
                 shard_param: bool = False,
                 shard_grad: bool = False,
                 rm_torch_payload_on_the_fly: bool = False,
                 model_numel_tensor: torch.Tensor = torch.zeros(1, dtype=torch.int),
                 dp_process_group: Optional[ProcessGroup] = None):
        super().__init__()
        self.convert_fp16 = convert_fp16
        self.target_device = target_device
        self.shard_param = shard_param
        self.shard_grad = shard_grad
        self.shard_strategy = shard_strategy
        # FIXME(jiaruifang) now setting it to True is invalid.
        self.rm_torch_payload_on_the_fly = False
        self.initialized_param_list = []
        self.model_numel_tensor = model_numel_tensor
        self.dp_process_group = dp_process_group or gpc.get_group(ParallelMode.DATA)

    def _post_context_exec(self):
        """The callback function when the context exits.
        """
        if not self.rm_torch_payload_on_the_fly:
            for param in self.initialized_param_list:
                assert hasattr(param, 'col_attr')
                param.col_attr.remove_torch_payload()

            del self.initialized_param_list

    def _post_init_method(self, module):
        r"""The function to call at the end of the constructor of each nn.Module.
        """
        for param in module.parameters():
            # avoid adapting a param to ShardedParam twice
            if hasattr(param, 'col_attr'):
                continue

            self.model_numel_tensor += param.numel()

            target_device = self.target_device

            # convert to fp16 if necessary
            if self.convert_fp16:
                param.data = param.data.to(torch.half)
                if param.grad is not None:
                    param.grad = param.grad.to(torch.half)

            # move torch parameters to the target device
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)

            param.col_attr = ShardedParamV2(param, rm_torch_payload=self.rm_torch_payload_on_the_fly)

            self.initialized_param_list.append(param)

            if self.shard_param:
                self.shard_strategy.shard([param.col_attr.sharded_data_tensor], self.dp_process_group)
                GLOBAL_MODEL_DATA_TRACER.add_tensor(param.col_attr.sharded_data_tensor.payload)
            # if param.col_attr.grad and self.shard_grad:
            #     self.shard_strategy.shard([param.col_attr._grad_sharded_tensor], self.dp_process_group)
            #     GLOBAL_MODEL_DATA_TRACER.add_tensor(param.col_attr._grad_sharded_tensor.payload)
        # We must cast buffers
        # If we use BN, buffers may be on CPU and Float
        # We must cast them
        for buffer in module.buffers():
            buffer.data = buffer.data.to(device=torch.cuda.current_device())
            if self.convert_fp16:
                buffer.data = cast_tensor_to_fp16(buffer.data)
