import contextlib
import functools
from typing import Optional

import torch
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.logging import get_dist_logger
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model._utils import cast_tensor_to_fp16
from colossalai.zero.sharded_param import ShardedParamV2
from torch.distributed import ProcessGroup
from contextlib import AbstractContextManager


def _substitute_init_recursively(cls, func):
    for subcls in cls.__subclasses__():
        _substitute_init_recursively(subcls, func)
        func(subcls)


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
        _substitute_init_recursively(torch.nn.modules.module.Module, _enable_class)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (torch.nn.modules.module.Module.__init_subclass__)
        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

        self._pre_context_exec()

    def __exit__(self, exc_type, exc_value, traceback):

        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        _substitute_init_recursively(torch.nn.modules.module.Module, _disable_class)

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


class ZeroContextConfig(object):
    """The configuration used to control zero context initialization.

    Args:
        target_device (torch.device): The device where param data are after exiting the context.
        replicated (bool, optional): Whether the param is replicated across data parallel (DP) group.
            We do not need to synchronize (reduce) the grads of the replicated params among DP group.
            Some parameters are not replicated, e.g. parameters in MOE experts.
        shard_param (bool, optional): Is param sharded after exiting the context. Defaults to False.
            The process group among which tensors are sharded is assigned as an runtime arg.
        rm_torch_payload_on_the_fly (bool, optional): If set to `True`, remove tensor payload on `param.data` after module init finished.
            This will reduce memory usage when initializing model.
            But it's not suitable for all models, especially when there are `weight init` operations in `__init__`.
            If set to `False`, remove tensor payload on param.data afther the context exist.
            This is used when you add some logic to operate tensors in __init__ of module.
            See torchvision resnet18. Defaults to False.
    """

    def __init__(self,
                 target_device: torch.device,
                 replicated: bool = True,
                 shard_param: bool = False,
                 rm_torch_payload_on_the_fly: bool = False):
        super().__init__()
        self.target_device = target_device
        self.is_replicated: bool = replicated
        self.shard_param: bool = shard_param

        if self.is_replicated is False:
            assert self.shard_param is True, f"ZeroContextConfig shard_param must be False when is_replicated is False"
        self.rm_torch_payload_on_the_fly: bool = rm_torch_payload_on_the_fly


class ZeroInitContext(InsertPostInitMethodToModuleSubClasses):
    """A context to initialize model.

    1. Convert the model to fp16.
    2. The paramaters of the module are adapted to type `ShardedParameter`.
    3. Shard the param and grad according to flag `shard_param`.

    Args:
        target_device (torch.device): The device where param data are after exiting the context.
        shard_strategy (BaseShardStrategy): Shard strategy instance.
        shard_param (bool, optional): Is param sharded after exiting the context. Defaults to False.
        rm_torch_payload_on_the_fly (bool, optional): If set to `True`, remove tensor payload on `param.data` after module init finished.
            This will reduce memory usage when initializing model. 
            But it's not suitable for all models, especially when there are `weight init` operations in `__init__`.
            If set to `False`, remove tensor payload on param.data afther the context exist.
            This is used when you add some logic to operate tensors in __init__ of module.
            See torchvision resnet18. Defaults to False.
        model_numel_tensor (torch.Tensor, optional): A tensor which will store the number of elements of model. Defaults to torch.zeros(1, dtype=torch.int).
        dp_process_group (Optional[ProcessGroup], optional): Data parallel process group. Defaults to None.
    """

    def __init__(self,
                 target_device: torch.device,
                 shard_strategy: BaseShardStrategy,
                 shard_param: bool = False,
                 rm_torch_payload_on_the_fly: bool = False,
                 model_numel_tensor: torch.Tensor = torch.zeros(1, dtype=torch.long),
                 dp_process_group: Optional[ProcessGroup] = None):

        super().__init__()
        self.shard_strategy = shard_strategy
        # a list contains params that could be sharded.
        self.shardable_param_list = []
        self.model_numel_tensor = model_numel_tensor
        self.dp_process_group = dp_process_group or gpc.get_group(ParallelMode.DATA)

        self.config = ZeroContextConfig(target_device=target_device,
                                        replicated=True,
                                        shard_param=shard_param,
                                        rm_torch_payload_on_the_fly=rm_torch_payload_on_the_fly)

        ZeroContextMgr().current_context = self

    @property
    def target_device(self):
        return self.config.target_device

    @property
    def is_replicated(self):
        return self.config.is_replicated

    @property
    def shard_param(self):
        return self.config.shard_param

    @property
    def rm_torch_payload_on_the_fly(self):
        return self.config.rm_torch_payload_on_the_fly

    def _pre_context_exec(self):
        """ 
        The Callback function when entering the context
        """
        self.logger = get_dist_logger("ZeroInitContext")

    def _post_context_exec(self):
        """The callback function when exiting context.
        """
        if not self.rm_torch_payload_on_the_fly:
            for param in self.shardable_param_list:
                assert hasattr(param, 'colo_attr')
                param.colo_attr.remove_torch_payload()

            del self.shardable_param_list

    def _post_init_method(self, module: torch.nn.Module):
        """
        The function to call at the end of the constructor of each module.
        NOTE() The module may be passed to this function multiple times.
        """
        for param in module.parameters(recurse=False):
            # avoid adapting a param to ShardedParam twice
            if hasattr(param, 'colo_attr'):
                continue

            self.model_numel_tensor += param.numel()

            # mark whether the param is replicated
            param.is_replicated = self.is_replicated

            # convert parameters to half
            param_half = cast_tensor_to_fp16(param.data)
            param.data = param_half
            if param.grad is not None:
                grad_half = cast_tensor_to_fp16(param.grad)
                param.grad.data = grad_half

            # move torch parameters to the target device
            target_device = self.target_device
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)

            param.colo_attr = ShardedParamV2(param, rm_torch_payload=self.rm_torch_payload_on_the_fly)

            if self.shard_param:
                self.shard_strategy.shard([param.colo_attr.sharded_data_tensor], self.dp_process_group)
                self.shardable_param_list.append(param)

        # We must cast buffers
        # If we use BN, buffers may be on CPU and Float
        # We must cast them
        for buffer in module.buffers(recurse=False):
            buffer.data = buffer.data.to(device=torch.cuda.current_device())
            buffer.data = cast_tensor_to_fp16(buffer.data)


class ZeroContextMgr(metaclass=SingletonMeta):
    current_context: Optional[ZeroInitContext] = None

    @contextlib.contextmanager
    def hijack_context_config(self, **kwargs):
        if self.current_context is None:
            yield
        else:
            old_config = self.current_context.config
            self.current_context.config = ZeroContextConfig(**kwargs)
            yield
            self.current_context.config = old_config


def no_shard_zero_context(is_replicated: bool = True) -> AbstractContextManager:
    return ZeroContextMgr().hijack_context_config(target_device=torch.device('cuda', torch.cuda.current_device()),
                                                  replicated=is_replicated,
                                                  shard_param=False,
                                                  rm_torch_payload_on_the_fly=False)


def no_shard_zero_decrator(is_replicated: bool = True):
    """
    A decorator used to wrap an __init__ function of Module.
    The parameters initialized by the model will not sharded.
    is_replicated indicates the grad of the param won't be reduced among the data parallel process group.

    >>> def MyModule(torch.nn.Module):
    >>>     @no_shard_zero_decrator(is_replicated = False)
    >>>     def __init__(self, ...)
    >>>         ....
    """

    def _wrapper(init_func):

        def _no_shard(*args, **kwargs):
            with no_shard_zero_context(is_replicated):
                init_func(*args, **kwargs)

        return _no_shard

    return _wrapper
