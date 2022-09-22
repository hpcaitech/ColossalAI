import contextlib
import functools
from typing import Optional
from contextlib import AbstractContextManager

import torch
import torch.nn as nn
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.logging import get_dist_logger
from colossalai.zero.shard_utils import BaseShardStrategy
from colossalai.zero.sharded_model._utils import cast_tensor_to_fp16
from colossalai.zero.sharded_model.sharded_model_v2 import ShardedModelV2
from colossalai.zero.sharded_param import ShardedParamV2
from colossalai.utils.model.utils import InsertPostInitMethodToModuleSubClasses


class ZeroContextConfig(object):
    """The configuration used to control zero context initialization.

    Args:
        target_device (torch.device): The device where param data are after exiting the context.
        replicated (bool, optional): Whether the param is replicated across data parallel group.
            Some parameters are not replicated, e.g. parameters in MOE experts.
        shard_param (bool, optional): Is param sharded after exiting the context. Defaults to False.
    """

    def __init__(self, target_device: torch.device, replicated: bool = True, shard_param: bool = False):
        super().__init__()

        if shard_param:
            assert replicated, "Non-replicated parameters can't be sharded."

        # replicated no-shard parameters should locate in cuda, since we will broadcast them soon
        if replicated and not shard_param:
            assert target_device.type == 'cuda', "Replicated no-shard paramters should locate in cuda."

        self.target_device = target_device
        self.is_replicated: bool = replicated
        self.shard_param: bool = shard_param


class ZeroInitContext(InsertPostInitMethodToModuleSubClasses):
    """A context to initialize model.

    1. Convert the model to fp16.
    2. The paramaters of the module are adapted to type ShardedParameter.
    3. Shard the param and grad according to flags.

    Args:
        target_device (torch.device): The device where param data are after exiting the context.
        shard_strategy (BaseShardStrategy): Shard strategy instance.
        seed (int, optional): Random seed for weight initialization
        shard_param (bool, optional): Is param sharded after exiting the context. Defaults to False.
        default_dtype (torch.dtype, optional): If it's not None, parameters will be initialized as ``default_dtype`` then converted to fp16.
        model_numel_tensor (torch.Tensor, optional): A tensor which will store the number of elements of model. Defaults to torch.zeros(1, dtype=torch.int).
    """

    def __init__(self,
                 target_device: torch.device,
                 shard_strategy: BaseShardStrategy,
                 seed: int = 2**10 - 1,
                 shard_param: bool = False,
                 default_dtype: Optional[torch.dtype] = None,
                 model_numel_tensor: torch.Tensor = torch.zeros(1, dtype=torch.long)):

        super().__init__(default_dtype=default_dtype)
        self.shard_strategy = shard_strategy
        self.param_list = []
        self.model_numel_tensor = model_numel_tensor
        self.seed = seed
        self.dp_process_group = gpc.get_group(ParallelMode.DATA)

        self.config = ZeroContextConfig(target_device=target_device, replicated=True, shard_param=shard_param)

        ZeroContextMgr().current_context = self

        self.param_numel = {}
        self.top_module = None

    @property
    def target_device(self):
        return self.config.target_device

    @property
    def is_replicated(self):
        return self.config.is_replicated

    @property
    def shard_param(self):
        return self.config.shard_param

    @staticmethod
    def calc_fanin_fanout(tensor: torch.Tensor):
        """We use this function to substitute fan-in and fan-out calculation in torch.nn.init.
        This can help us get correct fan-in and fan-out for sharded tensor.
        """
        assert isinstance(tensor, nn.Parameter), "Sharded tensor initilization is only allowed for paramters"

        # get correct shape of input tensor
        if not hasattr(tensor, 'colo_attr') or not tensor.colo_attr.param_is_sharded:
            tensor_shape = tensor.shape
        else:
            tensor_shape = tensor.colo_attr.sharded_data_tensor.origin_shape

        dimensions = len(tensor_shape)
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor_shape[1]
        num_output_fmaps = tensor_shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            # math.prod is not always available, accumulate the product manually
            # we could use functools.reduce but that is not supported by TorchScript
            for s in tensor_shape[2:]:
                receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _pre_context_exec(self):
        """ 
        The Callback function when entering the context
        """
        self.logger = get_dist_logger("ZeroInitContext")

        # substitute fan-in and fan-out calculation
        self.nn_fanin_fanout = nn.init._calculate_fan_in_and_fan_out
        nn.init._calculate_fan_in_and_fan_out = self.calc_fanin_fanout

        self.module_load_from_state_dict = nn.Module._load_from_state_dict
        shard_strategy = self.shard_strategy if self.config.shard_param else None
        nn.Module._load_from_state_dict = functools.partialmethod(ShardedModelV2._colo_load_from_state_dict,
                                                                  shard_strategy=shard_strategy)
        self.module_state_dict = nn.Module.state_dict
        nn.Module.state_dict = functools.partialmethod(ShardedModelV2._colo_state_dict,
                                                       shard_strategy=shard_strategy,
                                                       state_dict_func=self.module_state_dict,
                                                       process_group=self.dp_process_group)

        # reserve rng states
        self.cpu_rng_state = torch.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state()

        # set new seed for initialization, since we initialize sharded tensor separately
        # we don't want all processes have the same seed
        # otherwise all sharded tensors are same after init
        offset = self.seed + 1    # we want to have more 1 in binary format seed
        torch.manual_seed(self.seed + offset * dist.get_rank())

    def _post_context_exec(self):
        """The callback function when exiting context.
        """
        # broadcast replicated no-shard parameters
        src_rank = gpc.get_ranks_in_group(ParallelMode.DATA)[0]
        for param in self.param_list:
            assert hasattr(param, 'colo_attr')
            if not param.colo_attr.param_is_sharded and param.colo_attr.is_replicated:
                dist.broadcast(tensor=param.data, src=src_rank, group=self.dp_process_group)
            param.colo_attr.set_data_none()

        del self.param_list

        nn.init._calculate_fan_in_and_fan_out = self.nn_fanin_fanout
        nn.Module.load_state_dict = self.module_load_from_state_dict
        nn.Module.state_dict = self.module_state_dict
        torch.set_rng_state(self.cpu_rng_state)
        torch.cuda.set_rng_state(self.cuda_rng_state)

        params = frozenset(self.top_module.parameters())
        for param in self.param_numel.keys():
            if param not in params:
                self.param_numel[param] = 0
        self.model_numel_tensor.fill_(sum(self.param_numel.values()))

    def _post_init_method(self, module: torch.nn.Module, *args, **kwargs):
        """
        The function to call at the end of the constructor of each module.
        NOTE() The module may be passed to this function multiple times.
        """
        self.top_module = module

        def half_fn(t: torch.Tensor):
            return t.half() if t.is_floating_point() else t

        for param in module.parameters(recurse=False):
            # avoid adapting a param to ShardedParam twice
            if hasattr(param, 'colo_attr'):
                continue

            self.param_numel[param] = param.numel()

            # convert parameters to half
            param_half = half_fn(param)
            param.data = param_half
            if param.grad is not None:
                grad_half = half_fn(param.grad)
                param.grad.data = grad_half

            # move torch parameters to the target device
            target_device = self.target_device
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)

            param.colo_attr = ShardedParamV2(param, set_data_none=True)

            if self.shard_param:
                self.shard_strategy.shard([param.colo_attr.sharded_data_tensor], self.dp_process_group)

            param.data = param.colo_attr.data_payload    # set param.data to payload

            # mark whether the param is replicated
            param.colo_attr.is_replicated = self.is_replicated

            # mark whether the param should keep not sharded
            # if True, the param is used as Zero stage 2
            param.colo_attr.keep_not_shard = not self.shard_param

            self.param_list.append(param)

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
                                                  shard_param=False)


def no_shard_zero_decrator(is_replicated: bool = True):

    def _wrapper(init_func):

        def _no_shard(*args, **kwargs):
            with no_shard_zero_context(is_replicated):
                ret = init_func(*args, **kwargs)
            return ret

        return _no_shard

    return _wrapper
