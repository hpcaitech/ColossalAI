import torch
import torch.nn as nn
from torch.optim import Optimizer
from colossalai.amp.naive_amp import NaiveAMPModel
from colossalai.utils import is_no_pp_or_last_stage
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode

from .zero_redundancy_optimizer_level_2 import ZeroRedundancyOptimizer_Level_2
from .zero_redundancy_optimizer_level_3 import ZeroRedundancyOptimizer_Level_3


def convert_to_zero(model: nn.Module,
                    optimizer: Optimizer,
                    level: int,
                    zero_config):
    """
    A helper function to integrate the model and optimizer with ZeRO optimizer and off-loading

    :param model: your model object
    :type model: :class:`torch.nn.Module`
    :param optimizer: your optimizer object
    :type optimizer: :class:`torch.optim.Optimizer`
    :param level: optimizer level, can be 2 or 3
    :type level: int
    :param zero_config: configuration for zero
    :type zero_config: dict

    :return: (model, optimizer)
    :rtype: Tuple
    """
    import deepspeed
    assert level == 2 or level == 3, 'Only ZERO Optimizer Level 2 and 3 are provided'
    model = NaiveAMPModel(model, output_to_fp32=False)

    if level == 2:
        optimizer = ZeroRedundancyOptimizer_Level_2(init_optimizer=optimizer, **zero_config)
    else:
        optimizer = ZeroRedundancyOptimizer_Level_3(init_optimizer=optimizer, module=model, **zero_config)
    return model, optimizer


def zero3_model_context(dtype=torch.half):
    """A context to enable massive model construction for training with
        ZeRO-3. Models are automatically partitioned (or, sharded) across the
        system and converted to half precision. Note that the config of ZeRO-3 will be loaded automatically from `gpc.config`.

        Args:
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``torch.half``

        This context accelerates model initialization and enables models that
        are too large to allocate in their entirety in CPU memory. It has the
        following effects:

        #. allocates tensors to either GPU or CPU memory or NVMe
        #. converts floating point tensors to half precision
        #. immediately partitions tensors among the group of data-parallel devices
        #. (*optional*) replaces ``torch.nn.functional.linear`` with a more
           memory-efficient implementation

        These modifications allow for models that exceed the size of local CPU/GPU
        memory/NVMe, but fit within the total NVMe capacity (*i.e.*, aggregate CPU
        or GPU memory or NVMe) across all nodes. Consider initializing a model with one
        trillion parameters, whose weights occupy two terabytes (TB) in half
        precision. The initial CPU allocation in full precision requires 4TB of
        memory *per process*, and so a system with 8 GPUs per node would need 32TB of
        CPU memory due to data-parallel redundancies. Instead, by immediately
        partitioning tensors we remove the redundancies. The result is that
        regardless of the number of GPUs, we still only require the original 4TB. This
        allows for a linear increase in model size with the aggregate system memory.
        For example, if a node has 1TB of memory and 8 GPUs, we could fit a trillion
        parameter model with 4 nodes and 32 GPUs.

        Important: If the fp16 weights of the model can't fit onto a single GPU memory
        this feature must be used.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python

                with zero3_model_context():
                    model = MyLargeModel()

    """
    assert dtype == torch.half or dtype == torch.float, f'Invalid dtype, except torch.half or torch.float, got {dtype}'
    import deepspeed
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "offload_param": getattr(gpc.config.zero, 'offload_param_config', None),
            "offload_optimizer": getattr(gpc.config.zero, 'offload_optimizer_config'),
        },
        "aio": getattr(gpc.config.zero, 'aio_config', None)
    }
    remote_device = getattr(ds_config['zero_optimization']['offload_param'], 'device', None)
    pin_memory = getattr(ds_config['zero_optimization']['offload_param'], 'pin_memory', False)
    return deepspeed.zero.Init(data_parallel_group=gpc.get_group(ParallelMode.DATA),
                               remote_device=remote_device,
                               config_dict_or_path=ds_config,
                               pin_memory=pin_memory,
                               dtype=dtype)


__all__ = ['convert_to_zero', 'ZeroRedundancyOptimizer_Level_2',
           'ZeroRedundancyOptimizer_Level_3', 'zero3_model_context']
