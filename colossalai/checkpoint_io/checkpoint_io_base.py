from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from colossalai.interface import ModelWrapper

from .utils import has_index_file

__all__ = ['CheckpointIO']


class CheckpointIO(ABC):
    """
    CheckpointIO is the base class for all checkpoint IO classes. It defines the interface for checkpoint IO.


    Examples:
        >>> from colossalai.checkpoint_io import GeneralCheckpointIO
        >>> checkpoint_io = CheckpointIO()
        >>>
        >>> # load model from checkpoint
        >>> model = checkpoint_io.load_model(model, 'model.pt')
        >>>
        >>> # save model to checkpoint, any distributed tensor is gathered by default
        >>> checkpoint_io.save_model(model, 'model.pt')
        >>>
        >>> # if the model contains distributed tensor, and you don't want to gather it
        >>> # each rank will save its own shard of the distributed tensor
        >>> checkpoint_io.save_model(model, 'model.pt', gather_dtensor=False)
        >>>
        >>> # save model to sharded checkpoints
        >>> checkpoint_io.save_model(model, './checkpoints/', shard=True)
        >>>
        >>> # save model to sharded  and assume we don't want to gather distributed tensors
        >>> checkpoint_io.save_model(model, './checkpoints/', shard=True, gather_dtensor=False)
        >>>
        >>> # Note:
        >>> # 1. we don't support loading from distributed tensors, conversion from distributed tensors
        >>> # checkpoints to full tensor checkpoint should be done offline via our CLI
        >>> # 2. you don't have to specify whether the model is sharded or not when loading the model
        >>> # as it will be automatically detected
        >>>
        >>> # load model from sharded checkpoints
        >>> model = checkpoint_io.load_model(model, './checkpoints/')
        >>>
        >>> # load model from unsharded checkpoints
        >>> model = checkpoint_io.load_model(model, './checkpoints/')
        >>>
        >>> # load optimizer from checkpoint
        >>> optimizer = checkpoint_io.load_optimizer(optimizer, 'optimizer.pt')
        >>>
        >>> # save optimizer to checkpoint
        >>> checkpoint_io.save_optimizer(optimizer, 'optimizer.pt')
    """

    # ======================================
    # Public methods
    # ======================================
    def load_model(self,
                   model: Union[nn.Module, ModelWrapper],
                   checkpoint: str,
                   strict: bool = True) -> Union[nn.Module, ModelWrapper]:
        """
        Load model from checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            checkpoint (str): checkpoint path. This value is made compatibility with the model checkpoints in the
                        mainstream model zoos such as Hugging Face and TIMM. The checkpoint path can be:
                        1. a file path, e.g. 'model.pt'
                        2. a path to a json file which defines the index to the sharded checkpoint
                        3. a path to a folder containing a unique .index.json file for sharded checkpoint
                        Distributed tensors cannot be loaded directly unless gathered offline via our CLI.
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
        """
        # since we only support loaded sharded and unsharded weight format
        # containing no distributed tensors, dtensor -> full tensor conversion
        # should be done offline via our CLI
        # the existence of index file means it is a sharded checkpoint
        index_file_exists, index_file_path = has_index_file(checkpoint)

        # return the origin model instead of the unwrapped model
        origin_model = model

        if isinstance(model, ModelWrapper):
            model = model.unwrap()

        if index_file_exists:
            self.load_sharded_model(model, index_file_path, strict)
        else:
            self.load_unsharded_model(model, checkpoint, strict)

        return origin_model

    def save_model(self,
                   model: Union[nn.Module, ModelWrapper],
                   checkpoint: str,
                   shard: bool = False,
                   gather_dtensor: bool = True,
                   prefix: str = None,
                   size_per_shard: int = 1024,
                   use_safetensors: bool = False):
        """
        Save model to checkpoint.

        Examples:
            >>> from colossalai.checkpoint_io import GeneralCheckpointIO
            >>> checkpoint_io = CheckpointIO()
            >>>
            >>> # save model to a single file
            >>> save_model(model, 'model.pt')
            >>>
            >>> # save model to a sharded checkpoint
            >>> save_model(model, './checkpoints/', shard=True)

        Args:
            model (nn.Module): model to be saved.
            checkpoint (str): checkpoint path. The checkpoint path can be :
                1. a file path, e.g. 'model.pt'
                2. a directory path to save the sharded checkpoint, e.g. './checkpoints/' when shard = True.
            shard (bool): whether to shard the checkpoint. Default: False. If set to True, the checkpoint will be sharded into
                multiple files. The model shards will be specified by a `model.index.json` file. When shard = True, please ensure
                that the checkpoint path is a directory path instead of a file path.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str): If specified, weights are saved in the format pytorch_model.<prefix>.bin. Default: None.
            size_per_shard (int): size per shard in MB. Default: 1024. This value is only used when shard = True.
            use_safetensors (bool): whether to use safe tensors. Default: False. If set to True, the checkpoint will be saved
        """

        if isinstance(model, ModelWrapper):
            model = model.unwrap()

        if shard:
            self.save_sharded_model(model, checkpoint, gather_dtensor, prefix, size_per_shard, use_safetensors)
        else:
            self.save_unsharded_model(model, checkpoint, gather_dtensor, use_safetensors)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str, prefix: str = None, size_per_shard: int = 1024):
        """
        Load optimizer from checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. This value is made compatibility with the model checkpoints in the
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            size_per_shard (int, optional): Maximum size of checkpoint shard file in MB. This is useful only when ``shard=True``. Defaults to 1024.
        """

        index_file_exists, index_file_path = has_index_file(checkpoint)

        if Path(checkpoint).is_dir() and not index_file_exists:
            # if the checkpoint is a directory and there is no index file, raise error
            raise ValueError(f'Cannot find index file in {checkpoint}')

        if index_file_exists:
            # the existence of index file means it is a sharded checkpoint
            self.load_sharded_optimizer(optimizer, index_file_path, prefix)
        else:
            self.load_unsharded_optimizer(optimizer, checkpoint)

    def save_optimizer(self,
                       optimizer: Optimizer,
                       checkpoint: str,
                       shard: bool = False,
                       gather_dtensor=True,
                       prefix: str = None,
                       size_per_shard: int = 1024):
        """
        Save optimizer to checkpoint. Optimizer states saving is not compatible with safetensors.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (str): checkpoint path. The checkpoint path can be :
                1. a file path, e.g. 'model.pt'
                2. a path to a json file which defines the index to the sharded checkpoint for the optimizer
                3. a path to a folder containing a unique .index.json file for sharded checkpoint
            shard (bool): whether to shard the checkpoint. Default: False. If set to True, the checkpoint will be sharded into
                multiple files. The optimizer shards will be specified by a `optimizer.index.json` file.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str): prefix for the optimizer checkpoint when shard = True. Default: None.
            size_per_shard (int): size per shard in MB. Default: 1024. This value is only used when shard is set to True.
        """

        if shard:
            self.save_sharded_optimizer(optimizer, checkpoint, gather_dtensor, prefix, size_per_shard)
        else:
            self.save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor)

    # ========================================================
    # Abstract methods for model loading/saving implementation
    # ========================================================
    @abstractmethod
    def load_sharded_model(self, model: nn.Module, index_file_path: str, strict: bool):
        """
        Load model from sharded checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            index_file_path (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
        """
        pass

    @abstractmethod
    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        """
        Load model from unsharded checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
        """
        pass

    @abstractmethod
    def save_sharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, prefix: Optional[str],
                           size_per_shard: int, use_safetensors: bool):
        """
        Save model to sharded checkpoint.

        Args:
            model (nn.Module): model to be saved.
            checkpoint (str): checkpoint path. It should be a directory path.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
            prefix (str): prefix for the model checkpoint.
            size_per_shard (int): size per shard in MB.
            use_safetensors (bool): whether to use safe tensors.
        """
        pass

    @abstractmethod
    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save model to unsharded checkpoint.

        Args:
            model (nn.Module): model to be saved.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
            use_safetensors (bool): whether to use safe tensors.
        """
        pass

    # ========================================================
    # Abstract methods for optimizer loading/saving implementation
    # ========================================================

    @abstractmethod
    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, prefix: str):
        """
        Load optimizer from sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            index_file_path (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
            prefix (str): prefix for the optimizer checkpoint.
        """
        pass

    @abstractmethod
    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        """
        Load optimizer from unsharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
        """
        pass

    @abstractmethod
    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool, prefix: str,
                               size_per_shard: int):
        """
        Save optimizer to sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (Path): checkpoint path. It should be a directory path.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
            prefix (str): prefix for the optimizer checkpoint.
            size_per_shard (int): size per shard in MB.
        """
        pass

    @abstractmethod
    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool):
        """
        Save optimizer to unsharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
        """
        pass

    # ============================================
    # methods for loading and saving lr scheduler
    # as this is quite standard, there is no need
    # to make them abstract
    # ============================================
    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save lr scheduler to checkpoint.

        Args:
            lr_scheduler (LRScheduler): lr scheduler to be saved.
            checkpoint: checkpoint path. The checkpoint path can only be a file path.
        """
        torch.save(lr_scheduler.state_dict(), checkpoint)

    def load_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Load lr scheduler from checkpoint.

        Args:
            lr_scheduler (LRScheduler): lr scheduler to be loaded.
            checkpoint (str): the path for a single checkpoint file.
        """
        state_dict = torch.load(checkpoint)
        lr_scheduler.load_state_dict(state_dict)
