from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from colossalai.interface import ModelWrapper
from colossalai.logging import get_dist_logger

from .utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, has_index_file

__all__ = ["CheckpointIO"]


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
    def __init__(self):
        super().__init__()
        self.pinned_state_dicts: Dict[int, dict] = {}
        self.async_writers = []

    def _sync_io(self):
        for writer in self.async_writers:
            writer.synchronize()
        self.async_writers.clear()

    def _sync_d2h(self):
        for writer in self.async_writers:
            writer.sync_before_step()

    def synchronize(self):
        """This method must be called before updating the model weights."""
        self._sync_d2h()

    def __del__(self):
        self._sync_d2h()
        self._sync_io()

    def load_model(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        strict: bool = True,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ) -> Union[nn.Module, ModelWrapper]:
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
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """
        # since we only support loaded sharded and unsharded weight format
        # containing no distributed tensors, dtensor -> full tensor conversion
        # should be done offline via our CLI
        # the existence of index file means it is a sharded checkpoint
        index_file_exists, index_file_path = has_index_file(checkpoint)

        # return the origin model instead of the unwrapped model
        origin_model = model

        if index_file_exists:
            self.load_sharded_model(
                model, index_file_path, strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
            )
        else:
            path = Path(checkpoint, SAFE_WEIGHTS_NAME)
            if path.is_file():
                self.load_unsharded_model(
                    model, str(path), strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
                )
            else:
                path = Path(checkpoint, WEIGHTS_NAME)
                if path.is_file():
                    self.load_unsharded_model(
                        model, str(path), strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
                    )
                else:
                    self.load_unsharded_model(
                        model, checkpoint, strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
                    )

        return origin_model

    def save_model(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: str = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
        use_async: bool = False,
    ):
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
        self._sync_io()
        if use_async and not use_safetensors:
            logger = get_dist_logger()
            logger.warning(
                "Async save is only supported when use_safetensors is set to True. "
                "Setting use_safetensors to True for async save."
            )
            use_safetensors = True

        if shard:
            self.save_sharded_model(
                model, checkpoint, gather_dtensor, prefix, size_per_shard, use_safetensors, use_async
            )
        else:
            self.save_unsharded_model(model, checkpoint, gather_dtensor, use_safetensors, use_async)

    def load_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        prefix: str = None,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        """
        Load optimizer from checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. This value is made compatibility with the model checkpoints in the
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """

        index_file_exists, index_file_path = has_index_file(checkpoint)

        if Path(checkpoint).is_dir() and not index_file_exists:
            # if the checkpoint is a directory and there is no index file, raise error
            raise ValueError(f"Cannot find index file in {checkpoint}")

        if index_file_exists:
            # the existence of index file means it is a sharded checkpoint
            self.load_sharded_optimizer(
                optimizer, index_file_path, prefix, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
            )
        else:
            self.load_unsharded_optimizer(
                optimizer, checkpoint, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
            )

    def save_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        shard: bool = False,
        gather_dtensor=True,
        prefix: str = None,
        size_per_shard: int = 1024,
        use_async: bool = False,
    ):
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
            self.save_sharded_optimizer(
                optimizer, checkpoint, gather_dtensor, prefix, size_per_shard, use_async=use_async
            )
        else:
            self.save_unsharded_optimizer(optimizer, checkpoint, gather_dtensor, use_async=use_async)

    # ========================================================
    # Abstract methods for model loading/saving implementation
    # ========================================================
    @abstractmethod
    def load_sharded_model(
        self, model: nn.Module, index_file_path: str, strict: bool, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        """
        Load model from sharded checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            index_file_path (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """

    @abstractmethod
    def load_unsharded_model(
        self, model: nn.Module, checkpoint: str, strict: bool, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        """
        Load model from unsharded checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """

    @abstractmethod
    def save_sharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool,
        prefix: Optional[str],
        size_per_shard: int,
        use_safetensors: bool,
        use_async: bool = False,
    ):
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

    @abstractmethod
    def save_unsharded_model(
        self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool, use_async: bool = False
    ):
        """
        Save model to unsharded checkpoint.

        Args:
            model (nn.Module): model to be saved.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
            use_safetensors (bool): whether to use safe tensors.
        """

    # ========================================================
    # Abstract methods for optimizer loading/saving implementation
    # ========================================================

    @abstractmethod
    def load_sharded_optimizer(
        self,
        optimizer: Optimizer,
        index_file_path: str,
        prefix: str,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        """
        Load optimizer from sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            index_file_path (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
            prefix (str): prefix for the optimizer checkpoint.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """

    @abstractmethod
    def load_unsharded_optimizer(
        self, optimizer: Optimizer, checkpoint: Path, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        """
        Load optimizer from unsharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """

    @abstractmethod
    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
        use_async: bool = False,
    ):
        """
        Save optimizer to sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (Path): checkpoint path. It should be a directory path.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
            prefix (str): prefix for the optimizer checkpoint.
            size_per_shard (int): size per shard in MB.
        """

    @abstractmethod
    def save_unsharded_optimizer(
        self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool, use_async: bool = False
    ):
        """
        Save optimizer to unsharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device.
        """

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

    # ================================================================================
    # Abstract method for lora saving implementation.
    # ================================================================================

    @abstractmethod
    def save_lora_as_pretrained(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        use_safetensors: bool = False,
        state_dict: Optional[dict] = None,
    ) -> None:
        """
        Save the lora adapters and adapter configuration file to a pretrained checkpoint directory.

        Args:
            model (Union[nn.Module, ModelWrapper]): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint directory. It must be a local path.
            use_safetensors (bool, optional): Whether to use safe tensors when saving. Defaults to False.
            state_dict (Optional[dict], optional): The state dict to save. Defaults to None.
        """
