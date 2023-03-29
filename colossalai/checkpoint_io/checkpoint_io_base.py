import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from colossalai.interface import ModelWrapper

__all__ = ['CheckpointIO', 'ShardCheckpointIndexFile']


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
        >>> # save model to checkpoint
        >>> checkpoint_io.save_model(model, 'model.pt')
        >>>
        >>> # save model to sharded checkpoints
        >>> checkpoint_io.save_model(model, './checkpoints/', shard=True)
        >>>
        >>> # load model from sharded checkpoints
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
            checkpoint (str): checkpoint path. This value is made compatiblity with the model checkpoints in the
                        mainstream model zoos such as Hugging Face and TIMM. The checkpoint path can be:
                        1. a file path, e.g. 'model.pt'
                        2. a path to a json file which defines the index to the sharded checkpoint
                        3. a path to a folder containing a unique .index.json file for sharded checkpoint
            strict (bool): whether to strictly enforce that the param name in
                the checkpoint match the keys returned by this module's.
        """
        ckpt_path = Path(checkpoint)
        is_sharded = self.is_sharded_checkpoint(ckpt_path)

        origin_model = model

        if isinstance(model, ModelWrapper):
            model = model.unwrap()

        if is_sharded:
            self.load_sharded_model(model, ckpt_path, strict)
        else:
            self.load_unsharded_model(model, ckpt_path, strict)

        return origin_model

    def save_model(self,
                   model: Union[nn.Module, ModelWrapper],
                   checkpoint: str,
                   shard: bool = False,
                   prefix: str = None,
                   size_per_shard: int = 1024):
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
                multiple files. The model shards will be specificed by a `model.index.json` file. When shard = True, please ensure
                that the checkpoint path is a directory path instead of a file path.
            prefix (str): prefix for the model checkpoint file name when shard=True. Default: None.
            size_per_shard (int): size per shard in MB. Default: 1024. This value is only used when shard = True.
        """

        if isinstance(model, ModelWrapper):
            model = model.unwrap()

        if shard:
            self.save_sharded_model(model, checkpoint, prefix, size_per_shard)
        else:
            self.save_unsharded_model(model, checkpoint)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str):
        """
        Load optimizer from checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. This value is made compatiblity with the model checkpoints in the
        """
        ckpt_path = Path(checkpoint)
        is_sharded = self.is_sharded_checkpoint(ckpt_path)

        if is_sharded:
            self.load_sharded_optimizer(optimizer, ckpt_path)
        else:
            self.load_unsharded_optimizer(optimizer, ckpt_path)

    def save_optimizer(self,
                       optimizer: Optimizer,
                       checkpoint: str,
                       shard: bool = False,
                       prefix: str = None,
                       size_per_shard: int = 1024):
        """
        Save optimizer to checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (str): checkpoint path. The checkpoint path can be :
                1. a file path, e.g. 'model.pt'
                2. a path to a json file which defines the index to the sharded checkpoint for the optimizer
                3. a path to a folder containing a unique .index.json file for sharded checkpoint
            shard (bool): whether to shard the checkpoint. Default: False. If set to True, the checkpoint will be sharded into
                multiple files. The optimizer shards will be specificed by a `optimizer.index.json` file.
            prefix (str): prefix for the optimizer checkpoint when shard = True. Default: None.
            size_per_shard (int): size per shard in MB. Default: 1024. This value is only used when shard is set to True.
        """
        if shard:
            self.save_sharded_optimizer(optimizer, checkpoint, prefix, size_per_shard)
        else:
            self.save_unsharded_optimizer(optimizer, checkpoint)

    # ========================================================
    # Abstract methods for model loading/saving implementation
    # ========================================================
    @abstractmethod
    def load_sharded_model(self, model: nn.Module, checkpoint: Path, strict: bool):
        """
        Load model from sharded checkpoint.

        Args:
            model (nn.Module): model to be loaded.
            checkpoint (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
        """
        pass

    @abstractmethod
    def load_unsharded_model(self, model: nn.Module, checkpoint: Path, strict: bool):
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
    def save_sharded_model(self, model: nn.Module, checkpoint: Path, prefix: str, size_per_shard: int):
        """
        Save model to sharded checkpoint.

        Args:
            model (nn.Module): model to be saved.
            checkpoint (Path): checkpoint path. It should be a directory path.
            prefix (str): prefix for the model checkpoint.
            size_per_shard (int): size per shard in MB.
        """
        pass

    @abstractmethod
    def save_unsharded_model(self, model: nn.Module, checkpoint: Path):
        """
        Save model to unsharded checkpoint.

        Args:
            model (nn.Module): model to be saved.
            checkpoint (Path): checkpoint path. It should be a single file path pointing to a model weight binary.
        """
        pass

    # ========================================================
    # Abstract methods for optimizer loading/saving implementation
    # ========================================================

    @abstractmethod
    def load_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, prefix: str, size_per_shard: int):
        """
        Load optimizer from sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be loaded.
            checkpoint (str): checkpoint path. It should be path to the .index.json file or a path to a directory which contains a .index.json file.
            prefix (str): prefix for the optimizer checkpoint.
            size_per_shard (int): size per shard in MB.
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
    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, prefix: str, size_per_shard: int):
        """
        Save optimizer to sharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (Path): checkpoint path. It should be a directory path.
            prefix (str): prefix for the optimizer checkpoint.
            size_per_shard (int): size per shard in MB.
        """
        pass

    @abstractmethod
    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        """
        Save optimizer to unsharded checkpoint.

        Args:
            optimizer (Optimizer): optimizer to be saved.
            checkpoint (str): checkpoint path. It should be a single file path pointing to a model weight binary.
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

    # ========================================
    # Helper functions for loading state dict
    # ========================================

    def get_sharded_checkpoint_index_file(self, checkpoint_path: Path):
        """
        Get the index file path for a sharded checkpoint.

        Args:
            checkpoint_path (Path): path to the checkpoint.

        Returns:
            Path: path to the index file.
        """
        if checkpoint_path.is_file():
            # check if it is .index.json
            if checkpoint_path.name.endswith('.index.json'):
                return checkpoint_path
            else:
                raise ValueError(f'Invalid checkpoint path: {checkpoint_path}. ')
        elif checkpoint_path.is_dir():
            # check if there is only one a file ending with .index.json in this directory
            index_files = list(checkpoint_path.glob('*.index.json'))
            if len(index_files) == 1:
                return index_files[0]
            else:
                raise ValueError(f'Found {len(index_files)} index files in {checkpoint_path}. ')

    def is_sharded_checkpoint(self, checkpoint_path: Path):
        """
        Check whether the checkpoint is sharded.

        Args:
            checkpoint (str): checkpoint path.

        Returns:
            bool: whether the checkpoint is sharded.
        """
        if checkpoint_path.is_file():
            # check if it is .index.json
            if checkpoint_path.name.endswith('.index.json'):
                return True
            else:
                return False
        elif checkpoint_path.is_dir():
            # check if there is only one a file ending with .index.json in this directory
            index_files = list(checkpoint_path.glob('*.index.json'))
            if len(index_files) == 1:
                return True
            else:
                raise ValueError(f'Found {len(index_files)} index files in {checkpoint_path}. ')

    def get_checkpoint_shard_filenames(self, index_file_path: Path):
        """
        Get checkpoint shard filenames from a json file.

        Args:
            index_file_path (Path): path to the json file.

        Returns:
            list: checkpoint shard filenames.
        """
        with open(str(index_file_path), 'r') as f:
            shard_filenames = json.load(f)

        if "weight_map" in index:
            index = index["weight_map"]

        checkpoint_root_path = index_file_path.absolute().parent

        # read the checkpoint file list from the json file and get a list of unique file names
        checkpoint_files = sorted(list(set(index.values())))

        # get the absolute paths for all checkpoint files
        checkpoint_files = [checkpoint_root_path.joinpath(f) for f in checkpoint_files]
        return shard_filenames

    def load_safetensors_state_dict(self, *args, **kwargs):
        """
        Load safetensors state dict from checkpoint.
        """
        # TODO(FrankLeeeee): support huggingface safetensors
        raise NotImplementedError("This method is not implemented to support safe tensors")

    def load_state_dict(self, checkpoint_file_path: Path):
        """
        Load state dict from checkpoint.

        Args:
            checkpoint_file_path (Path): path to the checkpoint file.

        Returns:
            dict: state dict.
        """
        return torch.load(str(checkpoint_file_path))

    # ======================================
    # Helper functions for saving state dict
    # ======================================

    def save_safetensors_state_dict(self, *args, **kwargs):
        """
        Save safetensors state dict to checkpoint.
        """
        # TODO(FrankLeeeee): support huggingface safetensors
        raise NotImplementedError("This method is not implemented to support safe tensors")

    def generate_checkpoint_shard_file_name(self, index: int, total_number: int, prefix: str = None):
        """
        Generate checkpoint shard file name.

        Args:
            index (int): index of the shard.
            total_number (int): total number of shards.
            prefix (str): prefix of the shard file name. Default: None.
        """
        if prefix is None:
            return f"{index}-of-{total_number}.bin"
        else:
            return f"{prefix}-{index}-of-{total_number}.bin"

    def save_checkpoint(self, state_dict: dict, checkpoint_file_path: Path):
        """
        Save state dict to checkpoint.

        Args:
            state_dict (dict): state dict.
            checkpoint_file_path (Path): path to the checkpoint file.
        """
        torch.save(state_dict, str(checkpoint_file_path))

    def save_state_dict_as_shard(self, state_dict: dict, index: int, total_number: int, prefix: str,
                                 checkpoint_path: Path):
        """
        Save state dict as shard.

        Args:
            state_dict (dict): state dict.
            checkpoint_path (Path): path to the checkpoint file.
        """
        # generate the shard name
        shard_file_name = self.generate_checkpoint_shard_file_name(index, total_number, prefix)
        shard_file_path = checkpoint_path.joinpath(shard_file_name)

        # save the shard
        self.save_checkpoint(state_dict, shard_file_path)

    def calculate_param_size(self, param: torch.Tensor):
        """
        Calculate the size of a parameter in MB. Used to compute whether a group of params exceed the shard size.
        If so, a new shard should be created.

        ArgsL
            param (torch.Tensor): parameter tensor.
        """
        # TODO(FrankLeeeee): check if this tensor is a DTensor, compute its global size if so
        return param.numel() * param.element_size() / 1024 / 1024


class ShardCheckpointIndexFile:
    """
    This class is a data structure to keep the content in the index.json file for sharded checkpoint.

    Example:
        >>> index = ShardCheckpointIndexFile()
        >>> index.load('index.json')
        >>> index.append_metadata('model_type', 'bert')
        >>> index.append_weight_map('bert.embeddings.word_embeddings.weight', 'bert.embeddings.word_embeddings.weight-0-of-2.bin')
        >>> index.export('index.json')
    """

    def __init__(self) -> None:
        self.metadata: dict = dict()
        self.weight_map: dict = dict()

    def load(self, json_path: str):
        """
        Load the index file from a json file.

        Args:
            json_path (str): path to the json file.
        """
        # load the json file
        with open(json_path, 'r') as f:
            index = json.load(f)

        # assign attributes if exists
        if "metadata" in index:
            self.metadata = index["metadata"]
        if "weight_map" in index:
            self.weight_map = index["weight_map"]

    def export(self, json_path: str):
        """
        Export the index file to a json file.

        Args:
            json_path (str): path to the json file.
        """
        # create the index file
        index = dict()
        index["metadata"] = self.metadata
        index["weight_map"] = self.weight_map

        # export the index file
        with open(json_path, 'w') as f:
            json.dump(index, f, indent=4)

    def append_weight_map(self, param_name: str, shard_file: str):
        """
        Append a weight map entry to the index file.

        Args:
            param_name (str): name of the parameter.
            shard_file (str): name of the shard file.
        """
        self.weight_map[param_name] = shard_file

    def append_meta_data(self, name: str, val: Any):
        """
        Append a metadata entry to the index file.

        Args:
            name (str): name of the metadata.
            val (Any): value of the metadata.
        """
        self.metadata[name] = val
