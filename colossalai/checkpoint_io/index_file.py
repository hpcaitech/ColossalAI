import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Union

from .utils import is_dtensor_checkpoint

__all__ = ['CheckpointIndexFile']


class CheckpointIndexFile:
    """
    This class is a data structure to keep the content in the index.json file for sharded checkpoint.

    Example:
        >>> index = CheckpointIndexFile.from_file('model.index.json')
        >>> index.append_metadata('model_type', 'bert')
        >>> index.append_weight_map('bert.embeddings.word_embeddings.weight', 'model_0001-of-0002.bin')
        >>> index.export('new_index.json')
    """

    def __init__(self, root_path=None) -> None:
        self.root_path = root_path

        # use ordered dict to preserve the tensor checkpoint order
        self.metadata: Dict = OrderedDict()
        self.weight_map: Dict = OrderedDict()

    @staticmethod
    def from_file(index_path: Union[str, Path]):
        """
        Create a CheckpointIndexFile object from a json file.

        Args:
            index_path (str): path to the json file.

        Returns:
            CheckpointIndexFile: CheckpointIndexFile object.
        """
        index = CheckpointIndexFile()
        index.load(index_path)
        return index

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

        # assign the root directory for the index file
        self.root_path = Path(json_path).absolute().parent

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

    def contains_dtensor(self):
        """
        Check if the index file contains any distributed tensor. The distributed tensors will be stored in
        `dtensor/module.linear.weight.*.bin` or `dtensor/module.linear.weight.*.safetensors` in the weight map.

        Returns:
            bool: True if the index file contains any distributed tensor, False otherwise.
        """
        for value in self.weight_map.values():
            if value.endswith(".*.bin") or value.endswith(".*.safetensors"):
                return True
        return False

    def get_checkpoint_filenames(self) -> List[str]:
        """
        Get the set of checkpoint filenames in the weight map.

        Returns:
            list: checkpoint shard filenames.
        """
        # read the checkpoint file list from the json file and get a list of unique file names
        checkpoint_files = sorted(list(set(self.weight_map.values())))

        # get the absolute paths for all checkpoint files
        checkpoint_files = [str(self.root_path.joinpath(f)) for f in checkpoint_files]

        dtensor_list = []
        checkpoint_list = []

        for ckpt_file in checkpoint_files:
            if is_dtensor_checkpoint(ckpt_file):
                dtensor_list.append(ckpt_file)
            else:
                checkpoint_list.append(ckpt_file)

        return checkpoint_list, dtensor_list

    def assert_no_dtensor_checkpoint(self):
        for val in self.weight_map.values():
            if is_dtensor_checkpoint(val):
                raise ValueError(f"Checkpoint file {val} contains distributed tensor")

    def get_checkpoint_file(self, param_name: str) -> str:
        """
        Get the checkpoint file name for a parameter.

        Args:
            param_name (str): name of the parameter.

        Returns:
            str: checkpoint file name.
        """
        ckpt_path = self.weight_map[param_name]
        return ckpt_path

    def get_all_param_names(self):
        """
        Get all the weight keys.
        """
        return list(self.weight_map.keys())

    def get_param_group_filename(self) -> Union[str, None]:
        """
        Get the file name of param_group file if this is a checkpoint for optimizer.
        Returns:
            str: param_group file name
        """
        filename = self.metadata.get("param_groups", None)
        if filename:
            return str(self.root_path.joinpath(filename))
        else:
            return None

    def write_index_file(self, save_index_file):
        """
        Write index file.
        """
        save_index_file = os.path.join(self.root_path, save_index_file)
        index = {"metadata": self.metadata, "weight_map": self.weight_map}
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)
