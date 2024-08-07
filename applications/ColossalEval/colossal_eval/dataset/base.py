from abc import abstractstaticmethod

from colossal_eval.utils import jdump
from torch.utils.data import Dataset

from colossalai.logging import DistributedLogger


class BaseDataset:
    """
    Base class for dataset wrapper.

    Args:
        path: The path to the original dataset.
        logger: Logger for the dataset.
    """

    def __init__(self, path, logger, *args, **kwargs):
        self.dataset = self.load(path, logger, *args, **kwargs)

    def save(self, save_path):
        """Save the converted dataset"""
        jdump(self.dataset, save_path)

    @abstractstaticmethod
    def load(path, logger: DistributedLogger, *args, **kwargs):
        """Load the original dataset and convert it into the inference dataset"""


class DistributedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
