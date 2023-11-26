from abc import abstractstaticmethod

from colossal_eval.utils import jdump


class BaseDataset:
    """
    Base class for dataset wrapper.

    Args:
        path: The path to the original dataset.
        logger: Logger for the dataset.
    """

    def __init__(self, path, logger, few_shot):
        self.dataset = self.load(path, logger, few_shot)

    def save(self, save_path):
        """Save the converted dataset"""
        jdump(self.dataset, save_path)

    @abstractstaticmethod
    def load(path, logger):
        """Load the original dataset and convert it into the inference dataset"""
