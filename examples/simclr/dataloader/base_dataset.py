from abc import ABC

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from colossalai.builder import build_transform


class SimCLRTransform():
    def __init__(self, transform_list):
        self.transform = transforms.Compose(transform_list)
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 


class BaseDataset(Dataset, ABC):

    def __init__(self, transform_pipeline: list):
        transform_list = [build_transform(cfg) for cfg in transform_pipeline]
        transform = SimCLRTransform(transform_list)
        self._transform_pipeline = transform