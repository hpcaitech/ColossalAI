from vilt.datasets import CocoCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class CocoCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CocoCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return CocoCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "coco"
