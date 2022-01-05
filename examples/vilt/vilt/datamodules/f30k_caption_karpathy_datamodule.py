from vilt.datasets import F30KCaptionKarpathyDataset
from .datamodule_base import BaseDataModule


class F30KCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return F30KCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return F30KCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "f30k"
