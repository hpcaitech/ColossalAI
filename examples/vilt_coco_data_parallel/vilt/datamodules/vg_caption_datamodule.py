from vilt.datasets import VisualGenomeCaptionDataset
from .datamodule_base import BaseDataModule


class VisualGenomeCaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VisualGenomeCaptionDataset

    @property
    def dataset_name(self):
        return "vg"
