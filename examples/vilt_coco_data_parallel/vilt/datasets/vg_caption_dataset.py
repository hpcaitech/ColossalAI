from .base_dataset import BaseDataset


class VisualGenomeCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = ["vg"]
        elif split == "val":
            names = []

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
