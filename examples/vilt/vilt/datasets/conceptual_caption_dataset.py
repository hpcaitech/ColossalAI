from glob import glob
from .base_dataset import BaseDataset


class ConceptualCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"conceptual_caption_train_{i}" for i in range(30)]
        elif split == "val":
            names = ["conceptual_caption_val_0"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        return self.get_suite(index)
