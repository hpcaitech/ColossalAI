import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class BeansDataset(Dataset):
    def __init__(self, image_processor, tp_size=1, split="train"):
        super().__init__()
        self.image_processor = image_processor
        self.ds = load_dataset("beans")[split]
        self.label_names = self.ds.features["labels"].names
        while len(self.label_names) % tp_size != 0:
            # ensure that the number of labels is multiple of tp_size
            self.label_names.append(f"pad_label_{len(self.label_names)}")
        self.num_labels = len(self.label_names)
        self.inputs = []
        for example in self.ds:
            self.inputs.append(self.process_example(example))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def process_example(self, example):
        input = self.image_processor(example["image"], return_tensors="pt")
        input["labels"] = example["labels"]
        return input


def beans_collator(batch):
    return {
        "pixel_values": torch.cat([data["pixel_values"] for data in batch], dim=0),
        "labels": torch.tensor([data["labels"] for data in batch], dtype=torch.int64),
    }
