import torch
from torch.utils.data import Dataset, DataLoader

class DummyLLMDataset(Dataset):
    def __init__(self, keys, seq_len, size=500):
        self.keys = keys
        self.seq_len = seq_len
        self.data = self._generate_data()
        self.size = size

    def _generate_data(self):
        data = {}
        for key in self.keys:
            data[key] = torch.ones(self.seq_len, dtype = torch.long)
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {key: self.data[key] for key in self.keys}