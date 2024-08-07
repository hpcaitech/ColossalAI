from typing import Callable

from torch.utils.data import Dataset


class DummyLLMDataset(Dataset):
    def __init__(self, keys, seq_len, size=500, gen_fn={}):
        self.keys = keys
        self.gen_fn = gen_fn
        self.seq_len = seq_len
        self.data = self._generate_data()
        self.size = size

    def _generate_data(self):
        data = {}
        for key in self.keys:
            if key in self.gen_fn:
                data[key] = self.gen_fn[key]
            else:
                data[key] = [1] * self.seq_len
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            key: self.data[key] if not isinstance(self.data[key], Callable) else self.data[key](idx)
            for key in self.keys
        }
