import torch


class DummyDataloader:
    def __init__(self, batch_size, vocab_size, seq_length):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.step = 0

    def generate(self):
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(
                self.batch_size,
                self.seq_length,
            ),
        )
        types = torch.randint(
            low=0,
            high=3,
            size=(
                self.batch_size,
                self.seq_length,
            ),
        )
        sentence_order = torch.randint(low=0, high=2, size=(self.batch_size,))
        loss_mask = torch.randint(
            low=0,
            high=2,
            size=(
                self.batch_size,
                self.seq_length,
            ),
        )
        lm_labels = torch.randint(low=0, high=self.vocab_size, size=(self.batch_size, self.seq_length))
        padding_mask = torch.randint(low=0, high=2, size=(self.batch_size, self.seq_length))
        return dict(
            text=tokens,
            types=types,
            is_random=sentence_order,
            loss_mask=loss_mask,
            labels=lm_labels,
            padding_mask=padding_mask,
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate()
