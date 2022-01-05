import torch
from torch.utils.data import SequentialSampler


def get_random_data_loader(
    batch_size,
    total_samples,
    sequence_length,
    device,
    data_type=torch.float,
    is_distrbuted=False,
):
    train_data = torch.randint(
        low=0,
        high=1000,
        size=(total_samples, sequence_length),
        device=device,
        dtype=torch.long,
    )
    train_label = torch.randint(
        low=0, high=2, size=(total_samples,), device=device, dtype=torch.long
    )
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader