import argparse

import torch
from benchmark_utils import benchmark
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AlbertConfig,
    AlbertForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2.4e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
SEQ_LEN = 512
VOCAB_SIZE = 1000
NUM_LABELS = 10
DATASET_LEN = 1000


class RandintDataset(Dataset):
    def __init__(self, dataset_length: int, sequence_length: int, vocab_size: int, n_class: int):
        self._sequence_length = sequence_length
        self._vocab_size = vocab_size
        self._n_class = n_class
        self._dataset_length = dataset_length
        self._datas = torch.randint(
            low=0,
            high=self._vocab_size,
            size=(
                self._dataset_length,
                self._sequence_length,
            ),
            dtype=torch.long,
        )
        self._labels = torch.randint(low=0, high=self._n_class, size=(self._dataset_length, 1), dtype=torch.long)

    def __len__(self):
        return self._dataset_length

    def __getitem__(self, idx):
        return self._datas[idx], self._labels[idx]


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="mrpc", help="GLUE task to run")
    parser.add_argument(
        "-p",
        "--plugin",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "torch_ddp_fp16", "gemini", "low_level_zero"],
        help="plugin to use",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help="bert or albert",
    )

    args = parser.parse_args()

    # ==============================
    # Launch Distributed Environment
    # ==============================
    colossalai.launch_from_torch(seed=42)
    coordinator = DistCoordinator()

    # local_batch_size = BATCH_SIZE // coordinator.world_size
    lr = LEARNING_RATE * coordinator.world_size

    # ==============================
    # Instantiate Plugin and Booster
    # ==============================
    booster_kwargs = {}
    if args.plugin == "torch_ddp_fp16":
        booster_kwargs["mixed_precision"] = "fp16"
    if args.plugin.startswith("torch_ddp"):
        plugin = TorchDDPPlugin()
    elif args.plugin == "gemini":
        plugin = GeminiPlugin(placement_policy="cuda", strict_ddp_mode=True, initial_scale=2**5)
    elif args.plugin == "low_level_zero":
        plugin = LowLevelZeroPlugin(initial_scale=2**5)

    booster = Booster(plugin=plugin, **booster_kwargs)

    # ==============================
    # Prepare Dataloader
    # ==============================

    train_dataset = RandintDataset(
        dataset_length=DATASET_LEN, sequence_length=SEQ_LEN, vocab_size=VOCAB_SIZE, n_class=NUM_LABELS
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # ====================================
    # Prepare model, optimizer
    # ====================================
    # bert pretrained model

    if args.model_type == "bert":
        cfg = BertConfig(vocab_size=VOCAB_SIZE, num_labels=NUM_LABELS)
        model = BertForSequenceClassification(cfg)
    elif args.model_type == "albert":
        cfg = AlbertConfig(vocab_size=VOCAB_SIZE, num_labels=NUM_LABELS)
        model = AlbertForSequenceClassification(cfg)
    else:
        raise RuntimeError

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = HybridAdam(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    # lr scheduler
    total_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_FRACTION * total_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # criterion
    criterion = lambda inputs: inputs[0]

    # ==============================
    # Boost with ColossalAI
    # ==============================
    model, optimizer, _, _, lr_scheduler = booster.boost(model, optimizer, lr_scheduler=lr_scheduler)

    # ==============================
    # Benchmark model
    # ==============================

    results = benchmark(
        model, booster, optimizer, lr_scheduler, train_dataloader, criterion=criterion, epoch_num=NUM_EPOCHS
    )

    coordinator.print_on_master(results)


if __name__ == "__main__":
    main()
