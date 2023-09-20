import torch

from colossalai.legacy.context import ParallelMode
from colossalai.legacy.context.parallel_context import ParallelContext
from colossalai.legacy.core import global_context as gpc
from colossalai.logging import get_dist_logger

from .datasets.builder import build_train_valid_test_datasets
from .datasets.data_samplers import build_pretraining_data_loader


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_iterators(
    train_iters, global_batch_size, eval_interval, eval_iters, dataloader_type="single", **kwargs
):
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    logger = get_dist_logger()
    logger.info("> building train, validation, and test datasets ...", ranks=[0])

    # Backward compatibility, assume fixed batch size.
    # if iteration > 0 and consumed_train_samples == 0:
    #     assert train_samples is None, \
    #         'only backward compatibility support for iteration-based training'
    #     consumed_train_samples = iteration * global_batch_size
    # if iteration > 0 and consumed_valid_samples == 0:
    #     if train_samples is None:
    #         consumed_valid_samples = (iteration // eval_interval) * \
    #             eval_iters * global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if not gpc.is_initialized(ParallelMode.TENSOR) or gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        # Number of train/valid/test samples.
        train_samples = train_iters * global_batch_size
        eval_iters_ = (train_iters // eval_interval + 1) * eval_iters
        test_iters = eval_iters
        train_val_test_num_samples = [train_samples, eval_iters_ * global_batch_size, test_iters * global_batch_size]
        logger.info(" > datasets target sizes (minimum size):")
        logger.info("    train:      {}".format(train_val_test_num_samples[0]), ranks=[0])
        logger.info("    validation: {}".format(train_val_test_num_samples[1]), ranks=[0])
        logger.info("    test:       {}".format(train_val_test_num_samples[2]), ranks=[0])

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            train_valid_test_num_samples=train_val_test_num_samples, **kwargs
        )

        # Build dataloaders.
        dp_size = gpc.get_world_size(ParallelMode.DATA)
        train_dataloader = build_pretraining_data_loader(
            train_ds, consumed_samples=0, micro_batch_size=global_batch_size // dp_size
        )
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, consumed_samples=0, micro_batch_size=global_batch_size // dp_size
        )
        test_dataloader = build_pretraining_data_loader(test_ds, 0, micro_batch_size=global_batch_size // dp_size)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and train_iters > 0
        do_valid = valid_dataloader is not None and eval_iters > 0
        do_test = test_dataloader is not None and eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(
        flags, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], group=gpc.get_group(ParallelMode.TENSOR)
    )

    # Build iterators.
    dl_type = dataloader_type
    assert dl_type in ["single", "cyclic"]

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == "single" else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == "single" else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == "single" else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
