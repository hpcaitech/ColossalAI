#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .._common_utils import divide


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)



