#!/usr/bin/env python
# -*- encoding: utf-8 -*-


def _calc_incoming_device_range(i, rank, world_size, sub_seq_length):
    device_of_incoming_k = (rank - i - 1) % world_size
    start_idx = sub_seq_length * device_of_incoming_k
    end_idx = sub_seq_length * (device_of_incoming_k + 1)
    return start_idx, end_idx


def _calc_current_device_range(rank, sub_seq_length):
    start_idx = sub_seq_length * rank
    end_idx = sub_seq_length * (rank + 1)
    return start_idx, end_idx
