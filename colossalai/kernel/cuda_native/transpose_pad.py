import torch
import importlib

try:
    colossal_transpose_pad = importlib.import_module("colossal_transpose_pad")
except ImportError:
    raise RuntimeError('transpose_pad requires cuda extensions')


# from transpose import transpose_pad_wrapper, transpose_depad_wrapper

def transpose_pad(src, batch_size, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = colossal_transpose_pad.transpose_pad_wrapper(src, batch_size, max_seq_len, seq_len_list, head_num, size_per_head)

    return dst
    

def transpose_depad(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num, size_per_head):
    src = src.contiguous()

    dst = colossal_transpose_pad.transpose_depad_wrapper(src, batch_size, sum_seq, max_seq_len, seq_len_list, head_num, size_per_head)

    return dst


