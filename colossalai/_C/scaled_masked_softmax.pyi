from torch import Tensor

def forward(input: Tensor, mask: Tensor, scale: float) -> Tensor:
    ...


def backward(output_grads: Tensor, softmax_results: Tensor, scale: float) -> Tensor:
    ...


def get_batch_per_block(query_seq_len: int, key_seq_len: int, batches: int, attn_heads: int) -> int:
    ...
