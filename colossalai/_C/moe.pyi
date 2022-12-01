from torch import Tensor

def cumsum_sub_one(mask: Tensor) -> Tensor:
    ...


def dispatch_forward(s: int, ec: int, h: int, batch_tokens: Tensor, mask: Tensor, dest_idx: Tensor) -> Tensor:
    ...


def dispatch_backward(s: int, ec: int, h: int, expert_grad: Tensor, mask: Tensor, dest_idx: Tensor) -> Tensor:
    ...


def combine_forward(s: int, e: int, c: int, h: int, expert_tokens: Tensor, logits: Tensor, mask: Tensor, dest_idx: Tensor) -> Tensor:
    ...


def combine_backward(s: int, e: int, c: int, h: int, tokens_grad: Tensor, expert_tokens: Tensor, logits: Tensor, mask: Tensor, dest_idx: Tensor) -> Tensor:
    ...
