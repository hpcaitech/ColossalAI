import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.GRU)
@meta_patched_module.register(torch.nn.RNN)
def torch_nn_rnn(self, input, hx):
    assert (
        input.shape[-1] == self.input_size
    ), f"Expected input to have input size {self.input_size} but got {input.shape[-1]} for the torch.nn.RNN patch"
    assert (
        hx.shape[-1] == self.hidden_size
    ), f"Expected hx to have hidden size {self.hidden_size} but got {hx.shape[-1]} for the torch.nn.RNN patch"
    d = 2 if self.bidirectional else 1
    return torch.empty(input.shape[:-1] + (self.hidden_size * d,), device="meta"), hx
