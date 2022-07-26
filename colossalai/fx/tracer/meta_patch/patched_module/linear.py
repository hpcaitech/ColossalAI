import torch
from ..registry import meta_patched_module


@meta_patched_module.register(torch.nn.Linear)
def torch_nn_linear(self, input):
    last_dim = input.shape[-1]
    assert last_dim == self.in_features, f'Expected hidden size {self.in_features} but got {last_dim} for the torch.nn.Linear patch'
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")

@meta_patched_module.register(torch.nn.GRU)
@meta_patched_module.register(torch.nn.RNN)
def torch_nn_rnn(self, input, h0):
    assert input.shape[-1] == self.input_size, f'Expected input to have input size {self.input_size} but got {input.shape[-1]} for the torch.nn.RNN patch'
    assert h0.shape[-1] == self.hidden_size, f'Expected h0 to have hidden size {self.hidden_size} but got {h0.shape[-1]} for the torch.nn.RNN patch'
    d = 2 if self.bidirectional else 1
    return torch.empty(input.shape[:-1] + (self.hidden_size * d,), device="meta"), h0
    