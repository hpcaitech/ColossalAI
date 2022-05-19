import pytest
import colossalai
import os
import random
import numpy as np
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from transformers import GPT2Config, GPT2LMHeadModel
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, ColoTensor, ColoOptimizer, DistSpecManager, distspec
from colossalai.core import global_context as gpc
from functools import partial
# Hack huggingface Bert ModelOutput
# Make it available to our ColoTensor
from transformers.file_utils import ModelOutput
from dataclasses import fields
from tests.test_tensor._utils import tensor_equal


def _post_init_colotensor(self):
    class_fields = fields(self)
    # Safety and consistency checks
    if len(class_fields) == 0:
        raise ValueError(f"{self.__class__.__name__} has no fields.")
    if not all(field.default is None for field in class_fields[1:]):
        raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

    def is_tensor_with_colo(x):
        """
        Tests if `x` is a `ColoTensor` or `torch.Tensor`.
        """
        if isinstance(x, torch.Tensor):
            return True

        return isinstance(x, ColoTensor)

    if other_fields_are_none and not is_tensor_with_colo(first_field):
        if isinstance(first_field, dict):
            iterator = first_field.items()
            first_field_iterator = True
        else:
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

        # if we provided an iterator as first field and the iterator is a (key, value) iterator
        # set the associated fields
        if first_field_iterator:
            for element in iterator:
                if (not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str)):
                    break
                setattr(self, element[0], element[1])
                if element[1] is not None:
                    self[element[0]] = element[1]
        elif first_field is not None:
            self[class_fields[0].name] = first_field
    else:
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v


ModelOutput.__post_init__ = _post_init_colotensor


class GPTLMModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 max_seq_len=1024,
                 vocab_size=50304,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(
            GPT2Config(n_embd=hidden_size,
                       n_layer=num_layers,
                       n_head=num_attention_heads,
                       n_positions=max_seq_len,
                       n_ctx=max_seq_len,
                       vocab_size=vocab_size,
                       resid_pdrop=0.0,
                       embd_pdrop=0.0,
                       attn_pdrop=0.0))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


def gpt2_s(checkpoint=True):
    return GPTLMModel(checkpoint=checkpoint)


def gpt2_m(checkpoint=True):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def init_1d_row_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1D, parallel_mode=ParallelMode.PARALLEL_1D)])
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'ln' not in n:
                p.set_spec(spec)


def init_1d_col_spec(model):
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [-1], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        [ParallelAction(priority=1, compute_pattern=ComputePattern.TP1D, parallel_mode=ParallelMode.PARALLEL_1D)])
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'ln' not in n and ('weight' in n or 'bias' in n):
                p.set_spec(spec)


def check_tensor_equal_1d(tensor: torch.Tensor, shard: ColoTensor):
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    assert len(shard.spec.dist_spec.dims) == 1
    dim = shard.spec.dist_spec.dims[0]
    assert torch.equal(tensor.chunk(world_size, dim)[rank], shard.torch_tensor())


def tensor_shard_equal(tensor: torch.Tensor, shard: torch.Tensor):
    assert tensor.ndim == shard.ndim
    if tensor.shape == shard.shape:
        return tensor_equal(tensor, shard)
    else:
        dims_not_eq = torch.nonzero(torch.tensor(tensor.shape) != torch.tensor(shard.shape))
        if dims_not_eq.numel() == 1:
            # 1D shard
            dim = dims_not_eq.item()
            world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
            rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
            return tensor_equal(tensor.chunk(world_size, dim)[rank], shard)
        else:
            raise NotImplementedError


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p, p)


def check_grad_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        assert tensor_shard_equal(torch_p.grad, p.grad)


def run_gpt(init_spec_func):
    BATCH_SIZE = 4
    SEQ_LEN = 1024
    VOCAB_SIZE = 50304
    NUM_STEPS = 1
    criterion = GPTLMLoss()
    with ColoInitContext(device=get_current_device()):
        model = gpt2_s()
    model = model.cuda()
    torch_model = gpt2_s().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)
    init_spec_func(model)
    check_param_equal(model, torch_model)
    model.train()
    torch_model.train()
    for i in range(NUM_STEPS):
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        logits = model(input_ids, attn_mask)
        torch_logits = torch_model(input_ids, attn_mask)
        assert tensor_equal(torch_logits, logits)
        loss = criterion(logits, input_ids)
        torch_loss = criterion(torch_logits, input_ids)
        loss.backward()
        torch_loss.backward()
        check_grad_equal(model, torch_model)


def run_dist(rank, world_size, port):
    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt(init_1d_row_spec)
    run_gpt(init_1d_col_spec)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(1)
