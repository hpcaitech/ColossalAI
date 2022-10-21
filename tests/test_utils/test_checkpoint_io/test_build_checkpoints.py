import torch
import torch.nn as nn
from colossalai.utils.checkpoint_io.meta import ParamDistMeta
from colossalai.utils.checkpoint_io.utils import build_checkpoints
from torch.optim import Adam


class DummyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(20, 1)


def test_global_model():
    model = DummyModel()
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(0, model)
    assert len(model_checkpoints) == 1
    assert len(optimizer_checkpoints) == 0
    assert meta['dist_meta'] is None
    orig_state_dict = model.state_dict()
    global_state_dict = model_checkpoints[0]
    assert set(orig_state_dict.keys()) == set(global_state_dict.keys())
    for k, v in orig_state_dict.items():
        assert torch.equal(v, global_state_dict[k])


def test_global_model_shard():
    model = DummyModel()
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(80, model)
    assert len(model_checkpoints) == 2
    assert len(optimizer_checkpoints) == 0
    assert meta['dist_meta'] is None
    orig_state_dict = model.state_dict()
    assert set(orig_state_dict.keys()) == set(model_checkpoints[0].keys()) | set(model_checkpoints[1].keys())
    assert len(set(model_checkpoints[0].keys()) & set(model_checkpoints[1].keys())) == 0
    for k, v in orig_state_dict.items():
        for state_dict in model_checkpoints:
            if k in state_dict:
                assert torch.equal(v, state_dict[k])


def test_global_optimizer():
    model = DummyModel()
    for p in model.parameters():
        p.grad = torch.rand_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(0, model, optimizer)
    assert len(optimizer_checkpoints) == 1
    assert meta['param_to_os'] == {'fc.weight': 0, 'fc.bias': 1}
    for state in meta['paired_os'].values():
        for k, is_paired in state.items():
            if k == 'step':
                assert not is_paired
            else:
                assert is_paired
    orig_state_dict = optimizer.state_dict()
    state_dict = optimizer_checkpoints[0]
    for k, orig_state in orig_state_dict['state'].items():
        state = state_dict['state'][k]
        for v1, v2 in zip(orig_state.values(), state.values()):
            if isinstance(v2, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v2 == v2
    assert orig_state_dict['param_groups'] == state_dict['param_groups']


def test_global_optimizer_shard():
    model = DummyModel()
    for p in model.parameters():
        p.grad = torch.rand_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(80, model, optimizer)
    assert len(optimizer_checkpoints) == 2
    assert 'param_groups' in optimizer_checkpoints[0] and 'param_groups' not in optimizer_checkpoints[1]
    orig_state_dict = optimizer.state_dict()
    assert set(orig_state_dict['state'].keys()) == set(optimizer_checkpoints[0]['state'].keys()) | set(
        optimizer_checkpoints[1]['state'].keys())
    assert len(set(optimizer_checkpoints[0]['state'].keys()) & set(optimizer_checkpoints[1]['state'].keys())) == 0
    for k, orig_state in orig_state_dict['state'].items():
        state = optimizer_checkpoints[0]['state'][k] if k in optimizer_checkpoints[0][
            'state'] else optimizer_checkpoints[1]['state'][k]
        for v1, v2 in zip(orig_state.values(), state.values()):
            if isinstance(v2, torch.Tensor):
                assert torch.equal(v1, v2)
            else:
                assert v1 == v2

    assert orig_state_dict['param_groups'] == optimizer_checkpoints[0]['param_groups']


def test_dist_model_optimizer():
    model = DummyModel()
    for p in model.parameters():
        p.grad = torch.rand_like(p)
    optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    dist_meta = {'fc.weight': ParamDistMeta(0, 2, 0, 1), 'fc.bias': ParamDistMeta(1, 2, 0, 1)}
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(0, model, optimizer, dist_meta=dist_meta)
    assert dist_meta == meta['dist_meta']
    assert len(model_checkpoints) == 1
    assert len(optimizer_checkpoints) == 1
    assert 'fc.weight' in model_checkpoints[0] and 'fc.bias' in model_checkpoints[0]
    assert 0 in optimizer_checkpoints[0]['state'] and 1 in optimizer_checkpoints[0]['state']
    dist_meta = {'fc.weight': ParamDistMeta(1, 2, 0, 1), 'fc.bias': ParamDistMeta(1, 2, 0, 1)}
    model_checkpoints, optimizer_checkpoints, meta = build_checkpoints(0, model, optimizer, dist_meta=dist_meta)
    assert dist_meta == meta['dist_meta']
    assert len(model_checkpoints) == 1
    assert len(optimizer_checkpoints) == 1


if __name__ == '__main__':
    test_global_model()
    test_global_model_shard()
    test_global_optimizer()
    test_global_optimizer_shard()
    test_dist_model_optimizer()
