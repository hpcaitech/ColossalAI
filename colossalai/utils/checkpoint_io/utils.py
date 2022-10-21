import warnings
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from .meta import ParamDistMeta


def run_if_not_none(fn: Callable[[Any], Any], arg: Any) -> Any:
    if arg is not None:
        return fn(arg)


def get_param_to_os(model: Module, optimizer: Optimizer) -> Dict[str, int]:
    # ensure all params in optimizer are in model state dict
    params_set = set(id(p) for p in model.parameters())
    for group in optimizer.param_groups:
        for p in group['params']:
            assert id(p) in params_set
    param_mappings = {}
    start_index = 0

    def get_group_mapping(group):
        nonlocal start_index
        param_mappings.update(
            {id(p): i for i, p in enumerate(group['params'], start_index) if id(p) not in param_mappings})
        start_index += len(group['params'])

    for g in optimizer.param_groups:
        get_group_mapping(g)
    return {k: param_mappings[id(p)] for k, p in model.named_parameters()}


def compute_optimizer_state_size(state: Dict[str, Any]) -> int:
    size = 0
    for v in state.values():
        if isinstance(v, Tensor):
            size += v.numel() * v.element_size()
    return size


class ModelCheckpointSharder:

    def __init__(self, max_shard_size: int) -> None:
        self.max_shard_size = max_shard_size
        self.buffer: Dict[str, Tensor] = {}
        self.buffer_size: int = 0

    def append(self, key: str, tensor: Tensor) -> Optional[dict]:
        retval = None
        if self.max_shard_size > 0 and self.buffer_size >= self.max_shard_size:
            retval = self.buffer
            self.buffer = {}
            self.buffer_size = 0
        self.buffer[key] = tensor
        self.buffer_size += tensor.numel() * tensor.element_size()
        return retval

    def extend(self, state_dict: Dict[str, Tensor]) -> List[dict]:
        shards = []
        for key, tensor in state_dict.items():
            shard = self.append(key, tensor)
            run_if_not_none(shards.append, shard)
        return shards

    def complete(self) -> Optional[dict]:
        return self.buffer if len(self.buffer) > 0 else None


class OptimizerCheckpointSharder:

    def __init__(self, max_shard_size: int, param_groups: dict) -> None:
        self.max_shard_size = max_shard_size
        self.buffer: Dict[str, dict] = {'state': {}, 'param_groups': param_groups}
        self.buffer_size: int = 0
        self.returned_first: bool = False

    def append(self, key: int, state: dict) -> Optional[dict]:
        retval = None
        if self.max_shard_size > 0 and self.buffer_size >= self.max_shard_size:
            retval = self.buffer
            self.buffer = {'state': {}}
            self.buffer_size = 0
        self.buffer['state'][key] = state
        self.buffer_size += compute_optimizer_state_size(state)
        return retval

    def extend(self, state_dict: Dict[str, dict]) -> List[dict]:
        shards = []
        for key, state in state_dict['state'].items():
            shard = self.append(key, state)
            run_if_not_none(shards.append, shard)
        return shards

    def complete(self) -> Optional[dict]:
        return self.buffer if len(self.buffer['state']) > 0 else None


def shard_checkpoint(max_shard_size: int,
                     model_state_dict: Dict[str, Tensor],
                     optimizer_state_dict: Optional[dict] = None,
                     param_to_os: Optional[dict] = None) -> Tuple[List[dict], List[dict]]:
    has_optimizer: bool = False
    if optimizer_state_dict is not None:
        assert param_to_os is not None
        os_to_param = {v: k for k, v in param_to_os.items()}
        for os_key in optimizer_state_dict['state'].keys():
            assert os_key in os_to_param
            assert os_to_param[os_key] in model_state_dict
        has_optimizer = True
    model_sharder = ModelCheckpointSharder(max_shard_size)
    model_shards = model_sharder.extend(model_state_dict)
    run_if_not_none(model_shards.append, model_sharder.complete())
    if not has_optimizer:
        return model_shards, []
    optimizer_sharder = OptimizerCheckpointSharder(max_shard_size, optimizer_state_dict['param_groups'])
    optimizer_shards = optimizer_sharder.extend(optimizer_state_dict)
    run_if_not_none(optimizer_shards.append, optimizer_sharder.complete())
    return model_shards, optimizer_shards


def get_paired_os(model_state_dict: Dict[str, Tensor], optimizer_state_dict: dict, param_to_os: Dict[str, int]) -> dict:
    os_to_param = {v: k for k, v in param_to_os.items()}
    paired_os = {}
    for idx, state in optimizer_state_dict['state'].items():
        paired_os[idx] = {}
        p = model_state_dict[os_to_param[idx]]
        for k, v in state.items():
            if isinstance(v, Tensor) and v.shape == p.shape:
                paired_os[idx][k] = True
            else:
                paired_os[idx][k] = False
    return paired_os


def build_checkpoints(max_size: int,
                      model: Module,
                      optimizer: Optional[Optimizer] = None,
                      param_to_os: Optional[Dict[str, int]] = None,
                      dist_meta: Optional[Dict[str, ParamDistMeta]] = None,
                      eliminate_replica: bool = False) -> Tuple[List[dict], List[dict], dict]:
    save_global = dist_meta is None
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict() if optimizer else None
    meta = {'dist_meta': dist_meta}
    if optimizer:
        param_to_os = param_to_os or get_param_to_os(model, optimizer)
        paired_os = get_paired_os(model_state_dict, optimizer_state_dict, param_to_os)
        meta['param_to_os'] = param_to_os
        meta['paired_os'] = paired_os
    if not save_global and eliminate_replica:
        # filter dp replicated params
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if dist_meta[k].used_zero or dist_meta[k].dp_rank == 0
        }
        if optimizer:
            optimizer_state_dict['state'] = {
                param_to_os[k]: optimizer_state_dict['state'][param_to_os[k]]
                for k in model_state_dict.keys()
                if dist_meta[k].used_zero or dist_meta[k].dp_rank == 0
            }
    meta['params'] = list(model_state_dict.keys())
    if len(model_state_dict) == 0:
        warnings.warn('model state dict is empty, checkpoint is not saved')
        return [], [], meta
    model_checkpoints, optimizer_checkpoints = shard_checkpoint(max_size, model_state_dict, optimizer_state_dict,
                                                                param_to_os)
    return model_checkpoints, optimizer_checkpoints, meta


def is_duplicated_list(list_: List[Any]) -> bool:
    if len(list_) == 0:
        return True
    elem = list_[0]
    for x in list_[1:]:
        if x != elem:
            return False
    return True


def copy_optimizer_state(src_state: dict, dest_state: dict) -> None:
    for k, v in src_state.items():
        if k in dest_state:
            old_v = dest_state[k]
            if isinstance(old_v, Tensor):
                old_v.copy_(v)
        else:
            dest_state[k] = v


def optimizer_load_state_dict(optimizer: Optimizer, state_dict: dict, strict: bool = False) -> None:
    assert optimizer.state_dict()['param_groups'] == state_dict['param_groups']
    state_dict = deepcopy(state_dict)
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']
    idx_to_p: Dict[str, Parameter] = {
        old_id: p for old_id, p in zip(chain.from_iterable((g['params'] for g in saved_groups
                                                           )), chain.from_iterable((g['params'] for g in groups)))
    }
    missing_keys = list(set(idx_to_p.keys()) - set(state_dict['state'].keys()))
    unexpected_keys = []
    error_msgs = []
    for idx, state in state_dict['state'].items():
        if idx in idx_to_p:
            old_state = optimizer.state[idx_to_p[idx]]
            copy_optimizer_state(state, old_state)
        else:
            unexpected_keys.append(idx)
    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(optimizer.__class__.__name__,
                                                                                 "\n\t".join(error_msgs)))
