from typing import List, Optional, Dict, Any, Tuple
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
from .meta import ParamDistMeta
import warnings


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
    model_shards = []
    buffer = {}
    buffer_size = 0
    for k, tensor in model_state_dict.items():
        if buffer_size >= max_shard_size:
            model_shards.append(buffer)
            buffer = {}
            buffer_size = 0
        buffer[k] = tensor
        buffer_size += tensor.numel() * tensor.element_size()
    if len(buffer) > 0:
        model_shards.append(buffer)
    if not has_optimizer:
        return model_shards, []
    optimizer_shards = []
    buffer = {'state': {}, 'param_groups': optimizer_state_dict['param_groups']}
    buffer_size = 0
    for k, state in optimizer_state_dict['state'].items():
        if buffer_size >= max_shard_size:
            optimizer_shards.append(buffer)
            buffer = {'state': {}}
            buffer_size = 0
        buffer['state'][k] = state
        buffer_size += compute_optimizer_state_size(state)
    if len(buffer['state']) > 0:
        optimizer_shards.append(buffer)
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
                      dist_meta: Optional[Dict[str, ParamDistMeta]] = None) -> Tuple[List[dict], List[dict], dict]:
    save_global = dist_meta is None
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict() if optimizer else None
    meta = {'dist_meta': dist_meta}
    if optimizer:
        param_to_os = param_to_os or get_param_to_os(model, optimizer)
        paired_os = get_paired_os(model_state_dict, optimizer_state_dict, param_to_os)
        meta['param_to_os'] = param_to_os
        meta['paired_os'] = paired_os
    if not save_global:
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
    if len(model_state_dict) == 0:
        warnings.warn('model state dict is empty, checkpoint is not saved', category=RuntimeWarning)
        return [], [], meta
    if max_size <= 0:
        model_checkpoints = [model_state_dict]
        optimizer_checkpoints = [optimizer_state_dict] if optimizer else []
    else:
        model_checkpoints, optimizer_checkpoints = shard_checkpoint(max_size, model_state_dict, optimizer_state_dict,
                                                                    param_to_os)
    return model_checkpoints, optimizer_checkpoints, meta
