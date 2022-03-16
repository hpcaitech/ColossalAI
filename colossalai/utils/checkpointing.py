import os
import os.path as osp
import re
from typing import Tuple
from pathlib import Path

import torch

from colossalai.context import Config
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

__all__ = [
    'get_checkpoint_path', 'get_latest_checkpoint_path', 'get_latest_checkpoint_pattern', 'save_checkpoint',
    'load_checkpoint'
]


def unwrap_config(config: Config):
    """Unwrap Config objects to normal dicts
    """
    config_dict = dict()
    for k, v in config.items():
        if isinstance(v, dict):
            config_dict[k] = unwrap_config(v)
        else:
            config_dict[k] = v

    return config_dict


def _get_ranks_name(old_tp_size=None, old_pp_size=None, first=False):
    # tensor parallel
    tp_local_rank, tp_world_size = 0, 1
    if gpc.is_initialized(ParallelMode.TENSOR):
        tp_local_rank = gpc.get_local_rank(ParallelMode.TENSOR) if not first else 0
        if not old_tp_size:
            tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        else:
            tp_world_size = old_tp_size
        
    # pipeline parallel
    pp_local_rank, pp_world_size = 0, 1
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pp_local_rank = gpc.get_local_rank(ParallelMode.PIPELINE) if not first else 0
        if not old_pp_size:
            pp_world_size = gpc.get_world_size(ParallelMode.PIPELINE)
        else:
            pp_world_size = old_pp_size
    
    ranks_name = f'tp{tp_local_rank}_{tp_world_size}-pp{pp_local_rank}_{pp_world_size}'
    return ranks_name


def _get_standard_checkpoint_filename(epoch: int, suffix: str = ''):
    ranks_name = _get_ranks_name()
    return f'epoch{epoch}-{ranks_name}-{suffix}.pt'


def get_checkpoint_path(checkpoint_dir: str, epoch: int, suffix: str = ''):
    """This is a function to generate the checkpoint path from the (checkpoint_dir, epoch, suffix, gpu_parallel_rank) tuple.
    This is useful during generation and recuperation of the checkpoint.

    :param checkpoint_dir: Set up a directory for saving checkpoints
    :type checkpoint_dir: str
    :param epoch: Epoch number (indicate how many epochs have you trained this model)
    :type epoch: int
    :param suffix: Additional notation to specify the model or checkpoint, defaults to ''
    :type suffix: str, optional
    :return: Checkpoint path to be generated
    :rtype: path
    """
    ckpt_filename = _get_standard_checkpoint_filename(epoch, suffix)
    return os.path.join(checkpoint_dir, ckpt_filename)


def _ensure_directory_exists(filename: str):
    # ensure the directory exists
    dirpath = os.path.dirname(filename)
    if not os.path.exists(dirpath):
        Path(dirpath).mkdir(parents=True, exist_ok=True)


def assert_divisiblity(old_tp_size: int, old_pp_size: int):
    """assert old tp and pp sizes are divisible by new tp and pp sizes

    :param old_tp_size: Old tensor parallel size
    :type old_tp_size: int
    :param old_pp_size: Old pipeline parallel size
    :type old_pp_size: int
    """
    
    if gpc.is_initialized(ParallelMode.TENSOR):
        new_tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        assert max(old_tp_size,new_tp_size) % min(old_tp_size,new_tp_size) == 0
    if gpc.is_initialized(ParallelMode.PIPELINE):
        new_pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        assert max(old_pp_size,new_pp_size) % min(old_pp_size,new_pp_size) == 0 


def get_old_world_sizes(checkpoint_dir: str):
    """Get saved files' world sizes

    :param checkpoint_dir: Directory for saving checkpoints
    :type checkpoint_dir: str
    :return: old tp and pp world sizes
    :rtype: int
    """
    old_tp_size = int(os.listdir(checkpoint_dir)[0].split('-')[1].split('_')[1])
    old_pp_size = int(os.listdir(checkpoint_dir)[0].split('-')[2].split('_')[1])
    return old_tp_size, old_pp_size


def get_latest_checkpoint_pattern(old_tp_size, old_pp_size, suffix: str = ''):
    """Generate Regular expression of latest checkpoint's pattern

    :param suffix: Additional notation to specify the model or checkpoint, defaults to ''
    :type suffix: str, optional
    :return: Checkpoint pattern
    :rtype: regular expression
    """
    ranks_name = _get_ranks_name(old_tp_size, old_pp_size, True)
    pattern = r'epoch(\d+)-{}-{}\.pt'.format(ranks_name, suffix)
    ckpt_pattern = re.compile(pattern)
    return ckpt_pattern


def get_latest_epoch(checkpoint_dir: str, old_tp_size: int, old_pp_size: int, suffix: str = ''):
    """This is a function to retrieve the latest checkpoint path from the (checkpoint_dir, suffix, gpu_parallel_rank) tuple.
    This is useful during recuperation of the checkpoint, especially when you do not know the epoch number.

    :param checkpoint_dir: Directory for saving checkpoints
    :type checkpoint_dir: str
    :param suffix: Additional notation to specify the model or checkpoint, defaults to ''
    :type suffix: str, optional
    :raises FileNotFoundError: Raise error when we cannot find the latest checkpoint file with inputs given
    :return: The latest checkpoint path to be retrieved
    :rtype: path
    """
    CKPT_NAME_PAT = get_latest_checkpoint_pattern(old_tp_size, old_pp_size, suffix=suffix)

    last_epoch = -1
    assert osp.isdir(checkpoint_dir), f'{checkpoint_dir} is not a directory'

    for filename in os.listdir(checkpoint_dir):
        ret = CKPT_NAME_PAT.match(filename)
        if ret:
            epoch = int(ret[0].split('-')[0].lstrip('epoch'))
            if epoch > last_epoch:
                last_epoch = epoch
    if last_epoch == -1:
        ranks_name = _get_ranks_name()
        raise FileNotFoundError(f"Cannot find the latest checkpoint file for {ranks_name} in {checkpoint_dir}")
    else: 
        return last_epoch


def colossalai_load(checkpoint_path, old_tp_size, old_pp_size, epoch, suffix):
    if gpc.is_initialized(ParallelMode.TENSOR):
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        new_tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        tp_ratio = max(new_tp_size, old_tp_size) // min(new_tp_size, old_tp_size)
    else:
        tp_rank, new_tp_size, tp_ratio = 0, 0, 1
        
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        new_pp_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pp_ratio = max(new_pp_size, old_pp_size) // min(new_pp_size, old_pp_size)
    else:
        pp_rank, new_pp_size, pp_ratio = 0, 0, 1
    
    loaded_file = _file_selection(checkpoint_path, tp_rank, pp_rank, old_tp_size, old_pp_size, new_tp_size, new_pp_size, tp_ratio, pp_ratio, epoch, suffix)
    loaded_file = _pp_checkpoint(loaded_file, pp_rank, old_pp_size, new_pp_size, pp_ratio)
    loaded_file = _tp_checkpoint(loaded_file, tp_rank, old_tp_size, new_tp_size, tp_ratio)
    print('-----final-----')
    print('1 (model, opti ..)==',len(loaded_file))
    print('model ==',len(loaded_file['model']))
    return loaded_file


def _file_selection(checkpoint_path, tp_rank, pp_rank, old_tp_size, old_pp_size, new_tp_size, new_pp_size, tp_ratio, pp_ratio, epoch, suffix):
    '''
         tp
    pp[[[],[],[],[]]
       [[],[],[],[]]]
    '''
    file_list = [[] for _ in range(pp_ratio)]
    if new_tp_size >= old_tp_size:
        if new_pp_size >= old_pp_size:
            temp = torch.load(os.path.join(checkpoint_path, f'epoch{epoch}-tp{tp_rank//tp_ratio}_{old_tp_size}-pp{pp_rank//pp_ratio}_{old_pp_size}-{suffix}.pt'), map_location='cpu')
            file_list[0].append(temp)
        else:
            for i in range(pp_ratio):
                temp = torch.load(os.path.join(checkpoint_path, f'epoch{epoch}-tp{tp_rank//tp_ratio}_{old_tp_size}-pp{pp_rank*pp_ratio+i}_{old_pp_size}-{suffix}.pt'), map_location='cpu')
                file_list[i].append(temp)
    else:
        if new_pp_size >= old_pp_size:
            for j in range(tp_ratio):
                temp = torch.load(os.path.join(checkpoint_path, f'epoch{epoch}-tp{tp_rank*tp_ratio+j}_{old_tp_size}-pp{pp_rank//pp_ratio}_{old_pp_size}-{suffix}.pt'), map_location='cpu')
                file_list[0].append(temp)
        else:
            for i in range(pp_ratio):
                for j in range(tp_ratio):
                    temp = torch.load(os.path.join(checkpoint_path, f'epoch{epoch}-tp{tp_rank*tp_ratio+j}_{old_tp_size}-pp{pp_rank*pp_ratio+i}_{old_pp_size}-{suffix}.pt'), map_location='cpu')
                    file_list[i].append(temp)
    
    return file_list


def _pp_checkpoint(file_list, pp_rank, old_pp_size, new_pp_size, pp_ratio):
    '''
    in: [[[],[],[],[]]   or [[]
         [[],[],[],[]]]      []]

    out: [[],[],[],[]]
    '''
    print('-----pp-----')
    print('1 (pp)==' , len(file_list))
    print('2 (tp)==' , len(file_list[0]))
    print('3 (model, opti ..)==' , len(file_list[0][0]))
    print('model ==' , len(file_list[0][0]['model']))
    if new_pp_size >= old_pp_size:
        for indx in range(len(file_list[0])):
            temp_dict = dict()
            layer_count = int(list(file_list[0][indx]['model'].keys())[-1].split('.')[2]) + 1
            for (key, item) in file_list[0][indx]['model'].items():
                layer_offset = layer_count*(pp_rank%pp_ratio)//pp_ratio
                if int(key.split('.')[2]) >= layer_count*(pp_rank%pp_ratio)//pp_ratio and int(key.split('.')[2]) < layer_count*(pp_rank%pp_ratio+1)//pp_ratio:
                    new_key_list = key.split('.')[:2] + [str(int(key.split('.')[2])-layer_offset)] + key.split('.')[3:] 
                    new_key = '.'.join(new_key_list)
                    temp_dict[new_key] = item
            file_list[0][indx]['model'] = temp_dict
    else:
        for indx in range(len(file_list[0])):
            temp_dict = file_list[0][indx]['model']
            layer_count = int(list(temp_dict.keys())[-1].split('.')[2]) + 1
            for i in range(1, len(file_list)):
                for (key, item) in file_list[i][indx]['model'].items():
                    new_key_list = key.split('.')[:2] + [str(int(key.split('.')[2])+layer_count)] + key.split('.')[3:] 
                    new_key = '.'.join(new_key_list)
                    temp_dict[new_key] = item
                layer_count = int(list(temp_dict.keys())[-1].split('.')[2])
            file_list[0][indx]['model'] = temp_dict
    return file_list[0]


def _tp_checkpoint(file_list, tp_rank, old_tp_size, new_tp_size, tp_ratio):
    '''
    in: [[],[],[],[]] or [[]]
    out: []
    '''
    concat_0 = ['query_key_value', 'dense_1']
    concat_1 = ['dense_2.layer.layer.weight', 'dense.layer.layer.weight', 'dense.layer.weight']
    print('-----tp-----')
    print('1 (tp)==' , len(file_list))
    print('2 (model, opti ..)==' , len(file_list[0]))
    print('model ==' , len(file_list[0]['model']))
    if new_tp_size >= old_tp_size:
        for (key, item) in file_list[0]['model'].items():
            if concat_0[0] in key or concat_0[1] in key:
                file_list[0]['model'][key] = torch.split(file_list[0]['model'][key], file_list[0]['model'][key].size(dim=0)//tp_ratio, 0)[tp_rank%tp_ratio]
            elif concat_1[0]  in key or concat_1[1] in key or concat_1[2] in key:
                file_list[0]['model'][key] = torch.split(file_list[0]['model'][key], file_list[0]['model'][key].size(dim=1)//tp_ratio, 1)[tp_rank%tp_ratio]
    else:
        for i in range(1, len(file_list)):
            for (key, item) in file_list[i]['model'].items():
                if concat_0[0] in key or concat_0[1] in key:
                    file_list[0]['model'][key] = torch.cat((file_list[0]['model'][key], item), 0)
                elif concat_1[0]  in key or concat_1[1] in key or concat_1[2] in key:
                    file_list[0]['model'][key] = torch.cat((file_list[0]['model'][key], item), 1)    
    return file_list[0]


def save_checkpoint(checkpoint_path: str,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    **kwargs):
    """Given a directory to store the checkpoints, saves all the training components' parameters or buffers, such as model,
     optimizer, lr_scheduler and etc. into a checkpoint dictionary.

    This method can be used for both colosalai nn.BaseModel and normal pytorch nn.Module.


    :param checkpoint_path: Set up a directory for saving checkpoints
    :type checkpoint_path: str
    :param epoch: Epoch number (indicate how many epochs have you trained this model)
    :type epoch: int
    :param model: Model to be registered
    :type model: torch.nn.Module
    :param optimizer: Optimizer to be registered
    :type optimizer: torch.optim.Optimizer
    :param lr_scheduler: lr_scheduler to be registered, defaults to None
    :type lr_scheduler: torch.optim.lr_scheduler._LRScheduler, optional
    """
    # for compatibility with normal pytorch nn.Module
    if hasattr(model, 'state_dict_for_save_checkpoint'):
        model_sd = model.state_dict_for_save_checkpoint()
    else:
        model_sd = model.state_dict()
    # ckpt container
    checkpoint = {'epoch': epoch, 'model': model_sd, 'optimizer': optimizer.state_dict(), **kwargs}
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    _ensure_directory_exists(checkpoint_path)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str,
                    old_tp_size: int,
                    old_pp_size: int,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    suffix: str = '',
                    finetune: bool = False,
                    strict: bool = True) -> Tuple:
    """Loads the checkpoint file.
    If finetune is False, then we intend to continue/resume the training process from the checkpoint given.
    So we copy parameters and buffers from state_dict into these modules(model, optimizer,lr_scheduler)
     and its descendants.
    If finetune is True, then only the weights and buffers of model should be reload.
    If strict is True, then the keys of state_dict must exactly match the keys returned by this module's
     state_dict() function.

    :param checkpoint_path: The exact and matched checkpoint_path directory to retrieve appropriate state_dict
    :type checkpoint_path: str
    :param model: Model to reload parameters and buffers
    :type model: torch.nn.Module
    :param optimizer: Optimizer to recuperate
    :type optimizer: torch.optim.Optimizer
    :param lr_scheduler: lr_scheduler to recuperate, defaults to None
    :type lr_scheduler: torch.optim.lr_scheduler._LRScheduler, optional
    :param finetune: Whether to finetune the model with new dataset or continue the pre-training, defaults to False
    :type finetune: bool, optional
    :param strict: Whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model., defaults to True
    :type strict: bool, optional
    :raises ValueError: Raise error if the model/optimizer cannot successfully be recuperated
    :return: (the epoch number of the checkpoint retrieved, the checkpoint retrieved)
    :rtype: Tuple

    """
    # Load the checkpoint.
    checkpoint = colossalai_load(checkpoint_path, old_tp_size, old_pp_size, epoch, suffix) # load
    try:
        last_epoch = checkpoint.pop('epoch') if not finetune else 0
        model.load_state_dict(checkpoint.pop('model'), strict=strict)
    except KeyError:
        raise ValueError('Checkpoint is corrupted')

    # if not finetune:
    #     try:
    #         optimizer.load_state_dict(checkpoint.pop('optimizer'))
    #     except KeyError:
    #         raise ValueError('Checkpoint is corrupted')

    #     if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
    #         lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
    # else:
    #     lr_scheduler = None
    #     optimizer = None
    #     assert lr_scheduler is None, "Optimizer and lr_scheduler should be None when finetune is true"
    #     assert optimizer is None , "Optimizer and lr_scheduler should be None when finetune is true"
    lr_scheduler = None
    optimizer = None
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        except ValueError:
            raise ValueError('Optimizer should be None to load parameters with a different setting, finetune or inference')
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
        except ValueError:
            raise ValueError('LR_scheduler should be None to load parameters with a different setting, finetune or inference')
        

    return last_epoch, checkpoint
