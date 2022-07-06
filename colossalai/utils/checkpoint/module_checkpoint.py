import torch
import torch.nn as nn
import torch.distributed as dist
import collections
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from colossalai.utils.model.colo_init_context import colo_state_dict

def save_checkpoint(dire,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    *args,
                    **kwargs):
    """save_checkpoint 
    save a model, whose parameters are `ColoTensor`s.
    Args:
        dire (_type_): _description_
        epoch (int): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer, optional): _description_. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): _description_. Defaults to None.
    """
    model_state = {
        'epoch': epoch,
        'model': colo_state_dict(model, state_dict_func=nn.Module.state_dict)
    }
    if dist.get_rank() == 0:
        torch.save(model_state, dire + '/epoch_{}_model.pth'.format(epoch))
    lr_scheduler_dict = lr_scheduler.state_dict()
    lr_scheduler_dict['after_scheduler'] = lr_scheduler_dict['after_scheduler'].state_dict()
    optim_state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler_dict
    }
    torch.save(optim_state, dire + '/epoch_{}_optim_rank_{}.pth'.format(epoch, dist.get_rank()))




def load_checkpoint(dire,
                    epoch: int,
                    rank: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    *args,
                    **kwargs):
    """load_checkpoint 
    load a model, whose parameters are `ColoTensor`s.
    Args:
        dire (_type_): _description_
        epoch (int): _description_
        rank (int): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer, optional): _description_. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): _description_. Defaults to None.
    """
    model_state = torch.load(dire + '/epoch_{}_model.pth'.format(epoch))
    model_state['model'] = collections.OrderedDict([(k.split('.', 1)[1], v) for k, v in model_state['model'].items()])
    model.load_state_dict(model_state['model'])
    optim_state = torch.load(dire + '/epoch_{}_optim_rank_{}.pth'.format(epoch, rank))
    optimizer.load_state_dict(optim_state['optimizer'])
    lr_scheduler_dict = optim_state['lr_scheduler']
    after_scheduler_dict = lr_scheduler_dict['after_scheduler']
    lr_scheduler_dict['after_scheduler'] = _CosineAnnealingLR(
        optimizer, 
        after_scheduler_dict['T_max'],
        after_scheduler_dict['eta_min'],
        after_scheduler_dict['last_epoch']
        )
    lr_scheduler.load_state_dict(lr_scheduler_dict)
