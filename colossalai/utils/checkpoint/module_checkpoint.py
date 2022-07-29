import torch
import torch.distributed as dist
from colossalai.tensor import ColoTensor
from colossalai.nn.optimizer import ColossalaiOptimizer
from colossalai.utils.checkpoint.utils import gather_tensor, scatter_tensor
from typing import Optional, Dict


def save_checkpoint(path: str,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: Optional[ColossalaiOptimizer] = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    *args,
                    **kwargs):
    """save_checkpoint 
    save a model, whose parameters are `ColoTensor`s.
    Args:
        path (str): directory to save the checkpoint files.
        epoch (int): the number of epoch
        model (torch.nn.Module): a torch module initialized by ColoInitContext
        optimizer (ColossalaiOptimizer, optional): optimizers. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): lr schedule. Defaults to None.
    """
    rank = dist.get_rank()
    model_state = model.state_dict()
    # save the dist context about the tensors in a new dict, while still maintain the original dict.
    for k, v in model_state.items():
        if isinstance(v, ColoTensor):
            gather_tensor(v)    # gather shared tensors to rank0
            # don't recover tensors in rank0, since the dict is only a copy of model

    if rank == 0:
        # sanity check
        for k, v in model_state.items():
            if isinstance(v, ColoTensor):
                assert v.save_ready
                assert v.is_replicate()
                delattr(v, 'save_ready')
        # model saving
        save_state = {'epoch': epoch, 'model': model_state}
        torch.save(save_state, path + '/epoch_{}_model.pth'.format(epoch), *args, **kwargs)

    # delete old dicts
    del model_state
    # synchronize all the processes
    dist.barrier()

    if optimizer is not None:
        mapping = dict()
        optim_state = optimizer.state_dict()
        for k, v in optim_state['state'].items():
            for n, t in v.items():
                if isinstance(t, ColoTensor):
                    mapping[(k, n)] = t.dist_spec
                    gather_tensor(t)

        if rank == 0:
            save_state = {'epoch': epoch, 'optim': optim_state}
            torch.save(save_state, path + '/epoch_{}_optim.pth'.format(epoch), *args, **kwargs)
            # recover colo tensors in rank0
            for k, v in optimizer.state_dict()['state'].items():
                for n, t in v.items():
                    if isinstance(t, ColoTensor):
                        assert hasattr(t, 'save_ready')
                        t.set_dist_spec(mapping[(k, n)])
                        delattr(t, 'save_ready')

        del optim_state
        del mapping
        dist.barrier()


def load_checkpoint(path: str,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: Optional[ColossalaiOptimizer] = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    torch_load_kwargs: Optional[Dict] = None,
                    load_state_dict_kwargs: Optional[Dict] = None):
    """load_checkpoint 
    load a model, whose parameters are `ColoTensor`s.
    Args:
        path (str): directory to save the checkpoint files.
        epoch (int): the number of epoch
        model (torch.nn.Module): a torch module initialized by ColoInitContext
        optimizer (ColossalaiOptimizer, optional): optimizers. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): lr schedule. Defaults to None.
        torch_load_kwargs: (dict, optional): The kwargs of torch.load inside the function
        load_state_dict_kwargs (dict, optional): The kwargs of load_state_dict inside the function
    """
    # initialize the default paramters
    if not torch_load_kwargs:
        torch_load_kwargs = dict()
    if not load_state_dict_kwargs:
        load_state_dict_kwargs = dict()

    rank = dist.get_rank()
    mapping = dict()
    for n, p in model.named_parameters():
        if isinstance(p, ColoTensor):
            mapping[n] = p.dist_spec
            gather_tensor(p)

    if rank == 0:
        load_state = torch.load(path + '/epoch_{}_model.pth'.format(epoch), **torch_load_kwargs)
        model.load_state_dict(load_state['model'], **load_state_dict_kwargs)
    dist.barrier()

    # scatter loaded parameters
    for n, p in model.named_parameters():
        if isinstance(p, ColoTensor):
            scatter_tensor(p, mapping[n])
            if rank == 0:
                assert hasattr(p, 'save_ready')
                delattr(p, 'save_ready')
    del mapping

    if optimizer is not None:
        mapping = dict()
        for k, v in optimizer.state_dict()['state'].items():
            for n, t in v.items():
                if isinstance(t, ColoTensor):
                    mapping[(k, n)] = t.dist_spec
                    gather_tensor(t)

        if rank == 0:
            colo_checkpoint = torch.load(path + '/epoch_{}_optim.pth'.format(epoch), **torch_load_kwargs)
            optimizer.load_state_dict(colo_checkpoint['optim'], **load_state_dict_kwargs)
        dist.barrier()

        for k, v in optimizer.state_dict()['state'].items():
            for n, t in v.items():
                if isinstance(t, ColoTensor):
                    scatter_tensor(t, mapping[(k, n)])

        del mapping
