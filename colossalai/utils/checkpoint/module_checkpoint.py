import torch
import torch.distributed as dist
from colossalai.tensor import ColoTensor, DistSpecManager


def save_checkpoint(dire: str,
                    epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    *args,
                    **kwargs):
    """save_checkpoint 
    save a model, whose parameters are `ColoTensor`s.
    Args:
        dire (str): directory to save the checkpoint files.
        epoch (int): the number of epoch
        model (torch.nn.Module): a torch module initialized by ColoInitContext
        optimizer (torch.optim.Optimizer, optional): optimizers. Defaults to None.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): lr schedule. Defaults to None.
    """

    mapping = dict()
    new_dict = dict()

    # save the dist context about the tensors in a new dict, while still maintain the original dict.
    for k, v in model.state_dict().items():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            new_dict[k] = v.to_replicate().detach()
        else:
            new_dict[k] = v
    if dist.get_rank() == 0:
        for k, v in new_dict.items():
            if isinstance(v, ColoTensor):
                assert v.is_replicate()

        model_state = {'epoch': epoch, 'model': new_dict}
        torch.save(model_state, dire + '/epoch_{}_model.pth'.format(epoch))

    # delete the new dict
    del new_dict


def load_checkpoint(dire,
                    epoch: int,
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

    mapping = dict()
    for k, v in model.state_dict().items():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            v.to_replicate_()

    model_state = torch.load(dire + '/epoch_{}_model.pth'.format(epoch))
    model.load_state_dict(model_state['model'])

    # reset tensors to original dist spec.
    with DistSpecManager.no_grad():
        for k, v in model.state_dict().items():
            if isinstance(v, ColoTensor):
                v.set_tensor_spec(*mapping[k])
