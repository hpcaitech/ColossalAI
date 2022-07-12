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

    if dist.get_rank() == 0:
        for k, v in new_dict.items():
            if isinstance(v, ColoTensor):
                assert v.is_replicate()

        model_state = {'epoch': epoch, 'model': new_dict}
        torch.save(model_state, dire + '/epoch_{}_model.pth'.format(epoch))

    # delete the dicts
    del new_dict
    del mapping

    if optimizer is None:
        return
    mapping = dict()
    new_opt_dict = dict()
    print(optimizer.state_dict())
    for k, v in optimizer.state_dict()['param_groups'][0]['params'].items():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            new_opt_dict[k] = v.to_replicate().detach()

    optimizer['param_groups']['params'] = new_opt_dict
    if dist.get_rank() == 0:
        print('optimize state dict', optimizer.state_dict())
        print('save new_opt_dict', new_opt_dict)
        optimzer_state = {'epoch': epoch, 'optimizer_state_dict': new_opt_dict}
        torch.save(optimzer_state, dire + '/epoch_{}_optim.pth'.format(epoch))

    del new_opt_dict


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
    for k, v in model.named_parameters():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            v.to_replicate_()

    model_state = torch.load(dire + '/epoch_{}_model.pth'.format(epoch))
    model.load_state_dict(model_state['model'])

    # reset tensors to original dist spec.
    with DistSpecManager.no_grad():
        for k, v in model.named_parameters():
            if isinstance(v, ColoTensor):
                v.set_tensor_spec(*mapping[k])

    del mapping

    if optimizer is None:
        return

    mapping = dict()
    for k, v in optimizer.state_dict().items():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            v.to_replicate_()

    optim_state = torch.load(dire + '/epoch_{}_optim.pth'.format(epoch))
    print(optim_state)
    optimizer.load_state_dict(optim_state['optimizer_state_dict'])

    # reset tensors to original dist spec.
    with DistSpecManager.no_grad():
        for k, v in optimizer.state_dict['param_groups']['params'].state_dict().items():
            if isinstance(v, ColoTensor):
                v.set_tensor_spec(*mapping[k])
