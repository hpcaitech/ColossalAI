import torch
import torch.distributed as dist
from colossalai.tensor import ColoTensor


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
    # for p in model.parameters():
    #     if isinstance(p, ColoTensor):
    #         mapping[id(p)] = (p.dist_spec, p.compute_spec)
    #         p = p.to_replicate()

    for k, v in model.state_dict().items():
        if isinstance(v, ColoTensor):
            mapping[k] = (v.dist_spec, v.compute_spec)
            new_dict[k] = v.to_replicate()

    if dist.get_rank() == 0:
        for k, v in new_dict.items():
            if isinstance(v, ColoTensor):
                assert v.is_replicate(), f"{k} {v}"

        model_state = {'epoch': epoch, 'model': new_dict}
        torch.save(model_state, dire + '/epoch_{}_model.pth'.format(epoch))

    # for k, v in model.state_dict().items():
    #     if isinstance(v, ColoTensor):
    #         model.state_dict()[k].set_tensor_spec(*mapping[k])

    # optim_state = {'epoch': epoch, 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}

    # mapping = dict()
    # for p, v in optimizer.state.items():
    #     if isinstance(p, ColoTensor):
    #         mapping[id(p)] = (p.dist_spec, p.compute_spec)
    #         p = p.to_replicate()

    # if dist.get_rank() == 0:
    #     torch.save(optim_state, dire + '/epoch_{}_optim.pth'.format(epoch))

    # for p, v in optimizer.state.items():
    #     if isinstance(p, ColoTensor):
    #         if id(p) in mapping:
    #             p.set_tensor_spec(*mapping[id(p)])


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

    # mapping = dict()
    # for k, v in model.state_dict().items():
    #     if isinstance(v, ColoTensor):
    #         mapping[k] = (v.dist_spec, v.compute_spec)
    #         model.state_dict()[k] = v.to_replicate()

    model_state = torch.load(dire + '/epoch_{}_model.pth'.format(epoch))
    # model_state['model'] = collections.OrderedDict([(k.split('.', 1)[1], v) for k, v in model_state['model'].items()])
    model.load_state_dict(model_state['model'])

    # for k, v in model.state_dict().items():
    #     if isinstance(v, ColoTensor):
    #         model.state_dict()[k].set_tensor_spec(*mapping[k])

    # mapping = dict()
    # for p, v in optimizer.state.items():
    #     if isinstance(p, ColoTensor):
    #         mapping[id(p)] = (p.dist_spec, p.compute_spec)
    #         p = p.to_replicate()

    # optim_state = torch.load(dire + '/epoch_{}_optim.pth'.format(epoch))
    # optimizer.load_state_dict(optim_state['optimizer'])

    # for p, v in optimizer.state.items():
    #     if isinstance(p, ColoTensor):
    #         if id(p) in mapping:
    #             p.set_tensor_spec(*mapping[id(p)])
