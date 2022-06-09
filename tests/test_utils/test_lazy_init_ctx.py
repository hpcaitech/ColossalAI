import torch
import torch.nn as nn
from colossalai.utils.model.lazy_init_context import LazyInitContext

def test_lazy_init_ctx():

    with LazyInitContext() as ctx:
        model = nn.Linear(10, 10)
        model.weight.zero_()
    
    # make sure the weight is a meta tensor
    assert model.weight.is_meta
    
    # initialize weights
    ctx.lazy_init_parameters(model)
    
    # make sure the weight is not a meta tensor 
    # and initialized correctly
    assert not model.weight.is_meta and torch.all(model.weight == 0)


if __name__ == '__main__':
    test_lazy_init_ctx()
