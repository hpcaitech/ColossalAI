from colossalai.tensor import ColoParameter, ColoTensor
import torch
from numpy import allclose
from _utils import tensor_equal

def test_multiinheritance():
    colo_param = ColoParameter()
    assert isinstance(colo_param, ColoTensor)
    assert isinstance(colo_param, torch.nn.Parameter)

    # __deepcopy__ overload
    import copy
    colo_param2 = copy.deepcopy(colo_param)
    assert isinstance(colo_param2, ColoParameter)
    assert tensor_equal(colo_param.data, colo_param2.data)
    assert colo_param.requires_grad == colo_param2.requires_grad

    # __repr__ overload
    assert 'ColoParameter' in str(colo_param)

    # __torch_function__
    clone_param = torch.clone(colo_param)
    assert isinstance(clone_param, ColoTensor)

if __name__ == '__main__':
    test_multiinheritance()