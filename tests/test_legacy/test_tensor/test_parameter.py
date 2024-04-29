import pytest
import torch
from common_utils import tensor_equal

import colossalai
from colossalai.tensor import ColoParameter, ColoTensor
from colossalai.testing import free_port


@pytest.mark.skip
def test_multiinheritance():
    colossalai.legacy.launch(rank=0, world_size=1, host="localhost", port=free_port(), backend="nccl")
    colo_param = ColoParameter(None, requires_grad=True)
    assert colo_param.dist_spec.placement.value == "r"
    assert isinstance(colo_param, ColoTensor)
    assert isinstance(colo_param, torch.nn.Parameter)

    # __deepcopy__ overload
    import copy

    colo_param2 = copy.deepcopy(colo_param)
    assert isinstance(colo_param2, ColoParameter)
    assert tensor_equal(colo_param.data, colo_param2.data)
    assert colo_param.requires_grad == colo_param2.requires_grad

    # __repr__ overload
    assert "ColoParameter" in str(colo_param)

    # __torch_function__
    clone_param = torch.clone(colo_param)
    assert isinstance(clone_param, ColoTensor)


if __name__ == "__main__":
    test_multiinheritance()
