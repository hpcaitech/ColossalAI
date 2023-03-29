import pytest
import torch
from packaging import version
from torchaudio_utils import trace_and_compare

from tests.kit.model_zoo import model_zoo


# We cannot handle the tensors constructed with constant during forward, such as ``torch.empty(0).to(device=Proxy.device)``
# TODO: We could handle this case by hijacking torch.Tensor.to function.
@pytest.mark.skip
def test_torchaudio_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry('torchaudio')

    for name, (model_fn, data_gen_fn, output_transform_fn, attribute) in sub_model_zoo.items():
        model = model_fn()
        trace_and_compare(model,
                          data_gen_fn,
                          output_transform_fn,
                          need_meta=(attribute is not None and attribute.has_control_flow))
