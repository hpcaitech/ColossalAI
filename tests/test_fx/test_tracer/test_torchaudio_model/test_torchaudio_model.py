import pytest
import torch
from torchaudio_utils import trace_and_compare

from colossalai.testing import clear_cache_before_run
from tests.kit.model_zoo import model_zoo


# We cannot handle the tensors constructed with constant during forward, such as ``torch.empty(0).to(device=Proxy.device)``
# TODO: We could handle this case by hijacking torch.Tensor.to function.
@pytest.mark.skip
@clear_cache_before_run()
def test_torchaudio_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry("torchaudio")

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _, attribute) in sub_model_zoo.items():
        model = model_fn()
        trace_and_compare(
            model, data_gen_fn, output_transform_fn, need_meta=(attribute is not None and attribute.has_control_flow)
        )
