import pytest
import torch
from packaging import version
from torchaudio_utils import trace_and_compare

from tests.kit.model_zoo import model_zoo


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.12.0'), reason='torch version < 12')
def test_torchaudio_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry('torchaudio')

    for name, (model_fn, data_gen_fn, output_transform_fn, attribute) in sub_model_zoo.items():
        model = model_fn()
        trace_and_compare(model,
                          data_gen_fn,
                          output_transform_fn,
                          need_meta=(attribute is not None and attribute.has_control_flow))
