import re

import torch
from torchaudio_utils import trace_and_compare

from tests.kit.model_zoo import model_zoo


def test_torchaudio_models():
    torch.backends.cudnn.deterministic = True

    sub_model_zoo = model_zoo.get_sub_registry('torchaudio')

    for name, (model_fn, data_gen_fn, output_transform_fn, attribute) in sub_model_zoo.items():
        # FIXME(ver217): temporarily skip these models
        if re.search(f'(conformer|emformer|tacotron|wav2vec2_base|hubert_base)', name):
            continue
        model = model_fn()
        trace_and_compare(model,
                          data_gen_fn,
                          output_transform_fn,
                          need_meta=(attribute is not None and attribute.has_control_flow))
