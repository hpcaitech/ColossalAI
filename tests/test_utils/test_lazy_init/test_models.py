import pytest

from tests.kit.model_zoo import model_zoo

# FIXME(ver217): uncomment this line
# from utils import check_lazy_init


# FIXME(ver217): temporarily skip this test since torch 1.11 does not fully support meta tensor
@pytest.mark.skip
@pytest.mark.parametrize('subset', ['torchvision', 'diffusers', 'timm', 'transformers', 'torchaudio', 'deepfm', 'dlrm'])
def test_torchvision_models_lazy_init(subset):
    sub_model_zoo = model_zoo.get_sub_registry(subset)
    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in ('torchaudio_wav2vec2_base', 'torchaudio_hubert_base'):
            continue
        # FIXME(ver217): uncomment this line
        # check_lazy_init(entry, verbose=True)


if __name__ == '__main__':
    test_torchvision_models_lazy_init('torchvision')
