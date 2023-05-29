from colossalai.elixir.ctx import MetaContext
from colossalai.testing import run_on_environment_flag
from tests.test_elixir.utils import TEST_MODELS


@run_on_environment_flag('ELX')
def test_meta_context():
    builder, *_ = TEST_MODELS.get('resnet')
    with MetaContext():
        model = builder()

    for name, param in model.named_parameters():
        assert param.device.type == 'meta'
        print(name, param)

    for name, buffer in model.named_buffers():
        assert buffer.device.type == 'meta'
        print(name, buffer)


if __name__ == '__main__':
    test_meta_context()
