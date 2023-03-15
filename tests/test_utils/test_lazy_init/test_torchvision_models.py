from utils import check_lazy_init

from tests.kit.model_zoo import model_zoo


def test_torchvision_models_lazy_init():
    sub_model_zoo = model_zoo.get_sub_registry('torchvision')
    for name, entry in sub_model_zoo.items():
        check_lazy_init(entry, verbose=True)


if __name__ == '__main__':
    test_torchvision_models_lazy_init()
