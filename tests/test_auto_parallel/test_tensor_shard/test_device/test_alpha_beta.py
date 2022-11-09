from colossalai.device import get_alpha_beta


def test_get_alpha_beta():
    physical_devices = [0, 1, 2, 3]
    get_alpha_beta(1, 4, physical_devices)


if __name__ == '__main__':
    test_get_alpha_beta()
