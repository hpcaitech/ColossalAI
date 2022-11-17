import pytest

from colossalai.device import profile_alpha_beta


@pytest.mark.skip(reason="Skip because assertion fails for CI devices")
def test_profile_alpha_beta():
    physical_devices = [0, 1, 2, 3]
    (alpha, beta) = profile_alpha_beta(physical_devices)
    assert alpha > 0 and alpha < 1e-4 and beta > 0 and beta < 1e-10


if __name__ == '__main__':
    test_profile_alpha_beta()
