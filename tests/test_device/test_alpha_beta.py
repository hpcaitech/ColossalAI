from colossalai.device import profile_alpha_beta


def test_profile_alpha_beta():
    physical_devices = [0, 1, 2, 3]
    (alpha, beta) = profile_alpha_beta(physical_devices)
    print((alpha, beta))


if __name__ == '__main__':
    test_profile_alpha_beta()
