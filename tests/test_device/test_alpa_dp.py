from colossalai.device import get_submesh_choices


def test_alpa_dp():
    num_layers = 8
    num_hosts = 1
    num_devices_per_host = 8
    num_devices = num_hosts * num_devices_per_host
    num_micro_batches = 16
    num_autosharding_configs = 4    # Do we really need this?
    print(get_submesh_choices(num_hosts, num_devices_per_host))


if __name__ == '__main__':
    test_alpa_dp()
