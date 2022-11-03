from colossalai.auto_parallel.tensor_shard.utils import get_alpha_beta


def test_get_alpha_beta():
    get_alpha_beta(2, 1)
