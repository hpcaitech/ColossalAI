import pytest

from colossalai.core import global_context as gpc
from colossalai.initialize import launch, get_default_parser
from test_layer import check_linear, check_layernorm, check_attention, check_mlp, check_transformerlayer
from test_operation import check_AB, check_ABT, check_ATB

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=8, mode='2.5d', depth=2),
    ),
)


def check_operations():
    check_AB()
    check_ABT()
    check_ATB()


def check_layer():
    check_linear()
    check_layernorm()
    check_attention()
    check_mlp()
    check_transformerlayer()


@pytest.mark.dist
@pytest.mark.skip("This test should be invoked by test.sh in the same folder as it runs on multiple gpus")
def test_2p5d():
    parser = get_default_parser()
    args = parser.parse_args()
    launch(config=CONFIG,
           rank=args.rank,
           world_size=args.world_size,
           host=args.host,
           port=args.port,
           backend=args.backend)
    check_layer()
    check_operations()
    gpc.destroy()


if __name__ == '__main__':
    test_2p5d()
