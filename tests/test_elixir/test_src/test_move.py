from colossalai.kernel.op_builder import ElixirSimulatorBuilder
from colossalai.testing import run_on_environment_flag


@run_on_environment_flag('ELX')
def test_move_count():
    steps = [[0], [1, 2], [3], [3], [1, 2], [0]]
    size = 2
    simulator = ElixirSimulatorBuilder().load()
    assert simulator.move_count(steps, size) == 12


if __name__ == '__main__':
    test_move_count()
