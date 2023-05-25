from colossalai.elixir.simulator import move_count
from colossalai.testing import run_on_environment_flag


@run_on_environment_flag('ELX')
def test_move_count():
    steps = [[0], [1, 2], [3], [3], [1, 2], [0]]
    size = 2
    assert move_count(steps, size) == 12


if __name__ == '__main__':
    test_move_count()
