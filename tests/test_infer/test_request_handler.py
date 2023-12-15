from colossalai.inference.core.request_handler import RunningList


def test_running_list():
    running_list = RunningList(ratio=1.2)
    running_list.append()


if __name__ == "__main__":
    test_running_list()
    test_request_handler()
