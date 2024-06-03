from enum import EnumMeta


class GeminiMemoryManager(object):
    def __init__(self, states_cls: EnumMeta):
        super().__init__()
        self.states_cls = states_cls
        self._cnter = 0  # the counter of instances

        self.total_mem = dict()
        self.state_mem = dict()
        self.state_mem["cpu"] = dict()
        self.state_mem["cuda"] = dict()

        self.reset()

    @property
    def total_number(self):
        return self._cnter

    def reset(self):
        self._cnter = 0  # the counter of instances

        self.total_mem["cpu"] = 0  # memory occupation of instances in cpu
        self.total_mem["cuda"] = 0  # memory of occupation of instances in cuda

        # memory conditions for all states
        for state in self.states_cls:
            self.state_mem["cpu"][state] = 0
            self.state_mem["cuda"][state] = 0

    def register_new_instance(self):
        self._cnter += 1

    def delete_instance(self):
        self._cnter -= 1

    def print_info(self):
        print(
            f"Total number: {self.total_number}",
            f"Total CPU memory occupation: {self.total_mem['cpu']}",
            f"Total CUDA memory occupation: {self.total_mem['cuda']}\n",
            sep="\n",
        )

        for state in self.states_cls:
            print(
                f"{state}: CPU memory occupation: {self.state_mem['cpu'][state]}",
                f"{state}: CUDA memory occupation: {self.state_mem['cuda'][state]}\n",
                sep="\n",
            )
