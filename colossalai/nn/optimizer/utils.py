class CpuAdamCounter(object):
    """Used to record the total number of CPU Adam.
    We must use it to avoid hybrid cpu adam and cpu adam using the same id.
    """

    def __init__(self):
        self.number = 0

    def __call__(self):
        self.number += 1
        return self.number - 1


CPU_ADAM_CNT = CpuAdamCounter()
