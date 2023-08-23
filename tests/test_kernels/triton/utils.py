import torch
import numpy as np


def benchmark(func, *args):
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300

    for i in range(10):
        func(*args)
    
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            func(*args)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    return mean_syn