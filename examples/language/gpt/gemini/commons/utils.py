import time

import torch


class DummyProfiler:
    def __init__(self):
        self.step_number = 0

    def step(self):
        self.step_number += 1


# Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_time_stamp():
    cur_time = time.strftime("%d-%H:%M", time.localtime())
    return cur_time
