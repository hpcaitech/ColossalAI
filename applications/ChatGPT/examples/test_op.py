import torch

import colossalai
from colossalai.kernel.op_builder.cpu_adam import CPUAdamBuilder

builder = CPUAdamBuilder()
builder.load(verbose=True)
# print("before building")

    # builder = CPUAdamBuilder()
    # print("builder is created")

    # op = builder.load()

    # print("building completed")
