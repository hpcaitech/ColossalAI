# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import json
import torch
from enum import IntEnum

DEFAULT_INTERMEDIATE_SIZE = -1
class ActivationFuncType(IntEnum):
    UNKNOWN = 0
    ReLU = 1
    GELU = 2
    SiLU = 3
    GATED_GELU = 4
    GATED_SILU = 5


class CaiInferenceConfig():


    def __init__(self,
                 fp16=True,
                 gptq=False,
                 gptq_group_size=128,
                 gptq_quant_bits=4,
                 gptq_weight_dtype=torch.int64
                 ):
        self.fp16 = fp16
        self.gptq = gptq
        self.gptq_group_size = gptq_group_size
        self.gptq_quant_bits = gptq_quant_bits
        self.gptq_weight_dtype = gptq_weight_dtype


