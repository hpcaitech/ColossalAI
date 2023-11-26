#!/usr/bin/env python
# -*- encoding: utf-8 -*-

ALLOWED_MODES = [None, "1d", "2d", "2.5d", "3d", "sequence"]
TENSOR_PARALLEL_MODE = "tensor_parallel_mode"

# initializer
INITIALIZER_MAPPING = {
    "data": "Initializer_Data",
    "tensor": "Initializer_Tensor",
    "pipeline": "Initializer_Pipeline",
    "embedding": "Initializer_Embedding",
    "1d": "Initializer_1D",
    "2d": "Initializer_2D",
    "2.5d": "Initializer_2p5D",
    "3d": "Initializer_3D",
    "sequence": "Initializer_Sequence",
    "model": "Initializer_Model",
    "moe": "Initializer_Moe",
}

# 3D parallelism groups
INPUT_GROUP_3D = "input_group_3d"
WEIGHT_GROUP_3D = "weight_group_3d"
OUTPUT_GROUP_3D = "output_group_3d"
INPUT_X_WEIGHT_3D = "input_x_weight_group_3d"
OUTPUT_X_WEIGHT_3D = "output_x_weight_group_3d"

# Attributes of tensor parallel parameters
IS_TENSOR_PARALLEL = "is_tensor_parallel"
NUM_PARTITIONS = "num_partitions"
TENSOR_PARALLEL_ATTRIBUTES = [IS_TENSOR_PARALLEL, NUM_PARTITIONS]
