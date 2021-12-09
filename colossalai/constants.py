#!/usr/bin/env python
# -*- encoding: utf-8 -*-

ALLOWED_MODES = [None, '1d', '2d', '2.5d', '3d', 'sequence']

# intializer
INITIALIZER_MAPPING = {
    'data': 'Initializer_Data',
    'tensor': 'Initializer_Tensor',
    'pipeline': 'Initializer_Pipeline',
    'embedding': 'Initializer_Embedding',
    '1d': 'Initializer_1D',
    '2d': 'Initializer_2D',
    '2.5d': 'Initializer_2p5D',
    '3d': 'Initializer_3D',
    'sequence': 'Initializer_Sequence'
}

# 2D paralllel
SUMMA_DIM = 'SUMMA_DIM'

# 2.5D paralllel
TESSERACT_DIM = 'TESSERACT_DIM'
TESSERACT_DEP = 'TESSERACT_DEP'

# 3D parallel
DEPTH_3D = 'DEPTH_3D'
INPUT_GROUP_3D = 'PARALLEL_3D_INPUT'
WEIGHT_GROUP_3D = 'PARALLEL_3D_WEIGHT'
OUTPUT_GROUP_3D = 'PARALLEL_3D_OUTPUT'

# Tensor parallel attributes
IS_TENSOR_PARALLEL = 'is_tensor_parallel'
NUM_PARTITIONS = 'num_partitions'
TENSOR_PARALLEL_ATTRIBUTES = [IS_TENSOR_PARALLEL, NUM_PARTITIONS]
