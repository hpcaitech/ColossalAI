#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from enum import Enum


# parallel modes
class ParallelMode(Enum):
    """This is an enumeration class containing all possible parallel modes."""

    GLOBAL = "global"

    # common parallel
    DATA = "data"

    # model parallel - containing tensor and pipeline parallel groups
    # this is added to facilitate amp and grad clipping in hybrid parallel
    MODEL = "model"

    # pipeline parallel
    PIPELINE = "pipe"

    # containing all ranks in tensor parallel
    TENSOR = "tensor"

    # sequence parallel
    SEQUENCE = "sequence"
    SEQUENCE_DP = "sequence_dp"

    # 1D Parallel
    PARALLEL_1D = "1d"

    # 2D parallel
    PARALLEL_2D_ROW = "2d_row"
    PARALLEL_2D_COL = "2d_col"

    # 3D parallel
    PARALLEL_3D_INPUT = "3d_input"
    PARALLEL_3D_WEIGHT = "3d_weight"
    PARALLEL_3D_OUTPUT = "3d_output"
    PARALLEL_3D_INPUT_X_WEIGHT = "3d_input_x_weight"
    PARALLEL_3D_OUTPUT_X_WEIGHT = "3d_output_x_weight"

    # 2.5D parallel
    PARALLEL_2P5D_ROW = "2p5d_row"
    PARALLEL_2P5D_COL = "2p5d_col"
    PARALLEL_2P5D_DEP = "2p5d_dep"
    PARALLEL_2P5D_XZ = "2p5d_xz"
