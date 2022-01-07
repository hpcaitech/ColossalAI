#!/usr/bin/python
#
# helper functions for 3D object detection evaluation
#

import os
import numpy as np

from typing import List

# matching methods
MATCHING_AMODAL = 0
MATCHING_MODAL = 1


class EvaluationParameters:
    """Helper class managing the evaluation parameters

    Attributes:
        labels_to_evaluate: list of labels to evaluate
        min_iou_to_match: min iou required to accept as TP
        max_depth: max depth for evaluation
        step_size: step/bin size for DDTP metrics
        matching_method: use modal or amodal 2D boxes for matching
        cw: working confidence. If set to -1, it will be determined automatically
        num_conf: number of different confidence thresholds used for AP calculation
    """

    def __init__(
        self,
        labels_to_evaluate,               # type: List[str]
        min_iou_to_match=0.7,             # type: float
        max_depth=100,                    # type: int
        step_size=5,                      # type: int
        matching_method=MATCHING_AMODAL,  # type: int
        cw=-1.,                           # type: float
        num_conf=50                       # type: int
    ):
        # type: (...) -> None

        self._labels_to_evaluate = labels_to_evaluate
        self._min_iou_to_match = min_iou_to_match
        self._max_depth = max_depth
        self._step_size = step_size
        self._matching_method = matching_method
        self._cw = cw
        self._num_conf = num_conf

    @property
    def labels_to_evaluate(self):
        return self._labels_to_evaluate

    @property
    def min_iou_to_match(self):
        return self._min_iou_to_match

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def step_size(self):
        return self._step_size

    @property
    def matching_method(self):
        return self._matching_method

    @property
    def cw(self):
        return self._cw

    @cw.setter
    def cw(self, cw):
        self._cw = cw

    @property
    def num_conf(self):
        return self._num_conf


def calcIouMatrix(
    gts,        # type: np.ndarray,
    preds       # type: np.ndarray
):
    # type: (...) -> np.ndarray
    """Calculates the pairwise Intersection Over Union (IoU)
    matrix for a set of GTs and predictions.

    Args:
        gts (np.ndarray): GT boxes with shape Mx4
        preds (np.ndarray): predictions with shape Nx4

    Returns:
        np.ndarray: IoU matrix with shape MxN
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(gts, 4, axis=1)
    xmin_2, ymin_2, xmax_2, ymax_2 = np.split(preds, 4, axis=1)

    inter_xmin = np.maximum(xmin_1, np.transpose(xmin_2))
    inter_ymin = np.maximum(ymin_1, np.transpose(ymin_2))
    inter_xmax = np.minimum(xmax_1, np.transpose(xmax_2))
    inter_ymax = np.minimum(ymax_1, np.transpose(ymax_2))

    inter_area = np.maximum((inter_xmax - inter_xmin + 1), 0) * np.maximum((inter_ymax - inter_ymin + 1), 0)

    area_1 = (xmax_1 - xmin_1 + 1) * (ymax_1 - ymin_1 + 1)
    area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
    iou = inter_area / (area_1 + np.transpose(area_2) - inter_area + 1e-10)

    return iou


def calcOverlapMatrix(
    gt_ignores,     # type: np.ndarray,
    preds           # type: np.ndarray
):
    # type: (...) -> np.ndarray
    """Calculates the overlap matrix for a set
    of GT ignore regions and predictions.

    Args:
        gt_ignores (np.ndarray): GT ignore regions with shape Mx4
        preds (np.ndarray): predictions with shape Nx4

    Returns:
        np.ndarray: overlap matrix with shape MxN
    """
    xmin_1, ymin_1, xmax_1, ymax_1 = np.split(gt_ignores, 4, axis=1)
    xmin_2, ymin_2, xmax_2, ymax_2 = np.split(preds, 4, axis=1)

    inter_xmin = np.maximum(xmin_1, np.transpose(xmin_2))
    inter_ymin = np.maximum(ymin_1, np.transpose(ymin_2))
    inter_xmax = np.minimum(xmax_1, np.transpose(xmax_2))
    inter_ymax = np.minimum(ymax_1, np.transpose(ymax_2))

    inter_area = np.maximum((inter_xmax - inter_xmin + 1), 0) * np.maximum((inter_ymax - inter_ymin + 1), 0)

    area_2 = (xmax_2 - xmin_2 + 1) * (ymax_2 - ymin_2 + 1)
    overlap = inter_area / (np.transpose(area_2) + 1e-10)

    return overlap


def getFiles(
    folder,                     # type: str
    suffix=".json",             # type: str
    exclude=["results.json"]    # type: List[str]
):
    # type: (...) -> List[str]
    """Recursively walks through the folder and finds
    returns all files that end with ``"suffix"``.

    Args:
        folder (str): the directory
        suffix (str): the suffix used for filtering
        exclude (List[str]): filenames to exclude

    Returns:
        List[str]: list of all found files
    """
    file_list = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(suffix) and f not in exclude:
                file_list.append(os.path.join(root, f))
    file_list.sort()

    return file_list
