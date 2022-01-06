#!/usr/bin/python
#
# The evaluation script for Cityscapes 3D object detection (https://arxiv.org/abs/2006.07864)
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# The evaluation script expects one json annotation file per image with the format:
# {
#     "objects": [
#         {
#             "2d": {
#                 "modal": [xmin, ymin, w, h],
#                 "amodal": [xmin, ymin, w, h]
#             },
#             "3d": {
#                 "center": [x, y, z],
#                 "dimensions": [length, width, height],
#                 "rotation": [q1, q2, q3, q4],
#             },
#             "label": str,
#             "score": float
#         }
#     ]
# }
#
# Note: ["2d"]["modal"] and ["2d"]["amodal"] values are
# clipped to the image dimensions.
#
# Note: ["2d"]["modal"] is optional. If not provided,
# ["2d"]["amodal"] is used for both type of boxes.
#
# Note: For images without a single predicted box, you still need to provide
# a json file with content: {"objects": []}

# python imports
import coloredlogs
import logging
import numpy as np
import json
import os
import argparse
from typing import (
    List,
    Tuple
)

from pyquaternion import Quaternion
from tqdm import tqdm
# keep compatibility for python2
from collections import OrderedDict

from cityscapesscripts.helpers.annotation import (
    CsBbox3d,
    CsIgnore2d
)
from cityscapesscripts.helpers.box3dImageTransform import (
    Box3dImageTransform,
    Camera
)
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    EvaluationParameters,
    getFiles,
    calcIouMatrix,
    calcOverlapMatrix
)
from cityscapesscripts.evaluation.objectDetectionHelpers import (
    MATCHING_MODAL,
    MATCHING_AMODAL
)

logger = logging.getLogger('EvalObjectDetection3d')
logging.basicConfig(filename='eval.log',
                    filemode='w',
                    format='%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
coloredlogs.install(level='INFO')


class Box3dEvaluator:
    """The Box3dEvaluator object contains the data as well as the parameters
    for the evaluation of the dataset.

    :param eval_params: evaluation params including max depth, min iou etc.
    :type eval_params: EvaluationParameters
    :param gts: all GT annotations per image
    :type gts: dict
    :param preds: all GT annotations per image
    :type preds: dict
    :param ap: data for Average Precision (AP) calculation
    :type ap: dict
    :param results: evaluation results
    :type results: dict
    """

    def __init__(
        self,
        evaluation_params       # type: EvaluationParameters
    ):
        # type: (...) -> None

        self.eval_params = evaluation_params

        # dict containing the GTs per image
        self.gts = {}

        # dict containing the Camera object per image
        self.cameras = {}

        # dict containing the predictions per image
        self.preds = {}

        # dict containing information for AP per class
        self.ap = {}

        # dict containing all required results
        self.results = OrderedDict()

        # internal dict keeping additional statistics
        self._stats = OrderedDict()

        # the actual confidence thresholds
        self._conf_thresholds = np.arange(
            0.0, 1.01, 1.0 / self.eval_params.num_conf
        )

        # the actual depth bins
        self._depth_bins = range(0, self.eval_params.max_depth + 1, self.eval_params.step_size)

    def reset(self):
        # type: (...) -> None
        """Resets state of this instance to a newly initialized one."""

        self.gts = {}
        self.preds = {}
        self._stats = OrderedDict()
        self.ap = {}
        self.results = OrderedDict()

    def checkCw(self):
        # type: (...) -> None
        """Checks chosen working confidence value."""
        if (
            self.eval_params.cw not in self._conf_thresholds and
            self.eval_params.cw != -1.0
        ):
            old_cw = self.eval_params.cw
            # set 0 and 1 as lower and upper bound
            if old_cw < 0.0:
                self.eval_params.cw = 0.0
            elif old_cw > 1.0:
                self.eval_params.cw = 1.0
            else:  # determine closest possible confidence
                self.eval_params.cw = min(
                    filter(lambda c: c >= self.eval_params.cw, self._conf_thresholds)
                )

            logger.warning(
                "{:.2f} is used as working confidence instead of {}.".format(self.eval_params.cw, old_cw)
            )

    def loadGT(
        self,
        gt_folder   # type: str
    ):
        # type: (...) -> None
        """Loads ground truth from the given folder.

        Args:
            gt_folder (str): Ground truth folder
        """

        logger.info("Loading GT...")
        gts = getFiles(gt_folder)

        logger.info("Found {} GT files.".format(len(gts)))

        self._stats["GT_stats"] = OrderedDict((x, 0) for x in self.eval_params.labels_to_evaluate)

        for p in gts:
            gts_for_image = []
            ignores_for_image = []

            # extract CITY_RECORDID_IMAGE from filepath
            base = os.path.basename(p)
            base = base[:base.rfind("_")]

            # check for valid json file
            try:
                with open(p) as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.error("Invalid GT json file: {}".format(base))
                raise

            # check for 'objects' and 'sensor'
            if "objects" not in data.keys():
                msg = "'objects' missing in GT json file: {}".format(base)
                logger.error(msg)
                raise KeyError(msg)
            if "sensor" not in data.keys():
                msg = "'sensor' missing in GT json file: {}".format(base)
                logger.error(msg)
                raise KeyError(msg)

            # load Camera object
            camera = Camera(
                data["sensor"]["fx"],
                data["sensor"]["fy"],
                data["sensor"]["u0"],
                data["sensor"]["v0"],
                data["sensor"]["sensor_T_ISO_8855"]
            )

            # load 3D boxes
            for d in data["objects"]:
                if d["label"] in self.eval_params.labels_to_evaluate:
                    self._stats["GT_stats"][d["label"]] += 1
                    box_data = CsBbox3d()
                    box_data.fromJsonText(d)
                    gts_for_image.append(box_data)

            # load ignore regions
            for d in data["ignore"]:
                box_data = CsIgnore2d()
                box_data.fromJsonText(d)
                ignores_for_image.append(box_data)

            self.gts[base] = {
                "objects": gts_for_image,
                "ignores": ignores_for_image
            }

            self.cameras[base] = camera

    def loadPredictions(
        self,
        pred_folder   # type: str
    ):
        # type: (...) -> None
        """Loads all predictions from the given folder.

        Args:
            pred_folder (str): Prediction folder
        """

        logger.info("Loading predictions...")
        predictions = getFiles(pred_folder)

        predictions.sort()
        logger.info("Found {} prediction files.".format(len(predictions)))

        for p in predictions:
            preds_for_image = []

            # extract CITY_RECORDID_IMAGE from filepath
            base = os.path.basename(p)
            base = base[:base.rfind("_")]

            # check for valid json file
            try:
                with open(p) as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                logger.error("Invalid prediction json file: {}".format(base))
                raise

            # check for 'objects'
            if "objects" not in data.keys():
                logger.error("'objects' missing in prediction json file: {}".format(base))
                raise

            for d in data["objects"]:
                if (
                    "label" in d.keys() and
                    d["label"] in self.eval_params.labels_to_evaluate
                ):
                    try:
                        box_data = CsBbox3d()
                        box_data.fromJsonText(d)
                    except Exception:
                        logger.critical("Found incorrect annotation in {}.".format(p))
                        continue

                    preds_for_image.append(box_data)

            self.preds[base] = {
                "objects": preds_for_image
            }

    def evaluate(self):
        # type: (...) -> None
        """Main evaluation routine."""

        # fill up predictions dict with empty detections if prediction file not found
        for base in self.gts.keys():
            if base not in self.preds.keys():
                logger.critical(
                    "Could not find any prediction for image {}.".format(base))
                self.preds[base] = {"objects": []}

        # initialize empty data
        for s in self._conf_thresholds:
            self._stats[s] = {}
            self._stats[s]["data"] = {}

        logger.info("Evaluating images...")
        # calculate stats for each image
        self._calcImageStats()

        logger.info("Calculate AP...")
        # calculate 2D ap
        self._calculateAp()

        logger.info("Calculate TP stats...")
        # calculate TP stats (center dist, size similarity, orientation score)
        self._calcTpStats()

    def saveResults(
        self,
        result_folder   # type: str
    ):
        # type: (...) -> str
        """Saves the evaluation results to ``"results.json"``

        Args:
            result_folder (str): directory in which the result files are saved

        Returns:
            str: filepath of ``"results.json"``
        """

        result_file = os.path.join(result_folder, "results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=4)

        # dump internal stats for debugging
        # stats_file = os.path.join(result_folder, "stats.json")
        # with open(stats_file, 'w') as f:
        #    json.dump(self._stats, f, indent=4)

        return result_file

    def _calcImageStats(self):
        # type: (...) -> None
        """Internal method that calculates Precision and Recall values for whole dataset."""

        # single threaded
        results = []
        for x in tqdm(self.gts.keys()):
            results.append(self._worker(x))

        # update internal result dict with the corresponding results
        for thread_result in results:
            for score, eval_data in thread_result.items():
                data = eval_data["data"]
                for img_base, match_data in data.items():
                    self._stats[score]["data"][img_base] = match_data

    def _worker(
        self,
        base    # type: str
    ):
        # type: (...) -> dict
        """Internal method to run evaluation for a single image."""
        tmp_stats = {}

        gt_boxes = self.gts[base]
        pred_boxes = self.preds[base]
        camera = self.cameras[base]

        # recalculate the amodal bounding boxes
        box3dTransform = Box3dImageTransform(camera)

        for p in pred_boxes["objects"]:
            box3dTransform.initialize_box_from_annotation(p)
            p.bbox_2d.setAmodalBox(box3dTransform.get_amodal_box_2d())

        # calculate PR stats for each conf threshold
        for s in self._conf_thresholds:
            tmp_stats[s] = {
                "data": {}
            }
            (tp_idx_gt, tp_idx_pred, fp_idx_pred,
             fn_idx_gt) = self._addImageEvaluation(gt_boxes, pred_boxes, s)

            assert len(tp_idx_gt) == len(tp_idx_pred)

            tmp_stats[s]["data"][base] = {
                "tp_idx_gt": tp_idx_gt,
                "tp_idx_pred": tp_idx_pred,
                "fp_idx_pred": fp_idx_pred,
                "fn_idx_gt": fn_idx_gt
            }

        return tmp_stats

    def _addImageEvaluation(
        self,
        gt_boxes,    # type: List[CsBbox3d]
        pred_boxes,  # type: List[CsBbox3d]
        min_score    # type: float
    ):
        # type: (...) -> Tuple[dict, dict, dict, dict]
        """Internal method to evaluate a single image.

        Args:
            gt_boxes (List[CsBbox3d]): GT boxes
            pred_boxes (List[CsBbox3d]): Predicted boxes
            min_score (float): minimum required score

        Returns:
            tuple(dict, dict, dict, dict): tuple of TP, FP and FN data
        """
        tp_idx_gt = {}
        tp_idx_pred = {}
        fp_idx_pred = {}
        fn_idx_gt = {}

        # pre-load all ignore regions as they are the same for all classes
        gt_idx_ignores = [idx for idx,
                          box in enumerate(gt_boxes["ignores"])]

        # calculate stats per class
        for i in self.eval_params.labels_to_evaluate:
            # get idx for pred boxes for current class
            pred_idx = [idx for idx, box in enumerate(
                pred_boxes["objects"]) if box.label == i and box.score >= min_score]

            # get idx for gt boxes for current class
            gt_idx = [idx for idx, box in enumerate(
                gt_boxes["objects"]) if box.label == i]

            # if there is no prediction at all, just return an empty result
            if len(pred_idx) == 0:
                # dump data to result dicts
                tp_idx_gt[i] = []
                tp_idx_pred[i] = []
                fp_idx_pred[i] = pred_idx
                fn_idx_gt[i] = gt_idx
                continue

            # create 2D box matrix for predictions and gts
            boxes_2d_pred = np.zeros((0, 4))
            if len(pred_idx) > 0:
                # get modal or amodal boxes depending on matching strategy
                if self.eval_params.matching_method == MATCHING_AMODAL:
                    boxes_2d_pred = np.asarray(
                        [pred_boxes["objects"][x].bbox_2d.bbox_amodal for x in pred_idx])
                elif self.eval_params.matching_method == MATCHING_MODAL:
                    boxes_2d_pred = np.asarray(
                        [pred_boxes["objects"][x].bbox_2d.bbox_modal for x in pred_idx])
                else:
                    raise ValueError("Matching method {} not known!".format(self.eval_params.matching_method))

            boxes_2d_gt = np.zeros((0, 4))
            if len(gt_idx) > 0:
                # get modal or amodal boxes depending on matching strategy
                if self.eval_params.matching_method == MATCHING_AMODAL:
                    boxes_2d_gt = np.asarray(
                        [gt_boxes["objects"][x].bbox_2d.bbox_amodal for x in gt_idx])
                elif self.eval_params.matching_method == MATCHING_MODAL:
                    boxes_2d_gt = np.asarray(
                        [gt_boxes["objects"][x].bbox_2d.bbox_modal for x in gt_idx])
                else:
                    raise ValueError("Matching method {} not known!".format(self.eval_params.matching_method))

            boxes_2d_gt_ignores = np.zeros((0, 4))
            if len(gt_idx_ignores) > 0:
                boxes_2d_gt_ignores = np.asarray(
                    [gt_boxes["ignores"][x].bbox for x in gt_idx_ignores])

            # calculate IoU matrix between GTs and Preds
            iou_matrix = calcIouMatrix(boxes_2d_gt, boxes_2d_pred)

            # get matches
            (gt_tp_row_idx, pred_tp_col_idx, _) = self._getMatches(iou_matrix)

            # convert it to box idx
            gt_tp_idx = [gt_idx[x] for x in gt_tp_row_idx]
            pred_tp_idx = [pred_idx[x] for x in pred_tp_col_idx]
            gt_fn_idx = [x for x in gt_idx if x not in gt_tp_idx]
            pred_fp_idx_check_for_ignores = [
                x for x in pred_idx if x not in pred_tp_idx]

            # check if remaining FP idx match with ignored GT
            boxes_2d_pred_fp = np.zeros((0, 4))
            if len(pred_fp_idx_check_for_ignores) > 0:
                # as there are no amodal boxes for ignore regions
                # matching with ignore regions should only be performed on
                # modal predictions.
                boxes_2d_pred_fp = np.asarray(
                    [pred_boxes["objects"][x].bbox_2d.bbox_modal for x in pred_fp_idx_check_for_ignores])

            overlap_matrix = calcOverlapMatrix(
                boxes_2d_gt_ignores, boxes_2d_pred_fp)

            # get matches and convert to actual box idx
            (_, pred_tp_col_idx, _) = self._getMatches(overlap_matrix, matchIgnores=True)
            pred_tp_ignores_idx = [
                pred_fp_idx_check_for_ignores[x] for x in pred_tp_col_idx]
            pred_fp_idx = [
                x for x in pred_fp_idx_check_for_ignores if x not in pred_tp_ignores_idx]

            # dump data to result dicts
            tp_idx_gt[i] = gt_tp_idx
            tp_idx_pred[i] = pred_tp_idx
            fp_idx_pred[i] = pred_fp_idx
            fn_idx_gt[i] = gt_fn_idx

        return (tp_idx_gt, tp_idx_pred, fp_idx_pred, fn_idx_gt)

    def _getMatches(
        self,
        iou_matrix,         # type: np.ndarray
        matchIgnores=False  # type: bool
    ):
        # type: (...) -> Tuple[List[int], List[int], List[int]]
        """Internal method that gets the TP matches between the predictions and the GT data.

        Args:
            iou_matrix (np.ndarray): The NxM matrix containing the pairwise overlap or IoU
            matchIgnores (bool): If set to True, allow multiple matches with ignore regions

        Returns:
            tuple(list[int],list[int],list[float]): A tuple containing the TP indices
            for GT and predictions and the corresponding iou
        """
        matched_gts = []
        matched_preds = []
        matched_ious = []

        # we either have gt and no predictions or no predictions but gt
        if iou_matrix.shape[0] == 0 or iou_matrix.shape[1] == 0:
            return [], [], []

        # iteratively select the max of the iou_matrix and set the corresponding
        # rows and cols to 0.
        tmp_iou_max = np.max(iou_matrix)

        while tmp_iou_max > self.eval_params.min_iou_to_match:
            tmp_row, tmp_col = np.where(iou_matrix == tmp_iou_max)

            used_row = tmp_row[0]
            used_col = tmp_col[0]

            matched_gts.append(used_row)
            matched_preds.append(used_col)
            matched_ious.append(np.max(iou_matrix))

            if matchIgnores is False:
                iou_matrix[used_row, ...] = 0.0

            iou_matrix[..., used_col] = 0.0

            tmp_iou_max = np.max(iou_matrix)

        return (matched_gts, matched_preds, matched_ious)

    def _calcCenterDistances(
        self,
        label,       # type: str
        gt_boxes,    # type: List[CsBbox3d]
        pred_boxes,  # type: List[CsBbox3d]
    ):
        # type: (...) -> np.ndarray
        """Internal method that calculates the BEV distance for a TP box
        d = sqrt(dx*dx + dz*dz)

        Args:
            label (str): the class that will be evaluated
            gt_boxes (List[CsBbox3d]): GT boxes
            pred_boxes (List[CsBbox3d]): Predicted boxes

        Returns:
            np.ndarray: array containing the GT distances
        """

        gt_boxes = np.asarray([x.center for x in gt_boxes])
        pred_boxes = np.asarray([x.center for x in pred_boxes])

        gt_dists = np.sqrt(gt_boxes[..., 0]**2 +
                           gt_boxes[..., 1]**2).astype(int)

        center_dists = gt_boxes - pred_boxes
        center_dists = np.sqrt(center_dists[..., 0]**2 +
                               center_dists[..., 1]**2)

        for gt_dist, center_dist in zip(gt_dists, center_dists):
            if gt_dist >= self.eval_params.max_depth:
                continue

            # instead of unbound distances in m we want to transform this in a score between 0 and 1
            # e.g. if the max_depth == 100
            # score = 1. - (dist / 100)

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][label]["Center_Dist"][gt_dist].append(
                1. - min(center_dist / float(self.eval_params.max_depth), 1.))  # norm it to 1.

        return gt_dists

    def _calcSizeSimilarities(
        self,
        label,       # type: str
        gt_boxes,    # type: List[CsBbox3d]
        pred_boxes,  # type: List[CsBbox3d]
        gt_dists     # type: np.ndarray
    ):
        # type: (...) -> None
        """Internal method that calculates the size similarity for a TP box
        s = min(w/w', w'/w) * min(h/h', h'/h) * min(l/l', l'/l)

        Args:
            label (str): the class that will be evaluated
            gt_boxes (List[CsBbox3d]): GT boxes
            pred_boxes (List[CsBbox3d]): Predicted boxes
            gt_dists (np.ndarray): GT distances
        """

        gt_boxes = np.asarray([x.dims for x in gt_boxes])
        pred_boxes = np.asarray([x.dims for x in pred_boxes])

        size_similarities = np.prod(np.minimum(
            gt_boxes / pred_boxes, pred_boxes / gt_boxes), axis=1)

        for gt_dist, size_simi in zip(gt_dists, size_similarities):
            if gt_dist >= self.eval_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][label]["Size_Similarity"][gt_dist].append(
                size_simi)

    def _calcOrientationSimilarities(
        self,
        label,       # type: str
        gt_boxes,    # type: List[CsBbox3d]
        pred_boxes,  # type: List[CsBbox3d]
        gt_dists     # type: np.ndarray
    ):
        # type: (...) -> None
        """Internal method that calculates the orientation similarity for a TP box.
        os_yaw = (1 + cos(delta)) / 2.
        os_pitch/roll = 0.5 + (cos(delta_pitch) + cos(delta_roll)) / 4.

        Args:
            label (str): the class that will be evaluated
            gt_boxes (List[CsBbox3d]): GT boxes
            pred_boxes (List[CsBbox3d]): Predicted boxes
            gt_dists (np.ndarray): GT distances
        """

        gt_vals = np.asarray(
            [Quaternion(x.rotation).yaw_pitch_roll for x in gt_boxes])
        pred_vals = np.asarray(
            [Quaternion(x.rotation).yaw_pitch_roll for x in pred_boxes])

        os_yaws = (1. + np.cos(gt_vals[..., 0] - pred_vals[..., 0])) / 2.
        os_pitch_rolls = 0.5 + \
            (np.cos(gt_vals[..., 1] - pred_vals[..., 1]) +
             np.cos(gt_vals[..., 2] - pred_vals[..., 2])) / 4.

        for gt_dist, os_yaw, os_pitch_roll in zip(gt_dists, os_yaws, os_pitch_rolls):
            if gt_dist >= self.eval_params.max_depth:
                continue

            gt_dist = int(gt_dist / self.eval_params.step_size) * \
                self.eval_params.step_size

            self._stats["working_data"][label]["OS_Yaw"][gt_dist].append(
                os_yaw)
            self._stats["working_data"][label]["OS_Pitch_Roll"][gt_dist].append(
                os_pitch_roll)

    def _calculateAUC(
        self,
        label   # type: str
    ):
        # type: (...) -> None
        """Internal method that calculates the Area Under Curve (AUC)
        for the available DDTP metrics.

        Args:
            label (str): the class that will be evaluated
        """
        parameter_depth_data = self._stats["working_data"][label]

        for parameter_name, value_dict in parameter_depth_data.items():
            curr_mean = -1.
            result_dict = OrderedDict()
            result_items = OrderedDict()
            result_auc = 0.
            num_items = 0

            depths = []
            vals = []
            num_items_list = []
            all_items = []

            for depth, values in value_dict.items():
                if len(values) > 0:
                    num_items += len(values)
                    all_items += values

                    curr_mean = sum(values) / float(len(values))

                    depths.append(depth)
                    vals.append(curr_mean)
                    num_items_list.append(len(values))

            # AUC is calculated as the mean of all values for available depths
            if len(vals) > 1:
                result_auc = np.mean(vals)
            else:
                result_auc = 0.

            # remove the expanded entries
            for d, v, n in list(zip(depths, vals, num_items_list)):
                result_dict[d] = v
                result_items[d] = n

            self.results[parameter_name][label]["data"] = result_dict
            self.results[parameter_name][label]["auc"] = result_auc
            self.results[parameter_name][label]["items"] = result_items

    def _calcTpStats(self):
        # type (...) -> None
        """Internal method that calculates working point for each class and calculate TP stats.

        Calculated stats are:
          - BEV mean center distance
          - size similarity
          - orientation score for yaw and pitch/roll
        """

        parameters = ["AP", "Center_Dist",
                      "Size_Similarity", "OS_Yaw", "OS_Pitch_Roll"]

        # setup result dict
        for parameter in parameters:
            if parameter == "AP":
                continue
            self.results[parameter] = OrderedDict()
            for x in self.eval_params.labels_to_evaluate:
                self.results[parameter][x] = OrderedDict()
                self.results[parameter][x]["data"] = OrderedDict()
                self.results[parameter][x]["items"] = OrderedDict()
                self.results[parameter][x]["auc"] = 0.

        # calculate the statistics for each class
        for label in self.eval_params.labels_to_evaluate:
            working_confidence = self._stats["working_confidence"][label]
            working_data = self._stats[working_confidence]["data"]

            self._stats["working_data"] = {}
            self._stats["working_data"][label] = OrderedDict()
            self._stats["working_data"][label]["Center_Dist"] = OrderedDict((x, []) for x in self._depth_bins)
            self._stats["working_data"][label]["Size_Similarity"] = OrderedDict((x, []) for x in self._depth_bins)
            self._stats["working_data"][label]["OS_Yaw"] = OrderedDict((x, []) for x in self._depth_bins)
            self._stats["working_data"][label]["OS_Pitch_Roll"] = OrderedDict((x, []) for x in self._depth_bins)

            # loop over all images
            for base_img, tp_fp_fn_data in working_data.items():
                gt_boxes = self.gts[base_img]["objects"]
                pred_boxes = self.preds[base_img]["objects"]

                tp_idx_gt = tp_fp_fn_data["tp_idx_gt"]
                tp_idx_pred = tp_fp_fn_data["tp_idx_pred"]

                # only select the GT boxes
                gt_boxes = [gt_boxes[x] for x in tp_idx_gt[label]]
                pred_boxes = [pred_boxes[x] for x in tp_idx_pred[label]]

                # there is no prediction or GT -> no TP statistics
                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                # calculate center_dists for image
                gt_dists = self._calcCenterDistances(
                    label, gt_boxes, pred_boxes)

                # calculate size similarities
                self._calcSizeSimilarities(
                    label, gt_boxes, pred_boxes, gt_dists)

                # calculate orientation similarities
                self._calcOrientationSimilarities(
                    label, gt_boxes, pred_boxes, gt_dists)

            # calc AUC and detection score
            self._calculateAUC(label)

        # determine which categories have GT data and can be used for mean calculation
        accept_cats = []
        for cat, count in self._stats["GT_stats"].items():
            if count == 0:
                logger.warn("Category {} has no GT!".format(cat))
            else:
                accept_cats.append(cat)

        # add GT statistics and working confidence to results
        self.results["GT_stats"] = self._stats["GT_stats"]
        self.results["working_confidence"] = self._stats["working_confidence"]

        # add evaluation parameters to results
        modal_amodal_modifier = "Amodal"
        if self.eval_params.matching_method == MATCHING_MODAL:
            modal_amodal_modifier = "Modal"

        self.results["eval_params"] = OrderedDict()
        self.results["eval_params"]["labels"] = self.eval_params.labels_to_evaluate
        self.results["eval_params"]["min_iou_to_match"] = self.eval_params.min_iou_to_match
        self.results["eval_params"]["max_depth"] = self.eval_params.max_depth
        self.results["eval_params"]["step_size"] = self.eval_params.step_size
        self.results["eval_params"]["matching_method"] = modal_amodal_modifier

        # calculate detection scores and add them to results
        self.results["Detection_Score"] = OrderedDict()
        logger.info("========================")
        logger.info("======= Results ========")
        logger.info("========================")

        # calculate detection store for each class
        for label in self.eval_params.labels_to_evaluate:
            vals = {p: self.results[p][label]["auc"] for p in parameters}
            det_score = vals["AP"] * (vals["Center_Dist"] + vals["Size_Similarity"] +
                                      vals["OS_Yaw"] + vals["OS_Pitch_Roll"]) / 4.
            self.results["Detection_Score"][label] = det_score

            logger.info(label)
            logger.info(" -> 2D AP {:<6}                : {:8.4f}".format(modal_amodal_modifier, vals["AP"] * 100))
            logger.info(" -> BEV Center Distance (DDTP)  : {:8.4f}".format(vals["Center_Dist"] * 100))
            logger.info(" -> Yaw Similarity (DDTP)       : {:8.4f}".format(vals["OS_Yaw"] * 100))
            logger.info(" -> Pitch/Roll Similarity (DDTP): {:8.4f}".format(vals["OS_Pitch_Roll"] * 100))
            logger.info(" -> Size Similarity (DDTP)      : {:8.4f}".format(vals["Size_Similarity"] * 100))
            logger.info(" -> Detection Score             : {:8.4f}".format(det_score * 100))

        self.results["mDetection_Score"] = np.mean(
            [x for cat, x in self.results["Detection_Score"].items() if cat in accept_cats])
        logger.info("Mean Detection Score: {:8.4f}".format(self.results["mDetection_Score"] * 100))

        # add mean evaluation results
        for parameter_name in parameters:
            self.results["m" + parameter_name] = np.mean(
                [x["auc"] for cat, x in self.results[parameter_name].items() if cat in accept_cats])

    def _calculateAp(self):
        # type: (...) -> None
        """Internal method that calculates Average Precision (AP) values for the whole dataset."""

        for s in self._conf_thresholds:
            score_data = self._stats[s]["data"]

            # dicts containing TP, FP and FN per depth per class
            tp_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}
            fp_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}
            fn_per_depth = {x: {d: [] for d in self._depth_bins} for x in self.eval_params.labels_to_evaluate}

            # dicts containing precision and recall and AP per depth per class
            precision_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}
            recall_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}
            auc_per_depth = {x: {} for x in self.eval_params.labels_to_evaluate}

            # dicts containing overall TP, FP and FN per class
            tp = {x: 0 for x in self.eval_params.labels_to_evaluate}
            fp = {x: 0 for x in self.eval_params.labels_to_evaluate}
            fn = {x: 0 for x in self.eval_params.labels_to_evaluate}

            # dicts containing overall precision, recall and AP per class
            precision = {x: 0 for x in self.eval_params.labels_to_evaluate}
            recall = {x: 0 for x in self.eval_params.labels_to_evaluate}
            auc = {x: 0 for x in self.eval_params.labels_to_evaluate}

            # get the statistics for each image
            for img_base, img_base_stats in score_data.items():
                gt_depths = [x.depth for x in self.gts[img_base]["objects"]]
                pred_depths = [x.depth for x in self.preds[img_base]["objects"]]

                for label, idxs in img_base_stats["tp_idx_gt"].items():
                    tp[label] += len(idxs)

                    for idx in idxs:
                        tp_depth = gt_depths[idx]
                        if tp_depth >= self.eval_params.max_depth:
                            continue

                        tp_depth = int(tp_depth / self.eval_params.step_size) * self.eval_params.step_size

                        tp_per_depth[label][tp_depth].append(idx)

                for label, idxs in img_base_stats["fp_idx_pred"].items():
                    fp[label] += len(idxs)

                    for idx in idxs:
                        fp_depth = pred_depths[idx]
                        if fp_depth >= self.eval_params.max_depth:
                            continue

                        fp_depth = int(fp_depth / self.eval_params.step_size) * self.eval_params.step_size

                        fp_per_depth[label][fp_depth].append(idx)

                for label, idxs in img_base_stats["fn_idx_gt"].items():
                    fn[label] += len(idxs)

                    for idx in idxs:
                        fn_depth = gt_depths[idx]
                        if fn_depth >= self.eval_params.max_depth:
                            continue

                        fn_depth = int(fn_depth / self.eval_params.step_size) * self.eval_params.step_size

                        fn_per_depth[label][fn_depth].append(idx)

            # calculate per depth precision and recall per class
            for label in self.eval_params.labels_to_evaluate:
                for i in self._depth_bins:
                    tp_at_depth = len(tp_per_depth[label][i])
                    fp_at_depth = len(fp_per_depth[label][i])
                    accum_fn = len(fn_per_depth[label][i])

                    if tp_at_depth == 0 and accum_fn == 0:
                        precision_per_depth[label][i] = -1
                        recall_per_depth[label][i] = -1
                    elif tp_at_depth == 0:
                        precision_per_depth[label][i] = 0
                        recall_per_depth[label][i] = 0
                    else:
                        precision_per_depth[label][i] = tp_at_depth / \
                            float(tp_at_depth + fp_at_depth)
                        recall_per_depth[label][i] = tp_at_depth / \
                            float(tp_at_depth + accum_fn)

                    auc_per_depth[label][i] = precision_per_depth[label][i] * \
                        recall_per_depth[label][i]

                if tp[label] == 0:
                    precision[label] = 0
                    recall[label] = 0
                else:
                    precision[label] = tp[label] / \
                        float(tp[label] + fp[label])
                    recall[label] = tp[label] / \
                        float(tp[label] + fn[label])

                auc[label] = precision[label] * recall[label]

            # write to stats
            self._stats[s]["pr_data"] = {
                "tp": tp,
                "fp": tp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "tp_per_depth": tp_per_depth,
                "fp_per_depth": fp_per_depth,
                "fn_per_depth": fn_per_depth,
                "precision_per_depth": precision_per_depth,
                "recall_per_depth": recall_per_depth,
                "auc_per_depth": auc_per_depth,
            }

        # dict containing data for AP and mAP
        ap = OrderedDict()
        for x in self.eval_params.labels_to_evaluate:
            ap[x] = OrderedDict()
            ap[x]["data"] = OrderedDict()
            ap[x]["auc"] = 0.

        ap_per_depth = OrderedDict(
            (x, OrderedDict()) for x in self.eval_params.labels_to_evaluate
        )

        # dict containing the working point for DDTP metrics
        working_confidence = OrderedDict((x, 0) for x in self.eval_params.labels_to_evaluate)

        # calculate standard AP per class
        for label in self.eval_params.labels_to_evaluate:
            # best_auc and best_score are used for determining working point
            best_auc = 0.
            best_score = 0.

            recalls_ = []
            precisions_ = []
            for s in self._conf_thresholds:
                current_auc_for_score = self._stats[s]["pr_data"]["auc"][label]
                if current_auc_for_score > best_auc:
                    best_auc = current_auc_for_score
                    best_score = s

                recalls_.append(self._stats[s]["pr_data"]["recall"][label])
                precisions_.append(self._stats[s]["pr_data"]["precision"][label])

            # sort for an ascending recalls list
            sorted_pairs = sorted(zip(recalls_, precisions_), key=lambda pair: pair[0])
            recalls, precisions = map(list, zip(*sorted_pairs))

            # convert the data to numpy tensor for easier processing and add leading and trailing zeros/ones
            precisions = np.asarray([0] + precisions + [0])
            recalls = np.asarray([0] + recalls + [1])

            # precision values should be decreasing only
            # p(r) = max{r' > r} p(r')
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = np.maximum(precisions[i], precisions[i + 1])

            # gather indices of distinct recall values
            recall_idx = np.where(recalls[1:] != recalls[:-1])[0] + 1

            # calculate ap
            class_ap = np.sum(
                (recalls[recall_idx] - recalls[recall_idx - 1]) * precisions[recall_idx])

            ap[label]["auc"] = float(class_ap)
            ap[label]["data"]["recall"] = [float(x) for x in recalls_]
            ap[label]["data"]["precision"] = [float(x) for x in precisions_]

            # store best confidence value or use specified default
            if (self.eval_params.cw == -1.0):
                working_confidence[label] = best_score
            else:
                working_confidence[label] = self.eval_params.cw

        # calculate depth dependent mAP
        for label in self.eval_params.labels_to_evaluate:
            for d in self._depth_bins:
                tmp_dict = OrderedDict()
                tmp_dict["data"] = OrderedDict()
                tmp_dict["auc"] = 0.

                recalls_ = []
                precisions_ = []

                valid_depth = True
                for s in self._conf_thresholds:
                    if d not in self._stats[s]["pr_data"]["recall_per_depth"][label].keys():
                        valid_depth = False
                        break

                    tmp_recall = self._stats[s]["pr_data"]["recall_per_depth"][label][d]
                    tmp_precision = self._stats[s]["pr_data"]["precision_per_depth"][label][d]

                    if tmp_recall >= 0 and tmp_precision >= 0:
                        recalls_.append(tmp_recall)
                        precisions_.append(tmp_precision)

                if len(precisions_) > 0 and len(recalls_) > 0:
                    if not valid_depth:
                        continue

                    # sort for an ascending recalls list
                    sorted_pairs = sorted(
                        zip(recalls_, precisions_), key=lambda pair: pair[0])
                    recalls, precisions = map(list, zip(*sorted_pairs))

                    # convert the data to numpy tensor for easier processing and add leading and trailing zeros/ones
                    precisions = np.asarray([0] + precisions + [0])
                    recalls = np.asarray([0] + recalls + [1])

                    # precision values should be decreasing only
                    # p(r) = max{r' > r} p(r')
                    for i in range(len(precisions) - 2, -1, -1):
                        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

                    # gather indices of distinct recall values
                    recall_idx = np.where(recalls[1:] != recalls[:-1])[0] + 1

                    # calculate ap
                    class_ap = np.sum(
                        (recalls[recall_idx] - recalls[recall_idx - 1]) * precisions[recall_idx])

                    tmp_dict["auc"] = float(class_ap)
                    tmp_dict["data"]["recall"] = [float(x) for x in recalls_]
                    tmp_dict["data"]["precision"] = [
                        float(x) for x in precisions_]

                    ap_per_depth[label][d] = tmp_dict
                else:  # no valid detection until this depth
                    tmp_dict["auc"] = -1.
                    tmp_dict["data"]["recall"] = []
                    tmp_dict["data"]["precision"] = []

        # dump mAP and working points to internal stats
        self._stats["min_iou"] = self.eval_params.min_iou_to_match
        self._stats["working_confidence"] = working_confidence
        self.results["AP"] = ap
        self.results["AP_per_depth"] = ap_per_depth

# evaluation method


def evaluate3dObjectDetection(
    gt_folder,      # type: str
    pred_folder,    # type: str
    result_folder,  # type: str
    eval_params,    # type: EvaluationParameters
    plot=True       # type: bool
):
    # type: (...) -> None
    """Performs the 3D object detection evaluation.

    Args:
        gt_folder (str): directory of the GT annotation files
        pred_folder (str): directory of the prediction files
        result_folder (str): directory in which the result files are saved
        eval_params (EvaluationParameters): evaluation parameters
        plot (bool): plot the evaluation results
    """

    # initialize the evaluator
    boxEvaluator = Box3dEvaluator(eval_params)
    boxEvaluator.checkCw()

    logger.info("Use the following options")
    logger.info(" -> GT folder    : {}".format(gt_folder))
    logger.info(" -> Pred folder  : {}".format(pred_folder))
    logger.info(" -> Labels       : {}".format(", ".join(eval_params.labels_to_evaluate)))
    logger.info(" -> Min IoU:     : {:.2f}".format(eval_params.min_iou_to_match))
    logger.info(" -> Max depth [m]: {}".format(eval_params.max_depth))
    logger.info(" -> Step size [m]: {}".format(eval_params.step_size))
    if boxEvaluator.eval_params.cw == -1.0:
        logger.info(" -> cw           : -- automatically determined --")
    else:
        logger.info(" -> cw           : {:.2f}".format(boxEvaluator.eval_params.cw))

    # load GT and predictions
    boxEvaluator.loadGT(gt_folder)
    boxEvaluator.loadPredictions(pred_folder)

    # perform evaluation
    boxEvaluator.evaluate()

    # save results and plot them
    boxEvaluator.saveResults(result_folder)

    if plot:
        # lazy import as matplotlib does not run properly on all python version for all OSs
        from cityscapesscripts.evaluation.plot3dResults import plot_data
        plot_data(boxEvaluator.results)

    return


def main():
    """main method"""
    logger.info("========================")
    logger.info("=== Start evaluation ===")
    logger.info("========================")

    # get cityscapes paths
    cityscapesPath = os.environ.get(
        'CITYSCAPES_DATASET', os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', '..')
    )
    gtFolder = os.path.join(cityscapesPath, "gtBbox3d", "val")

    predictionPath = os.environ.get(
        'CITYSCAPES_RESULTS',
        os.path.join(cityscapesPath, "results")
    )
    predictionFolder = os.path.join(predictionPath, "predBbox3d")

    parser = argparse.ArgumentParser()
    # setup location
    gt_folder_arg = parser.add_argument("-gt", "--gt-folder",
                                        dest="gtFolder",
                                        help="path to folder that contains ground truth *.json files. If the "
                                        "argument is not provided this script will look for the *.json files in "
                                        "the 'gtBbox3d/val' folder in CITYSCAPES_DATASET.",
                                        default=gtFolder,
                                        type=str)

    pred_folder_arg = parser.add_argument("-pred", "--prediction-folder",
                                          dest="predictionFolder",
                                          help="path to folder that contains ground truth * .json files. If the "
                                          "argument is not provided this script will look for the * .json files in "
                                          "the 'predBbox3d' folder in CITYSCAPES_RESULTS.",
                                          default=predictionFolder,
                                          type=str)

    parser.add_argument("--results-folder",
                        dest="resultsFolder",
                        help="File to store evaluation results. Default: prediction folder",
                        default="",
                        type=str)

    # setup evaluation parameters
    evalLabels = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]
    parser.add_argument("--labels",
                        dest="evalLabels",
                        help="Labels to be evaluated separated with a space. Default: {}".format(" ".join(evalLabels)),
                        default=evalLabels,
                        nargs="+",
                        type=str)
    minIou = 0.7
    parser.add_argument("--min-iou",
                        dest="minIou",
                        help="Minimum IoU required to accept a detection as TP. Default: {}".format(minIou),
                        default=minIou,
                        type=float)
    maxDepth = 100
    parser.add_argument("--max-depth",
                        dest="maxDepth",
                        help="Maximum depth for DDTP metrics. Default: {}".format(maxDepth),
                        default=maxDepth,
                        type=int)
    stepSize = 5
    parser.add_argument("--step-size",
                        dest="stepSize",
                        help="Step size for DDTP metrics. Default: {}".format(stepSize),
                        default=stepSize,
                        type=int)

    cw = -1.
    parser.add_argument("--cw",
                        dest="cw",
                        help="Working confidence. If not set, it will be determined automatically during evaluation",
                        default=cw,
                        type=float)

    parser.add_argument("--modal",
                        action="store_true",
                        help="Use modal 2D boxes for matching")

    parser.add_argument("--noplot",
                        dest="plot_results",
                        action="store_false",
                        help="Don't plot the graphical results")

    args = parser.parse_args()

    if not os.path.exists(args.gtFolder):
        msg = "Could not find gt folder '{}'. Please run the script with '--help'".format(args.gtFolder)
        logger.error(msg)
        raise argparse.ArgumentError(gt_folder_arg, msg)

    if not os.path.exists(args.predictionFolder):
        msg = "Could not find prediction folder '{}'. Please run the script with '--help'".format(args.predictionFolder)
        logger.error(msg)
        raise argparse.ArgumentError(pred_folder_arg, msg)

    resultsFolder = args.resultsFolder
    if not resultsFolder:
        resultsFolder = args.predictionFolder
    # keep python 2 compatibility
    if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

    # setup the evaluation parameters
    eval_params = EvaluationParameters(
        args.evalLabels,
        min_iou_to_match=args.minIou,
        max_depth=args.maxDepth,
        step_size=args.stepSize,
        matching_method=int(args.modal),
        cw=args.cw
    )

    evaluate3dObjectDetection(
        args.gtFolder,
        args.predictionFolder,
        resultsFolder,
        eval_params,
        plot=args.plot_results
    )

    logger.info("========================")
    logger.info("=== Stop evaluation ====")
    logger.info("========================")


if __name__ == "__main__":
    # call the main method
    main()
