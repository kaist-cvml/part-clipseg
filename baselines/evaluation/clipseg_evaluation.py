# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.visualizer import ColorMode, Visualizer

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.utils.colormap import random_color
from detectron2.evaluation import SemSegEvaluator

from baselines.utils.visualizer import CustomVisualizer

import cv2
_CV2_IMPORTED = True


class CLIPSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        visualize=True,
        visualize_attn=True,
        ORACLE=False,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.oracle = ORACLE
        self.visualize = visualize

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        if self.visualize:
            self.vis_path = os.path.join(self._output_dir, "visualization")
            PathManager.mkdirs(self.vis_path)

        self.meta = meta

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True  # True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        super().reset()
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )

    def process(self, inputs, outputs, visualize=True):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):

            # Replace object mask with the sum of GT masks
            # (because there are PASCAL objects that are not included in the PASCAL-Part GT annotation)

            sem_seg_output, sem_seg_output_all = output["sem_seg"], output["sem_seg_all"]

            output = self.post_process_func(
                sem_seg_output, image=np.array(Image.open(input["file_name"]))
            )
            output_all = self.post_process_func(
                sem_seg_output_all,
                image=np.array(Image.open(input["file_name"]))
            )

            output = output.argmax(dim=0).to(self._cpu_device)
            output_all = output_all.argmax(dim=0).to(self._cpu_device)

            gt_classes = input["obj_part_instances"].gt_classes
            gt_masks = input["obj_part_instances"].gt_masks
            eval_image_size = tuple(output.shape[-2:])

            # GT: Union of Part GT
            if len(gt_masks) == 0:
                gt = np.zeros_like(pred) + self._ignore_label
            else:
                gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label
                for i in range(len(gt_classes)):
                    gt[gt_masks[i] == True] = gt_classes[i]

                gt = F.interpolate(
                    torch.tensor(gt).unsqueeze(0).unsqueeze(0),
                    size=eval_image_size,
                    mode='nearest'
                ).squeeze()
                gt = gt.int().numpy()

            # output    : Visualization for Oracle-Obj
            # output_all: Visualization for Pred-All
            # pred      : Evaluation for Oracle-Obj
            # pred_all  : Evaluation for Pred-All

            output[gt == self._ignore_label] = self.meta.ignore_label
            pred = np.array(output, dtype=np.int)
            pred[pred == self._ignore_label] = self._num_classes
            pred_all = np.array(output_all, dtype=np.int)
            pred_all[pred_all == self._ignore_label] = self._num_classes
            pred_all[(gt == self._ignore_label)] = self._num_classes
            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1)
                if self.oracle
                else (self._num_classes + 1) * pred_all.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8)).astype(np.int64)
                b_pred = self._mask_to_boundary(pred.astype(np.uint8) if self.oracle else pred_all.astype(np.uint8)).astype(np.int64)

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) *
                    b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred if self.oracle else pred_all, input["file_name"]))

            if self.visualize:
                ext = os.path.splitext(input["file_name"])[1]
                input_img_tensor = F.interpolate(input["image"].unsqueeze(0), size=eval_image_size, mode='bilinear').squeeze()
                input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()
                input_img_pil = Image.fromarray(input_img_npy.astype(np.uint8))

                pred_all_pil = Image.fromarray(pred_all.astype(np.uint8)).convert('P')
                pred_all_pil.putpalette([v for color in self.meta.stuff_colors for v in color])
                pred_all_pil = pred_all_pil.convert('RGB')
                pred_all_pil = Image.blend(input_img_pil, pred_all_pil, 0.5)
                pred_all_pil.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all_raw.jpg")))

                visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)

                vis_pred = visualizer_pred.draw_sem_seg(pred)
                vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))

                vis_pred_all = visualizer_pred_all.draw_sem_seg(np.array(output_all, dtype=np.int))
                vis_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all.jpg")))

                vis_gt = visualizer_gt.draw_sem_seg(gt)
                vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        acc = np.full(self._num_classes, np.nan, dtype=np.float) if self.oracle else np.full(self._num_classes + 1, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float) if self.oracle else np.full(self._num_classes + 1, np.nan, dtype=np.float)
        recall = np.full(self._num_classes, np.nan, dtype=np.float) if self.oracle else np.full(self._num_classes + 1, np.nan, dtype=np.float)

        if not self.oracle:
            self._conf_matrix[:, -1] = 0

        tp = self._conf_matrix.diagonal()[:-1].astype(np.float) if self.oracle else self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float) if self.oracle else np.sum(self._conf_matrix, axis=0).astype(np.float)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float) if self.oracle else np.sum(self._conf_matrix, axis=1).astype(np.float)

        class_weights = pos_gt / np.sum(pos_gt)
        recall_valid = pos_gt > 0
        acc[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        union = pos_gt + pos_pred - tp
        iou_valid = (pos_gt + pos_pred) > 0

        if self.oracle:
            iou[recall_valid] = tp[recall_valid] / union[recall_valid]
        else:
            iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        recall[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        macc = np.nanmean(acc)
        miou = np.nanmean(iou)
        fiou = np.nansum(iou * class_weights)
        pacc = np.nansum(tp) / np.nansum(pos_gt)
        mRecall = np.nanmean(recall)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=np.float) if self.oracle else np.full(self._num_classes + 1, np.nan, dtype=np.float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float) if self.oracle else self._b_conf_matrix.diagonal().astype(np.float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float) if self.oracle else np.sum(self._b_conf_matrix, axis=0).astype(np.float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float) if self.oracle else np.sum(self._b_conf_matrix, axis=1).astype(np.float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res["mRecall"] = 100 * mRecall

        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
            res["Recall-{}".format(name)] = 100 * recall[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]

        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                biou_list = []
                recall_list = []

                set_inds = np.array(set_inds, dtype=int)
                mask = np.zeros(len(iou), dtype=bool)
                mask[set_inds] = True

                subset_iou_valid = recall_valid & mask if self.oracle else iou_valid & mask
                subset_acc_valid = recall_valid & mask

                if np.any(subset_iou_valid):
                    miou = np.sum(iou[mask][recall_valid[mask]]) / np.sum(recall_valid[mask]) if not self.oracle else np.nanmean(iou[mask])
                else:
                    miou = np.nan

                if np.any(subset_acc_valid):
                    mrecall = np.nanmean(recall[subset_acc_valid])
                else:
                    mrecall = np.nan

                if np.nansum(pos_gt[mask]) > 0:
                    pacc = np.nansum(tp[mask]) / np.nansum(pos_gt[mask])
                else:
                    pacc = np.nan

                res[f"mIoU-{set_name}"] = 100 * miou
                res[f"pAcc-{set_name}"] = 100 * pacc
                res[f"mRecall-{set_name}"] = 100 * mrecall
                iou_list.append(miou)
                recall_list.append(mrecall)

                if self._compute_boundary_iou:
                    if np.any(mask):
                        b_miou = np.nanmean(b_iou[mask])
                    else:
                        b_miou = np.nan
                    res[f"Boundary-mIoU-{set_name}"] = 100 * b_miou
                    biou_list.append(b_miou)
                inv_mask = ~mask

                subset_iou_valid = recall_valid & inv_mask if self.oracle else iou_valid & inv_mask
                subset_acc_valid = recall_valid & inv_mask

                if np.any(subset_iou_valid):
                    miou = np.sum(iou[inv_mask][recall_valid[inv_mask]]) / np.sum(recall_valid[inv_mask]) if not self.oracle else np.nanmean(iou[inv_mask])
                else:
                    miou = np.nan

                if np.any(subset_acc_valid):
                    mrecall = np.nanmean(recall[subset_acc_valid])
                else:
                    mrecall = np.nan

                if np.nansum(pos_gt[inv_mask]) > 0:
                    pacc = np.nansum(tp[inv_mask]) / np.nansum(pos_gt[inv_mask])
                else:
                    pacc = np.nan

                res[f"mIoU-un{set_name}"] = 100 * miou
                res[f"pAcc-un{set_name}"] = 100 * pacc
                res[f"mRecall-un{set_name}"] = 100 * mrecall
                iou_list.append(miou)
                recall_list.append(mrecall)

                if self._compute_boundary_iou:
                    if np.any(inv_mask):
                        b_miou = np.nanmean(b_iou[inv_mask])
                    else:
                        b_miou = np.nan
                    res[f"Boundary-mIoU-un{set_name}"] = 100 * b_miou
                    biou_list.append(b_miou)

        res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase'])
        res['h-bIoU'] = 2 * (res['Boundary-mIoU-base'] * res['Boundary-mIoU-unbase']) / (res['Boundary-mIoU-base'] + res['Boundary-mIoU-unbase'])
        res['h-Recall'] = 2 * (res['mRecall-base'] * res['mRecall-unbase']) / (res['mRecall-base'] + res['mRecall-unbase'])

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results