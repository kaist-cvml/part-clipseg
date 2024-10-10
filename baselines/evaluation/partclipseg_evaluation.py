# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
import gc
from collections import OrderedDict, defaultdict
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
from baselines.utils.vis_uitls import show_image_relevance

import cv2
_CV2_IMPORTED = True
# try:
#     import cv2  # noqa
# except ImportError:
#     # OpenCV is an optional dependency at the moment
#     _CV2_IMPORTED = False

# TODO:
json_index_counter = 0


class PartCLIPSegEvaluator(SemSegEvaluator):
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
        ORACLE=False,
        visualize=True,
        visualize_attn=False
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
        self.visualize_attn = visualize_attn

        if self.visualize:
            self.vis_path = os.path.join(self._output_dir, "visualization")
            PathManager.mkdirs(self.vis_path)

        self.meta = meta

        # for dataset's statistics
        self.obj_class_num = defaultdict(int)
        self.part_class_num = defaultdict(int)
        self.obj_part_class_num = defaultdict(int)
        self.total_num = 0

        self.obj_class_ratio_sum = dict()
        self.part_class_ratio_sum = dict()
        self.part_class_ratio_in_obj_sum = dict()
        self.obj_part_class_ratio_sum = dict()
        self.obj_part_class_ratio_in_obj_sum = dict()

        for obj_part_class in meta.stuff_classes:
            # check if obj_part_clas is a key in obj_part_class_ratio
            if obj_part_class not in self.obj_part_class_ratio_sum:
                self.obj_part_class_ratio_sum[obj_part_class] = 0
                self.obj_part_class_ratio_in_obj_sum[obj_part_class] = 0

        for obj_class in meta.obj_classes:
            # check if obj_class is a key in obj_class_ratio
            if obj_class not in self.obj_class_ratio_sum:
                self.obj_class_ratio_sum[obj_class] = 0

        for part_class in meta.part_classes:
            # check if part_class is a key in part_class_ratio
            if part_class not in self.part_class_ratio_sum:
                self.part_class_ratio_sum[part_class] = 0
                self.part_class_ratio_in_obj_sum[part_class] = 0

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True # True
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

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        with torch.no_grad():
            for input, output in zip(inputs, outputs):
                if self.visualize_attn:
                    attns = output["attns"]
                obj_mask = input["instances"].gt_masks[0]

                sem_seg_output, sem_seg_output_all = output["sem_seg"], output["sem_seg_with_bg"]
                obj_sem_seg_output_all = output["obj_sem_seg_with_bg"]
                part_sem_seg_output = output["part_sem_seg"]

                output = self.post_process_func(sem_seg_output, image=np.array(Image.open(input["file_name"])))
                output_all = self.post_process_func(sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
                obj_output_all = self.post_process_func(obj_sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
                part_output = self.post_process_func(part_sem_seg_output, image=np.array(Image.open(input["file_name"])))

                output = output.argmax(dim=0).to(self._cpu_device)
                output_all = output_all.argmax(dim=0).to(self._cpu_device)
                obj_output_all = obj_output_all.to(self._cpu_device)
                part_output = part_output.argmax(dim=0).to(self._cpu_device)

                obj_mask = F.interpolate(obj_mask.float().unsqueeze(0).unsqueeze(0), size=output.shape[-2:], mode='nearest').squeeze()
                output[obj_mask == 0.0] = self.meta.ignore_label

                pred = np.array(output, dtype=np.int)
                pred_all = np.array(output_all, dtype=np.int)
                obj_pred_all = np.array(obj_output_all, dtype=np.int)
                part_pred = np.array(part_output, dtype=np.int)

                gt_classes = input["obj_part_instances"].gt_classes
                gt_masks = input["obj_part_instances"].gt_masks
                eval_image_size = pred.shape[-2:]

                # for dataset's statistics
                gt_size = eval_image_size[0] * eval_image_size[1]
                # category_id = input["category_id"]
                category_id = input['sem_seg'].unique()[0]
                mapped_category_id = self.meta.obj_map[category_id] if "val" not in self.meta.name else category_id
                obj_class_name = self.meta.obj_classes[mapped_category_id]
                self.obj_class_ratio_sum[obj_class_name] += (obj_mask.nonzero().shape[0] / gt_size)
                self.obj_class_num[obj_class_name] += 1

                if len(gt_masks) == 0:
                    gt = np.zeros_like(pred) + self._ignore_label
                else:
                    gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label
                    for i in range(len(gt_classes)):
                        gt[gt_masks[i] == True] = gt_classes[i]

                        # for dataset's statistics
                        gt_mask = F.interpolate(
                            gt_masks[i].clone().detach().float().unsqueeze(0).unsqueeze(0),
                            size=eval_image_size,
                            mode='nearest'
                        ).squeeze().squeeze()

                        gt_region = gt_mask.nonzero().shape[0]
                        gt_class_name = self.meta.stuff_classes[gt_classes[i]]
                        self.obj_part_class_ratio_sum[gt_class_name] += (gt_region / gt_size)
                        self.obj_part_class_ratio_in_obj_sum[gt_class_name] += (gt_region / obj_mask.nonzero().shape[0])
                        self.obj_part_class_num[gt_class_name] += 1

                        part_class_name = gt_class_name.split("'s")[1].strip()
                        self.part_class_ratio_sum[part_class_name] += (gt_region / gt_size)
                        self.part_class_ratio_in_obj_sum[part_class_name] += (gt_region / obj_mask.nonzero().shape[0])
                        self.part_class_num[part_class_name] += 1

                    # eval_image_size = pred.shape[-2:]
                    gt = F.interpolate(
                        torch.tensor(gt).unsqueeze(0).unsqueeze(0),
                        size=eval_image_size,
                        mode='nearest'
                    ).squeeze()
                    gt = gt.int().numpy()
                pred[pred == self._ignore_label] = self._num_classes
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
                        (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                        minlength=self._conf_matrix.size,
                    ).reshape(self._conf_matrix.shape)

                self._predictions.extend(self.encode_json_sem_seg(
                    pred if self.oracle else pred_all, input["file_name"]))
                

                if self.visualize:
                    ext = os.path.splitext(input["file_name"])[1]

                    input_img_tensor = F.interpolate(
                        input["image"].unsqueeze(0),
                        size=eval_image_size,
                        mode='bilinear'
                    ).squeeze()
                    input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()
                    input_img_pil = Image.fromarray(input_img_npy.astype(np.uint8))

                    if self.visualize_attn:
                        for tag, attn in attns.items():
                            attn_tensor = attn.unsqueeze(0)

                            attn = attn.squeeze(0).cpu().numpy()
                            attn = (attn * 255.).astype(np.uint8)
                            attn_img = Image.fromarray(attn).convert('L')
                            attn_img = attn_img.resize((input["image"].shape[-1], input["image"].shape[-2]))
                            attn_img.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, f"_{tag}.jpg")))

                            attn_npy = show_image_relevance(attn_tensor, input_img_pil)
                            attn_img = Image.fromarray(attn_npy)
                            attn_img.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, f"_relevance_{tag}.jpg")))

                    obj_pred_all_pil = Image.fromarray(obj_pred_all.astype(np.uint8)).convert('P')
                    obj_pred_all_pil.putpalette([v for color in self.meta.obj_colors for v in color])
                    obj_pred_all_pil = obj_pred_all_pil.convert('RGB')
                    obj_pred_all_pil = Image.blend(input_img_pil, obj_pred_all_pil, 0.5)
                    obj_pred_all_pil.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_obj_all_raw.jpg")))

                    part_pred_pil = Image.fromarray(part_pred.astype(np.uint8)).convert('P')
                    part_pred_pil.putpalette([v for color in self.meta.part_colors for v in color])
                    part_pred_pil = part_pred_pil.convert('RGB')
                    part_pred_pil = Image.blend(input_img_pil, part_pred_pil, 0.5)
                    part_pred_pil.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_part_raw.jpg")))

                    visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_obj_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_part_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                    visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)

                    vis_pred = visualizer_pred.draw_sem_seg(pred)
                    vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))

                    vis_pred_all = visualizer_pred_all.draw_sem_seg(pred_all)
                    vis_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all.jpg")))

                    vis_obj_pred_all = visualizer_obj_pred_all.draw_obj_sem_seg(obj_pred_all)
                    vis_obj_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_obj_all.jpg")))

                    vis_part_pred = visualizer_part_pred.draw_part_sem_seg(part_pred)
                    vis_part_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_part.jpg")))

                    vis_gt = visualizer_gt.draw_sem_seg(gt)
                    vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))

            # for dataset's statistics
            self.total_num += 1

            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                with PathManager.open(os.path.join(self._output_dir, "statistics.json"), "w") as f:
                    json.dump({
                        "obj_class_num": self.obj_class_num,
                        "part_class_num": self.part_class_num,
                        "obj_part_class_num": self.obj_part_class_num,
                        "total_num": self.total_num,
                        "obj_class_ratio_sum": self.obj_class_ratio_sum,
                        "part_class_ratio_sum": self.part_class_ratio_sum,
                        "obj_part_class_ratio_sum": self.obj_part_class_ratio_sum,
                        "obj_part_class_ratio_in_obj_sum": self.obj_part_class_ratio_in_obj_sum
                    }, f)

            torch.cuda.empty_cache()
            gc.collect()

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

        # TODO: epoch
        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     global json_index_counter
        #     json_index_counter += 1
        #     file_path = os.path.join(self._output_dir, f"sem_seg_predictions_{json_index_counter:04d}.json")
        #     # file_path = os.path.join(self._output_dir, f"sem_seg_predictions.json")
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        recall = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        # tp = self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        # pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        # pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        recall[acc_valid] = tp[acc_valid] / (tp[acc_valid] + (pos_gt[acc_valid] - tp[acc_valid]))
        mRecall = np.sum(recall[acc_valid]) / np.sum(acc_valid)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=np.float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float)
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

                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                mrecall = np.sum(recall[mask][acc_valid[mask]]) / np.sum(acc_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                res["mRecall-{}".format(set_name)] = 100 * mrecall
                iou_list.append(miou)
                recall_list.append(mrecall)

                if self._compute_boundary_iou:
                    b_miou = np.sum(b_iou[mask][b_iou_valid[mask]]) / np.sum(b_iou_valid[mask])
                    res["Boundary-mIoU-{}".format(set_name)] = 100 * b_miou
                    biou_list.append(b_miou)

                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                mrecall = np.sum(recall[~mask][acc_valid[~mask]]) / np.sum(acc_valid[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                res["mRecall-un{}".format(set_name)] = 100 * mrecall
                iou_list.append(miou)
                recall_list.append(mrecall)

                if self._compute_boundary_iou:
                    b_miou = np.sum(b_iou[~mask][b_iou_valid[~mask]]) / np.sum(b_iou_valid[~mask])
                    res["Boundary-mIoU-un{}".format(set_name)] = 100 * b_miou
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


    # def evaluate(self):
    #     """
    #     Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

    #     * Mean intersection-over-union averaged across classes (mIoU)
    #     * Frequency Weighted IoU (fwIoU)
    #     * Mean pixel accuracy averaged across classes (mACC)
    #     * Pixel Accuracy (pACC)
    #     """
    #     if self._distributed:
    #         synchronize()
    #         conf_matrix_list = all_gather(self._conf_matrix)
    #         b_conf_matrix_list = all_gather(self._b_conf_matrix)
    #         self._predictions = all_gather(self._predictions)
    #         self._predictions = list(itertools.chain(*self._predictions))
    #         if not is_main_process():
    #             return

    #         self._conf_matrix = np.zeros_like(self._conf_matrix)
    #         for conf_matrix in conf_matrix_list:
    #             self._conf_matrix += conf_matrix

    #         self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
    #         for b_conf_matrix in b_conf_matrix_list:
    #             self._b_conf_matrix += b_conf_matrix

    #     # TODO: epoch
    #     if self._output_dir:
    #         PathManager.mkdirs(self._output_dir)
    #         global json_index_counter
    #         json_index_counter += 1
    #         file_path = os.path.join(self._output_dir, f"sem_seg_predictions_{json_index_counter:04d}.json")
    #         # file_path = os.path.join(self._output_dir, f"sem_seg_predictions.json")
    #         with PathManager.open(file_path, "w") as f:
    #             f.write(json.dumps(self._predictions))

    #     acc = np.full(self._num_classes, np.nan, dtype=np.float)
    #     iou = np.full(self._num_classes, np.nan, dtype=np.float)
    #     recall = np.full(self._num_classes, np.nan, dtype=np.float)
    #     tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
    #     # tp = self._conf_matrix.diagonal().astype(np.float)
    #     pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
    #     # pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
    #     class_weights = pos_gt / np.sum(pos_gt)
    #     pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
    #     # pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
    #     acc_valid = pos_gt > 0
    #     acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    #     iou_valid = (pos_gt + pos_pred) > 0
    #     union = pos_gt + pos_pred - tp
    #     iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    #     macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    #     miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    #     fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    #     pacc = np.sum(tp) / np.sum(pos_gt)

    #     recall[acc_valid] = tp[acc_valid] / (tp[acc_valid] + (pos_gt[acc_valid] - tp[acc_valid]))
    #     mRecall = np.sum(recall[acc_valid]) / np.sum(acc_valid)

    #     if self._compute_boundary_iou:
    #         b_iou = np.full(self._num_classes, np.nan, dtype=np.float)
    #         b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float)
    #         b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float)
    #         b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float)
    #         b_union = b_pos_gt + b_pos_pred - b_tp
    #         b_iou_valid = b_union > 0
    #         b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

    #     res = {}
    #     res["mIoU"] = 100 * miou
    #     res["fwIoU"] = 100 * fiou
    #     for i, name in enumerate(self._class_names):
    #         res["IoU-{}".format(name)] = 100 * iou[i]
    #     res["mACC"] = 100 * macc
    #     res["pACC"] = 100 * pacc
    #     res["mRecall"] = 100 * mRecall

    #     for i, name in enumerate(self._class_names):
    #         res["ACC-{}".format(name)] = 100 * acc[i]
    #         res["Recall-{}".format(name)] = 100 * recall[i] 
    #         if self._compute_boundary_iou:
    #             res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]

    #     if self._evaluation_set is not None:
    #         for set_name, set_inds in self._evaluation_set.items():
    #             iou_list = []
    #             biou_list = []

    #             set_inds = np.array(set_inds, np.int)
    #             mask = np.zeros((len(iou),)).astype(np.bool)
    #             mask[set_inds] = 1
    #             miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
    #             pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
    #             res["mIoU-{}".format(set_name)] = 100 * miou
    #             res["pAcc-{}".format(set_name)] = 100 * pacc
    #             iou_list.append(miou)

    #             if self._compute_boundary_iou:
    #                 b_miou = np.sum(b_iou[mask][b_iou_valid[mask]]) / np.sum(b_iou_valid[mask])
    #                 res["Boundary-mIoU-{}".format(set_name)] = 100 * b_miou
    #                 biou_list.append(b_miou)

    #             miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
    #             pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
    #             res["mIoU-un{}".format(set_name)] = 100 * miou
    #             res["pAcc-un{}".format(set_name)] = 100 * pacc
    #             iou_list.append(miou)

    #             if self._compute_boundary_iou:
    #                 b_miou = np.sum(b_iou[~mask][b_iou_valid[~mask]]) / np.sum(b_iou_valid[~mask])
    #                 res["Boundary-mIoU-un{}".format(set_name)] = 100 * b_miou
    #                 biou_list.append(b_miou)

    #     res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase'])
    #     res['h-bIoU'] = 2 * (res['Boundary-mIoU-base'] * res['Boundary-mIoU-unbase']) / (res['Boundary-mIoU-base'] + res['Boundary-mIoU-unbase'])

    #     if self._output_dir:
    #         file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
    #         with PathManager.open(file_path, "wb") as f:
    #             torch.save(res, f)
    #     results = OrderedDict({"sem_seg": res})
    #     self._logger.info(results)
    #     return results
