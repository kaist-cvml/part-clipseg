from detectron2.modeling import META_ARCH_REGISTRY
from PIL import Image
import torch
import math
import gc
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import PartCLIPSegProcessor, PartCLIPSegForImageSegmentation
from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from baselines.utils.losses import focal_loss_with_logits, focal_loss, iou_loss
from baselines.utils.gaussian_smoothing import GaussianSmoothing
from collections import defaultdict


@META_ARCH_REGISTRY.register()
class PartCLIPSeg(nn.Module):
    @configurable
    def __init__(self, train_dataset, test_dataset, ORACLE, 
                 use_attention, obj_lambda, part_lambda, 
                 attn_mask_threshold, attn_sep_lambda, attn_enh_lambda, *kwargs):
        super().__init__()

        self.device = "cuda"
        self.oracle = ORACLE
        self.use_attention = use_attention
        self.obj_lambda = obj_lambda
        self.part_lambda = part_lambda
        # self.segmentation_background_threshold = 0.0

        self.ignore_label = MetadataCatalog.get(test_dataset).ignore_label
        self.init_train_metadata(train_dataset)
        self.init_test_metadata(test_dataset)

        self.partclipseg_processor = PartCLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.partclipseg_model = PartCLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

        for name, params in self.partclipseg_model.named_parameters():
            # VA+L+F+D
            if False \
                    or 'visual_adapter' in name \
                    or 'clip.text_model.embeddings' in name \
                    or 'film' in name \
                    or 'decoder' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.train_text_encoding = self.partclipseg_processor.tokenizer(
            self.train_part_classes + self.train_obj_in_part_classes,
            return_tensors="pt",
            padding="max_length",
        )

        # ------------
        # load weights
        # ------------
        self.partclipseg_model.decoder.part_obj_with_cond_embed.load_state_dict(self.partclipseg_model.decoder.transposed_convolution.state_dict())
        self.partclipseg_model.decoder.cond_film_add.load_state_dict(self.partclipseg_model.decoder.film_add.state_dict())
        self.partclipseg_model.decoder.cond_film_mul.load_state_dict(self.partclipseg_model.decoder.film_mul.state_dict())
        self.partclipseg_model.decoder.part_obj_with_cond_embed.requires_grad = True
        self.partclipseg_model.decoder.cond_film_add.requires_grad = True
        self.partclipseg_model.decoder.cond_film_mul.requires_grad = True

        if self.use_attention:
            self.attn_level = kwargs["attn_level"] if "attn_level" in kwargs else [2]
            self.attn_mask_threshold = attn_mask_threshold
            self.attn_sep_lambda = attn_sep_lambda
            self.attn_enh_lambda = attn_enh_lambda
            # self.lambda_attn_sep = kwargs["lambda_attn_sep"] if "lambda_attn_sep" in kwargs else 0.1
            # self.lambda_attn_enh = kwargs["lambda_attn_enh"] if "lambda_attn_enh" in kwargs else 0.01

    @classmethod
    def from_config(cls, cfg):
        # TODO: merge params
        ret = {}
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
        ret['test_dataset'] = cfg.DATASETS.TEST[0]
        ret["ORACLE"] = cfg.ORACLE
        ret["obj_lambda"] = cfg.MODEL.PARTCLIPSEG.OBJ_LAMBDA if hasattr(cfg.MODEL, "PARTCLIPSEG") else 1.0
        ret["part_lambda"] = cfg.MODEL.PARTCLIPSEG.PART_LAMBDA if hasattr(cfg.MODEL, "PARTCLIPSEG") else 1.0
        ret["use_attention"] = cfg.MODEL.PARTCLIPSEG.USE_ATTENTION if hasattr(cfg.MODEL, "PARTCLIPSEG") else False
        ret["attn_mask_threshold"] = cfg.MODEL.PARTCLIPSEG.ATTN_MASK_THRESHOLD if hasattr(cfg.MODEL, "PARTCLIPSEG") else 0.3
        ret["attn_sep_lambda"] = cfg.MODEL.PARTCLIPSEG.ATTN_SEP_LAMBDA if hasattr(cfg.MODEL, "PARTCLIPSEG") else 0.1
        ret["attn_enh_lambda"] = cfg.MODEL.PARTCLIPSEG.ATTN_ENH_LAMBDA if hasattr(cfg.MODEL, "PARTCLIPSEG") else 0.01
        return ret

    def init_train_metadata(self, train_dataset):
        '''
        text_classes            : object-specific part class names
        obj_classes             : object class names
        part_classes            : part class names
        text_to_part_map        : text class to part class index
        obj_to_obj_in_part_map  : object class index to object in part class index
        text_to_obj_in_part_map : text class index to object in part class index
        obj_in_part_to_text     : object in part class index to text class index

        text                    : object-specific part category names
        part_classes            : generalized part category names

        '''
        train_text_classes = MetadataCatalog.get(train_dataset).stuff_classes

        # "bus's door" -> "bus door" / "bus" / "door"
        self.train_text_classes = [c.replace('\'s', '') for c in train_text_classes]
        self.train_obj_classes = MetadataCatalog.get(train_dataset).obj_classes
        self.train_part_classes = sorted(list(set([c.split('\'s')[1].strip() for c in train_text_classes])))
        self.train_obj_in_part_classes = sorted(list(set([c.split('\'s')[0].strip() for c in train_text_classes])))

        # object-speicific part category names -> generalized part category name
        self.train_text_to_part_map = torch.full(
            (self.ignore_label + 1,),  # 255 + 1
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )

        # object category names -> object-level category name
        self.train_obj_to_obj_in_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )

        # object-speicific part category names -> object-level category name
        self.train_text_to_obj_in_part_map = torch.full(
            (len(self.train_text_classes),),
            len(self.train_text_classes),
            dtype=torch.long,
            device=self.device
        )

        # {0: 1, 1: 34, 2: 40, 3: 35, 4: 7, 5: 38, 6: 38, 7: 31, 8: 14, ...}
        self.train_class_to_part = {
            index: self.train_part_classes.index(
                class_text.split('\'s')[1].strip()
            ) for index, class_text in enumerate(train_text_classes)
        }
        for index, part_index in self.train_class_to_part.items():
            self.train_text_to_part_map[index] = part_index

        # {0: 0, 1: 1, 2: 255, 3: 2, 4: 3, 5: 4, 6: 255, 7: 5, 8: 255, ...}
        self.train_obj_to_obj_in_part = {}
        for index, class_text in enumerate(self.train_obj_classes):
            if class_text in self.train_obj_in_part_classes:
                self.train_obj_to_obj_in_part[index] = self.train_obj_in_part_classes.index(class_text)
            else:
                self.train_obj_to_obj_in_part[index] = self.ignore_label

        for index, obj_in_part_index in self.train_obj_to_obj_in_part.items():
            self.train_obj_to_obj_in_part_map[index] = obj_in_part_index

        # {0: [0, 1, 2, 3, 4, 5], 1: [6, 7, 8, 9, 10], 2: [11, 12], ...}
        self.train_obj_in_part_to_text = defaultdict(list)
        for index, class_text in enumerate(train_text_classes):
            obj_class, part = class_text.split('\'s', maxsplit=1)
            obj_in_part_index = self.train_obj_in_part_classes.index(obj_class)
            self.train_obj_in_part_to_text[obj_in_part_index].append(index)
            self.train_text_to_obj_in_part_map[index] = obj_in_part_index

        return None

    def init_test_metadata(self, test_dataset):
        # TODO:
        # consistency (style), sync with init_train_metadata()

        test_text_classes = MetadataCatalog.get(test_dataset).stuff_classes

        self.ori_test_text_classes = test_text_classes
        self.test_text_classes = [c.replace('\'s', '') for c in test_text_classes]
        self.test_obj_classes = MetadataCatalog.get(test_dataset).obj_classes
        self.test_part_classes = sorted(list(set([c.split('\'s')[1].strip() for c in test_text_classes])))
        self.test_class_to_part = {
            index: self.test_part_classes.index(
                class_text.split('\'s')[1].strip()
            ) for index, class_text in enumerate(test_text_classes)
        }
        self.test_obj_in_part_classes = sorted(list(set([c.split('\'s')[0].strip() for c in test_text_classes])))

        self.test_text_to_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        self.test_obj_in_part_to_obj_map = torch.full(
            (len(self.test_obj_in_part_classes) + 1,),
            len(self.test_obj_classes),
            dtype=torch.long,
            device=self.device
        )
        self.test_text_to_obj_in_part_map = torch.full(
            (len(self.test_text_classes),),
            len(self.test_text_classes),
            dtype=torch.long,
            device=self.device
        )
        self.test_obj_to_obj_in_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        for index, part_index in self.test_class_to_part.items():
            self.test_text_to_part_map[index] = part_index

        for index, class_text in enumerate(self.test_obj_in_part_classes):
            self.test_obj_in_part_to_obj_map[index] = self.test_obj_classes.index(class_text)

        self.test_obj_to_obj_in_part = {}
        for index, class_text in enumerate(self.test_obj_classes):
            if class_text in self.test_obj_in_part_classes:
                self.test_obj_to_obj_in_part[index] = self.test_obj_in_part_classes.index(class_text)
            else:
                self.test_obj_to_obj_in_part[index] = self.ignore_label

        for index, obj_in_part_index in self.test_obj_to_obj_in_part.items():
            self.test_obj_to_obj_in_part_map[index] = obj_in_part_index

        for index, class_text in enumerate(test_text_classes):
            obj_class, part = class_text.split('\'s', maxsplit=1)
            obj_in_part_index = self.test_obj_in_part_classes.index(obj_class)
            self.test_text_to_obj_in_part_map[index] = obj_in_part_index

        return None

    """
    def preds_to_semantic_inds(self, preds, threshold):
        flat_preds = preds.reshape((preds.shape[0], -1))
        # Initialize a dummy "unlabeled" mask with the threshold
        flat_preds_with_treshold = torch.full(
            (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
        )
        flat_preds_with_treshold[1: preds.shape[0] + 1, :] = flat_preds

        # Get the top mask index for each pixel
        semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
            (preds.shape[-2], preds.shape[-1])
        )
        return semantic_inds
    """

    def partclipseg_segmentation(
        self, model, images, test_text,
        num_text_classes, num_part_classes, num_obj_classes,
        part_index_map, obj_index_map, device
    ):
        logits = []
        input = self.partclipseg_processor(
            images=images, return_tensors="pt"
        ).to(device)
        if self.training:
            text = self.train_text_encoding
        else:
            text = test_text

        input["part_index_map"] = part_index_map
        input["obj_index_map"] = obj_index_map
        input.update(text)
        input = input.to(device)

        # no need to be loaded to GPU
        input["num_text_classes"] = num_text_classes
        input["num_part_classes"] = num_part_classes
        input["output_attentions"] = self.use_attention

        outputs = model(**input)
        logits = outputs.logits
        part_obj_logits = outputs.part_obj_logits

        if self.use_attention:
            attentions = outputs.decoder_output.attentions
            bs = len(images)
            agg_attentions = self.merge_attention(
                bs, attentions, num_text_classes, num_part_classes, num_obj_classes,
                part_index_map, obj_index_map, device
            )
            return (logits, part_obj_logits), agg_attentions

        return logits, part_obj_logits

    def merge_attention(
        self, bs, attentions, num_text_classes, num_part_classes, num_obj_classes,
        part_index_map, obj_index_map, device
    ):
        batch_class_num, n_head, h, w = attentions[0].shape
        class_num = batch_class_num // bs

        weighted_attentions = torch.zeros(bs, class_num, h, w, device=device)
        for i in range(len(attentions)):
            if i in self.attn_level:
                attention_reshaped = attentions[i].reshape(bs, -1, n_head, h, w)
                attention_head_avg = attention_reshaped.mean(dim=2)
                weighted_attentions += attention_head_avg

        weighted_attentions = weighted_attentions / len(self.attn_level)
        agg_attentions = torch.zeros(bs, num_text_classes + num_obj_classes, h, w, device=device)
        agg_attentions[:, :num_text_classes] += torch.gather(weighted_attentions[:, :num_part_classes], 1, part_index_map.permute(0, 3, 1, 2).repeat(bs, 1, h, w))
        agg_attentions[:, :num_text_classes] += torch.gather(weighted_attentions[:, num_part_classes:-1], 1, obj_index_map.permute(0, 3, 1, 2).repeat(bs, 1, h, w))
        agg_attentions[:, :num_text_classes] /= 2.
        agg_attentions[:, num_text_classes:] += weighted_attentions[:, num_part_classes:-1]

        return agg_attentions

    def inference(self, batched_inputs):
        image = Image.open(batched_inputs[0]["file_name"])
        image = image.convert("RGB")

        test_text_classes = self.ori_test_text_classes

        num_text_classes = len(test_text_classes)
        num_part_classes = len(self.test_part_classes)
        num_obj_classes = len(self.test_obj_in_part_classes)

        with torch.no_grad():
            part_index_map = self.test_text_to_part_map[None, None, None, :num_text_classes]
            obj_index_map = self.test_text_to_obj_in_part_map[None, None, None, :]

            outputs = self.partclipseg_segmentation(
                self.partclipseg_model,
                [image],
                self.partclipseg_processor.tokenizer(
                    [part.replace('\'s', '') for part in self.test_part_classes + self.test_obj_in_part_classes],
                    return_tensors="pt",
                    padding="max_length"
                ),
                num_text_classes, num_part_classes, num_obj_classes,
                part_index_map, obj_index_map,
                self.device,
            )

            if self.use_attention:
                outputs, attentions = outputs

            logits, part_obj_logits = outputs

            part_logits = logits[:, :num_part_classes]
            obj_logits_with_bg = logits[:, num_part_classes:]

            upscaled_logits = nn.functional.interpolate(
                part_obj_logits[:, :-1, :, :],
                size=(image.size[1], image.size[0]),
                mode="bilinear",
            )

            clipseg_preds = torch.sigmoid(upscaled_logits)
            preds = clipseg_preds.squeeze(0)

            upscaled_logits_with_bg = nn.functional.interpolate(
                part_obj_logits,
                size=(image.size[1], image.size[0]),
                mode="bilinear",
            )

            clipseg_preds_with_bg = torch.sigmoid(upscaled_logits_with_bg)
            preds_with_bg = clipseg_preds_with_bg.squeeze(0)

            upscaled_part_logits = nn.functional.interpolate(
                part_logits,
                size=(image.size[1], image.size[0]),
                mode="bilinear",
            )

            clipseg_part_preds = torch.sigmoid(upscaled_part_logits)
            part_preds = clipseg_part_preds.squeeze(0)

            upscaled_obj_logits_with_bg = nn.functional.interpolate(
                obj_logits_with_bg,
                size=(image.size[1], image.size[0]),
                mode="bilinear",
            )

            clipseg_obj_preds_with_bg = torch.sigmoid(upscaled_obj_logits_with_bg)
            obj_preds_with_bg = clipseg_obj_preds_with_bg.squeeze(0)

            gt_objs = [
                self.test_obj_classes[i]
                for i in torch.unique(batched_inputs[0]["sem_seg"]) if i != self.ignore_label
            ]
            part_inds = set()
            for obj in gt_objs:
                for i, part in enumerate(test_text_classes):
                    if part.split("\'s", maxsplit=1)[0] == obj:
                        part_inds.add(i)
            no_part_ids = [i for i in range(len(test_text_classes)) if i not in part_inds]

            obj_preds_ = obj_preds_with_bg.argmax(0)
            obj_preds_mask = F.one_hot(obj_preds_, num_classes=num_obj_classes + 1).float().permute(2, 0, 1)

            preds_with_bg[:-1] *= torch.gather(
                obj_preds_mask[:-1],
                0,
                self.test_text_to_obj_in_part_map[:, None, None].repeat(1, preds_with_bg.shape[1], preds_with_bg.shape[2])
            )
            preds_with_bg[-1] *= obj_preds_mask[-1]
            # preds_with_bg[-1][obj_preds_mask[-1] == 1] = obj_preds_mask[-1][obj_preds_mask[-1] == 1]

            obj_preds_with_bg = self.test_obj_in_part_to_obj_map[obj_preds_]
            preds[no_part_ids] = 0.0

            if self.use_attention:
                bs, _, h, w = logits.shape
                attns = self.get_attn_masks(bs, h, w, attentions, batched_inputs, num_text_classes, num_obj_classes)

        results = [{
            "sem_seg": preds,
            "sem_seg_with_bg": preds_with_bg,
            "obj_sem_seg_with_bg": obj_preds_with_bg,
            "part_sem_seg": part_preds,
            "attns": attns if self.use_attention else None,
        }]

        return results


    # TODO: modulate forward and inference

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        # images = [Image.open(x["file_name"]).convert("RGB") for x in batched_inputs]
        images = [x["image"].to(self.device) for x in batched_inputs]
        gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
        obj_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]

        num_text_classes = len(self.train_text_classes)
        num_part_classes = len(self.train_part_classes)
        num_obj_classes = len(self.train_obj_in_part_classes)

        part_index_map = self.train_text_to_part_map[None, None, None, :num_text_classes]
        obj_index_map = self.train_text_to_obj_in_part_map[None, None, None, :]

        outputs = self.partclipseg_segmentation(
            self.partclipseg_model, images, None,
            num_text_classes, num_part_classes, num_obj_classes,
            part_index_map, obj_index_map, self.device
        )  # [b, n, h, w]

        if self.use_attention:
            outputs, attentions = outputs

        logits, part_obj_logits = outputs
        bs, n, h, w = logits.shape

        targets = torch.stack(
            [
                nn.functional.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=(logits.shape[-2], logits.shape[-1]),
                    mode="nearest"
                ) for gt in gts
            ]
        ).long().squeeze(1).squeeze(1)  # [b,h,w]

        obj_targets = torch.stack(
            [
                nn.functional.interpolate(
                    gt.unsqueeze(0).unsqueeze(0).float(),
                    size=(logits.shape[-2], logits.shape[-1]),
                    mode="nearest"
                ) for gt in obj_gts
            ]
        ).long().squeeze(1).squeeze(1)

        part_targets = self.train_text_to_part_map[targets]
        obj_in_part_targets = self.train_obj_to_obj_in_part_map[obj_targets]

        # logits = logits[:, :-1] # [b,n-1,h,w]
        num_classes = logits.shape[1]
        mask = targets != self.ignore_label                       # [b,h,w]
        part_mask = part_targets != self.ignore_label
        obj_mask = obj_in_part_targets != self.ignore_label
        logits = logits.permute(0, 2, 3, 1)                       # [b,h,w,n]
        part_obj_logits = part_obj_logits.permute(0, 2, 3, 1)     # [b,h,w,n]
        _targets = torch.zeros(logits.shape, device=self.device)  # _targets[..., -1] = 1
        part_obj_targets = torch.zeros((bs, h, w, num_text_classes + 1), device=self.device)
        text_onehot = F.one_hot(targets[mask], num_classes=num_text_classes).float()
        part_onehot = F.one_hot(part_targets[part_mask], num_classes=num_part_classes).float()
        obj_onehot = F.one_hot(obj_in_part_targets[obj_mask], num_classes=num_obj_classes).float()

        _targets[..., :num_part_classes][part_mask] = part_onehot
        _targets[..., num_part_classes:-1][obj_mask] = obj_onehot
        _targets[..., -1][~obj_mask] = 1
        part_obj_targets[..., :num_text_classes][mask] = text_onehot
        part_obj_targets[..., -1][~obj_mask] = 1

        # weights
        class_weight = torch.ones(num_classes).to(self.device)
        class_weight[-1] = 0.05


        # TODO: param (args)
        results_weight = torch.ones(num_text_classes + 1).to(self.device)
        results_weight[-1] = 0.05


        # ---------------------------------------------------------------------
        # Generalized Parts with Object-level Contexts
        # ---------------------------------------------------------------------
        #   - part_obj_loss : object-specific part
        #   - obj_loss      : object-level guidance
        #   - part_loss     : generalized part guidance
        # ---------------------------------------------------------------------

        part_obj_loss, part_loss, obj_loss = 0.0, 0.0, 0.0
        if part_mask.sum() > 0:
            part_obj_loss = F.binary_cross_entropy_with_logits(
                part_obj_logits[part_mask],
                part_obj_targets[part_mask],
                weight=results_weight,
            )
            part_loss = F.binary_cross_entropy_with_logits(
                logits[..., :num_part_classes][part_mask],
                _targets[..., :num_part_classes][part_mask],
                weight=class_weight[:num_part_classes],
            )

        # TODO: check background (if all 0)
        if obj_mask.sum() > 0:
            part_obj_loss += F.binary_cross_entropy_with_logits(
                part_obj_logits[~obj_mask],
                part_obj_targets[~obj_mask],
                weight=results_weight,
            )

        obj_loss += F.binary_cross_entropy_with_logits(
            logits[..., num_part_classes:],
            _targets[..., num_part_classes:],
            weight=class_weight[num_part_classes:],
        )

        loss = part_obj_loss + self.obj_lambda * obj_loss + self.part_lambda * part_loss


        # ---------------------------------------------------------------------
        # Attention Losses
        # ---------------------------------------------------------------------
        #   - loss_sep : seperation loss
        #   - loss_enh : enhance loss
        # ---------------------------------------------------------------------

        if self.use_attention:
            # [b, h, w, n] -> [b, n, h, w]
            masks = part_obj_targets.permute(0, 3, 1, 2)
            obj_masks = _targets[..., num_part_classes:-1].permute(0, 3, 1, 2)

            loss_sep, loss_enh = self.compute_attn_loss(
                bs, attentions, num_text_classes, masks, obj_masks,
                targets, part_obj_targets, obj_in_part_targets
            )

            attn_loss = (
                self.attn_sep_lambda * loss_sep + self.attn_enh_lambda * loss_enh
            )

            # TODO: attention loss
            loss += attn_loss

        if torch.isnan(loss) or torch.isinf(loss):
            # initiate loss to be zero tensor
            loss = torch.nan_to_num(loss)
            loss.requires_grad_(True)

        losses = {"loss_sem_seg": loss}
        return losses



    def compute_attn_loss(
        self, bs, attentions, num_text_classes, masks, obj_masks,
        targets, part_obj_targets, obj_in_part_targets
    ):
        # attentions: [8, 90, 485, 485] (485:22*22+1)
        _, _, h, w = attentions.shape
        res = int(math.sqrt(h - 1))     # res: 22, h: 485

        smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).to(self.device)

        # masks_resized = F.max_pool2d(masks, kernel_size=int(img_h/res), stride=int(img_h/res), padding=0)
        masks_resized = F.interpolate(masks, size=(res, res), mode='nearest')  # [b, n, r, r]
        obj_masks_resized = F.interpolate(obj_masks, size=(res, res), mode='nearest')  # [b, n, r, r]

        loss_sep, loss_enh = 0.0, 0.0
        attn = attentions[..., 1:, 1:]  # [b,n,r*r+1,r*r+1] -> [b,n,r*r,r*r]
        obj_num = 0

        for batch_idx in range(bs):
            obj_in_part_list = obj_in_part_targets[batch_idx].unique().tolist()[:-1]

            for obj_idx in obj_in_part_list:
                part_indices = targets[batch_idx].unique().tolist()
                part_indices.remove(self.ignore_label) if self.ignore_label in part_indices else None
                union_part_attn = torch.zeros((1, res, res), device=self.device)  # [b,r,r]
                sum_attn_mask = torch.zeros((1, res, res), device=self.device)

                if len(part_indices) == 0:
                    continue

                obj_num += 1
                obj_attn_mask = obj_masks_resized[batch_idx, obj_idx].float()
                obj_attn_mask = obj_attn_mask.unsqueeze(0)
                min_of_max_part_attn = 1.0

                for j in part_indices:
                    part_attn = attn[batch_idx, j].reshape(1, res, res, -1)                     # [b,r,r,r*r]
                    masked_part_attn = part_attn * masks_resized[batch_idx, j].unsqueeze(-1)    # [b,r,r,r*r]
                    masked_part_attn_reshaped = masked_part_attn.reshape(1, res * res, -1)      # [b,r*r,r*r]
                    avg_part_attn = masked_part_attn_reshaped.mean(dim=1).reshape(1, res, res)  # [b,r,r]
                    norm_avg_part_attn = (avg_part_attn - avg_part_attn.min()) / (avg_part_attn.max() - avg_part_attn.min() + 1e-9)

                    # smoothing attention
                    padded_avg_part_attn = F.pad(avg_part_attn.unsqueeze(1), (1, 1, 1, 1), mode='reflect')            # [b,1,r,r]
                    padded_norm_avg_part_attn = F.pad(norm_avg_part_attn.unsqueeze(1), (1, 1, 1, 1), mode='reflect')  # [b,1,r,r]
                    smoothed_avg_part_attn = smoothing(padded_avg_part_attn).squeeze(1)                               # [b,r,r]
                    smoothed_norm_avg_part_attn = smoothing(padded_norm_avg_part_attn).squeeze(1)                     # [b,r,r]

                    min_of_max_part_attn = min(min_of_max_part_attn, smoothed_avg_part_attn.max())
                    part_attn_mask = torch.zeros((1, res, res), device=self.device)
                    part_attn_mask[obj_attn_mask == 1] = (smoothed_norm_avg_part_attn[obj_attn_mask == 1] > self.attn_mask_threshold).float()

                    union_part_attn = torch.max(union_part_attn, part_attn_mask)
                    sum_attn_mask += part_attn_mask

                # separation loss
                loss_sep += (sum_attn_mask > 1).float().sum() / ((sum_attn_mask >= 1).float().sum() + 1e-9)

                # enhancement loss
                loss_enh += (1. - min_of_max_part_attn)

        obj_num = 1 if obj_num == 0 else obj_num
        loss_sep /= (obj_num * bs)
        loss_enh /= (obj_num * bs)

        return loss_sep, loss_enh



    # For attention visualization
    def get_attn_masks(
        self, bs, h, w, attentions, batched_inputs, num_text_classes, num_obj_classes
    ):
        with torch.no_grad():
            attns = {}
            smoothing = GaussianSmoothing(
                channels=1, kernel_size=3, sigma=0.5, dim=2
            ).to(self.device)

            gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
            obj_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]

            targets = torch.stack([nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="nearest"
            ) for gt in gts]).long().squeeze(1).squeeze(1)  # [b,h,w]

            obj_targets_ = torch.stack([nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(h, w),
                mode="nearest"
            ) for gt in obj_gts]).long().squeeze(1).squeeze(1)

            part_obj_targets = torch.zeros(
                (bs, h, w, num_text_classes + 1),
                device=self.device
            )
            obj_in_part_targets = self.test_obj_to_obj_in_part_map[obj_targets_]

            mask = targets != self.ignore_label
            obj_mask = obj_in_part_targets != self.ignore_label

            text_onehot = F.one_hot(targets[mask], num_classes=num_text_classes).float()
            obj_onehot = F.one_hot(obj_in_part_targets[obj_mask], num_classes=num_obj_classes).float()

            part_obj_targets[..., :num_text_classes][mask] = text_onehot
            part_obj_targets[..., -1][~obj_mask] = 1.

            obj_targets = torch.zeros((bs, h, w, num_obj_classes), device=self.device)
            obj_targets[:, :, :, :][obj_mask] = obj_onehot

            # [b, h, w, n] -> [b, n, h, w]
            masks = part_obj_targets.permute(0, 3, 1, 2)
            obj_masks = obj_targets.permute(0, 3, 1, 2)

            attns = {}
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).to(self.device)

            # attentions: [8, 90, 485, 485] (485:22*22+1)
            _, _, attn_h, attn_w = attentions.shape
            res = int(math.sqrt(attn_h - 1))     # res: 22, h: 485

            attn = attentions[..., 1:, 1:].squeeze(0)

            masks_resized = F.interpolate(masks, size=(res, res), mode='nearest')  # [b, n, r, r]
            obj_masks_resized = F.interpolate(obj_masks, size=(res, res), mode='nearest')  # [b, n, r, r]

            obj_in_part_list = obj_in_part_targets.unique().tolist()
            obj_in_part_list.remove(self.ignore_label) if self.ignore_label in obj_in_part_list else None

            for obj in obj_in_part_list:
                union_part_attn = torch.zeros((1, res, res), device=self.device)

                part_indices = targets.unique().tolist()
                part_indices.remove(self.ignore_label) if self.ignore_label in part_indices else None

                for j in part_indices:
                    part_attn = attn[j, :, :].reshape(res, res, -1).unsqueeze(0)
                    masked_part_attn = part_attn * masks_resized[:, j].unsqueeze(-1)

                    masked_part_attn_reshaped = masked_part_attn.reshape(1, res * res, -1)

                    avg_part_attn = masked_part_attn_reshaped.mean(dim=1).reshape(1, res, res)
                    norm_avg_part_attn = (avg_part_attn - avg_part_attn.min()) / (avg_part_attn.max() - avg_part_attn.min() + 1e-9)

                    padded_norm_avg_part_attn = F.pad(norm_avg_part_attn.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    smoothed_norm_avg_part_attn = smoothing(padded_norm_avg_part_attn).squeeze(0)

                    attns[f"smoothed_part_{self.test_text_classes[j].replace(' ', '_')}"] = smoothed_norm_avg_part_attn
                    attns[f"avg_part_{self.test_text_classes[j].replace(' ', '_')}"] = avg_part_attn

                    union_part_attn = torch.max(union_part_attn, smoothed_norm_avg_part_attn)

                temp_obj_attn = attn[num_text_classes + obj, :, :].reshape(res, res, -1).unsqueeze(0)
                masked_obj_attn = temp_obj_attn * obj_masks_resized[:, obj].unsqueeze(-1)
                masked_obj_attn_reshaped = masked_obj_attn.reshape(1, res * res, -1)

                obj_attn = masked_obj_attn_reshaped.mean(dim=1).reshape(1, res, res)
                norm_obj_attn = (obj_attn - obj_attn.min()) / (obj_attn.max() - obj_attn.min() + 1e-9)

                padded_norm_obj_attn = F.pad(norm_obj_attn.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                smoothed_norm_obj_attn = smoothing(padded_norm_obj_attn).squeeze(0)

                norm_union_part_attn = (union_part_attn - union_part_attn.min()) / (union_part_attn.max() - union_part_attn.min() + 1e-9)

                attns[f"smoothed_obj_{self.test_obj_in_part_classes[obj].replace(' ', '_')}"] = smoothed_norm_obj_attn
                attns[f"union_part_{self.test_obj_in_part_classes[obj].replace(' ', '_')}"] = norm_union_part_attn

        return attns