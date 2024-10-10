# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .utils import load_binary_mask, load_obj_part_sem_seg

OBJ_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
OBJ_BASE_CLASS_NAMES = [
    c for i, c in enumerate(OBJ_CLASS_NAMES) if c not in ["bird", "car", "dog", "sheep", "motorbike"]
]


CLASS_NAMES = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
               "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", "bottle's body", "bottle's cap", "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", 
               "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
               "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
               "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", 
               "horse's muzzle", "horse's hoof", "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", 
               "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand","pottedplant's pot", "pottedplant's plant", 
               "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", 
               "train's coach", "tvmonitor's screen"]

BASE_CLASS_NAMES = [
    c for i, c in enumerate(CLASS_NAMES) if c not in ["bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot",
                                                      "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window",
                                                      "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle",
                                                      "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn",
                                                      "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight"]
]


PART_CLASS_NAMES = sorted(list(set([c.split('\'s')[1].strip() for c in CLASS_NAMES])))

PART_BASE_CLASS_NAMES = [
    c for i, c in enumerate(PART_CLASS_NAMES) if c not in ["chainwheel"]
]


# object and part colors
OBJ_COLORS = {
    "aeroplane": [220, 20, 60], "bicycle": [219, 142, 185], "bird": [220, 120, 60], "boat": [219, 42, 185], "bottle": [20, 20, 60],
    "bus": [19, 142, 185], "car": [220, 20, 160], "cat": [119, 142, 185], "chair": [220, 220, 60], "cow": [250, 0, 30],
    "diningtable": [165, 42, 42], "dog": [255, 77, 255], "horse": [0, 226, 252], "motorbike": [182, 182, 255], "person": [0, 82, 0],
    "pottedplant": [120, 166, 157], "sheep": [110, 76, 0], "sofa": [174, 57, 255], "train": [199, 100, 0], "tvmonitor": [72, 0, 118]
}

CLASS_COLORS = {
    "aeroplane's body": [231, 4, 237], "aeroplane's stern": [116, 80, 69], "aeroplane's wing": [214, 86, 123], "aeroplane's tail": [22, 174, 172], "aeroplane's engine": [197, 128, 182], "aeroplane's wheel": [82, 197, 247], 
    "bicycle's wheel": [240, 6, 206], "bicycle's saddle": [0, 67, 113], "bicycle's handlebar": [112, 158, 137], "bicycle's chainwheel": [89, 19, 193], "bicycle's headlight": [255, 182, 87], 
    "bird's wing": [61, 245, 238], "bird's tail": [68, 62, 33], "bird's head": [104, 202, 100], "bird's eye": [242, 114, 163], "bird's beak": [242, 41, 140], "bird's torso": [189, 249, 133], "bird's neck": [158, 181, 70], "bird's leg": [55, 126, 0], "bird's foot": [225, 182, 182], 
    "bottle's body": [172, 155, 96], "bottle's cap": [185, 14, 216], 
    "bus's wheel": [73, 10, 81], "bus's headlight": [83, 15, 206], "bus's front": [28, 29, 98], "bus's side": [25, 219, 63], "bus's back": [76, 11, 160], "bus's roof": [160, 204, 37], "bus's mirror": [253, 98, 118], "bus's license plate": [252, 213, 76], "bus's door": [104, 232, 186], "bus's window": [182, 248, 35], 
    "car's wheel": [211, 58, 216], "car's headlight": [253, 195, 114], "car's front": [57, 193, 239], "car's side": [193, 130, 234], "car's back": [226, 87, 2], "car's roof": [40, 117, 231], "car's mirror": [244, 27, 39], "car's license plate": [82, 204, 130], "car's door": [29, 145, 220], "car's window": [51, 163, 166], 
    "cat's tail": [22, 29, 245], "cat's head": [148, 109, 203], "cat's eye": [221, 235, 212], "cat's torso": [68, 44, 17], "cat's neck": [54, 164, 161], "cat's leg": [114, 251, 219], "cat's nose": [99, 126, 184], "cat's paw": [93, 47, 109], "cat's ear": [25, 226, 114], 
    "cow's tail": [186, 133, 2], "cow's head": [243, 197, 251], "cow's eye": [99, 87, 120], "cow's torso": [153, 207, 125], "cow's neck": [250, 136, 178], "cow's leg": [171, 46, 206], "cow's ear": [194, 7, 114], "cow's muzzle": [242, 122, 177], "cow's horn": [202, 242, 232], 
    "dog's tail": [121, 134, 63], "dog's head": [33, 191, 131], "dog's eye": [225, 95, 66], "dog's torso": [245, 11, 186], "dog's neck": [129, 83, 21], "dog's leg": [185, 76, 143], "dog's nose": [214, 234, 112], "dog's paw": [232, 132, 164], "dog's ear": [124, 25, 24], "dog's muzzle": [90, 58, 214], 
    "horse's tail": [57, 239, 109], "horse's head": [154, 96, 130], "horse's eye": [221, 98, 183], "horse's torso": [183, 92, 254], "horse's neck": [206, 114, 11], "horse's leg": [214, 238, 15], "horse's ear": [230, 145, 183], "horse's muzzle": [213, 203, 88], "horse's hoof": [223, 82, 151], 
    "motorbike's wheel": [124, 107, 252], "motorbike's saddle": [254, 180, 116], "motorbike's handlebar": [163, 225, 169], "motorbike's headlight": [119, 52, 22], 
    "person's head": [40, 30, 77], "person's eye": [237, 64, 148], "person's torso": [126, 117, 235], "person's neck": [13, 146, 62], "person's leg": [219, 60, 25], "person's foot": [45, 66, 254], "person's nose": [101, 145, 176], "person's ear": [49, 186, 234], "person's eyebrow": [164, 26, 135], "person's mouth": [31, 78, 216], "person's hair": [95, 148, 151], "person's lower arm": [1, 251, 147], "person's upper arm": [53, 133, 129], "person's hand": [58, 227, 163], 
    "pottedplant's pot": [80, 197, 134], "pottedplant's plant": [236, 149, 209], 
    "sheep's tail": [218, 195, 52], "sheep's head": [31, 13, 13], "sheep's eye": [34, 207, 63], "sheep's torso": [241, 111, 194], "sheep's neck": [121, 163, 14], "sheep's leg": [38, 79, 231], "sheep's ear": [249, 117, 121], "sheep's muzzle": [172, 128, 70], "sheep's horn": [97, 144, 104], 
    "train's headlight": [24, 148, 249], "train's head": [99, 250, 180], "train's front": [147, 90, 40], "train's side": [177, 157, 141], "train's back": [93, 241, 104], "train's roof": [15, 236, 94], "train's coach": [143, 232, 181], 
    "tvmonitor's screen": [186, 6, 38]
}


PART_CLASS_COLORS = {
    'back': [165, 42, 58], 'beak': [42, 214, 101], 'body': [247, 155, 10], 'cap': [203, 161, 225], 'chainwheel': [144, 32, 42], 'coach': [59, 0, 32], 
    'door': [33, 217, 217], 'ear': [143, 159, 244], 'engine': [181, 240, 27], 'eye': [71, 205, 244], 'eyebrow': [180, 240, 251], 'foot': [252, 199, 41], 
    'front': [7, 68, 174], 'hair': [241, 71, 29], 'hand': [210, 158, 8], 'handlebar': [211, 175, 110], 'head': [106, 65, 20], 'headlight': [186, 249, 129], 
    'hoof': [33, 189, 93], 'horn': [196, 200, 102], 'leg': [130, 2, 225], 'license plate': [213, 167, 144], 'lower arm': [17, 212, 110], 'mirror': [162, 81, 182], 
    'mouth': [252, 160, 9], 'muzzle': [184, 143, 141], 'neck': [121, 114, 90], 'nose': [125, 70, 43], 'paw': [111, 51, 252], 'plant': [215, 54, 24], 'pot': [83, 240, 55], 
    'roof': [138, 204, 25], 'saddle': [199, 155, 29], 'screen': [192, 52, 176], 'side': [214, 66, 158], 'stern': [126, 229, 103], 'tail': [98, 245, 66], 'torso': [196, 184, 186], 
    'upper arm': [175, 182, 15], 'wheel': [38, 193, 21], 'window': [140, 162, 48], 'wing': [45, 190, 14]
}


obj_map = {OBJ_CLASS_NAMES.index(c): i for i,c in enumerate(OBJ_BASE_CLASS_NAMES)}
obj_part_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}


def _get_voc_obj_part_meta(cat_list, obj_list):
    ret = {
        "stuff_classes": cat_list,
        "obj_classes": obj_list,
        "obj_base_classes": OBJ_BASE_CLASS_NAMES,
        "part_classes": PART_CLASS_NAMES,
        "part_base_classes": PART_BASE_CLASS_NAMES,
        "stuff_colors": [CLASS_COLORS[c] for c in cat_list],
        "obj_colors": [OBJ_COLORS[c] for c in obj_list],
        "part_colors": [PART_CLASS_COLORS[c] for c in PART_CLASS_NAMES],
    }
    return ret


def register_pascal_part_116(root):
    root = os.path.join(root, "PascalPart116")
    meta = _get_voc_obj_part_meta(CLASS_NAMES, OBJ_CLASS_NAMES)
    base_meta = _get_voc_obj_part_meta(BASE_CLASS_NAMES, OBJ_BASE_CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2_part/train"),
        ("val", "images/val", "annotations_detectron2_part/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        
        ################################ part sem seg without object sem seg ############################
        all_name = f"voc_obj_part_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_obj_part_sem_seg(
                y, x, gt_ext="png", image_ext="jpg"
            ),
        )
        MetadataCatalog.get(all_name).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
            )
        if name == 'train':
            MetadataCatalog.get(all_name).set(**base_meta)
            MetadataCatalog.get(all_name).set(obj_map=obj_map)
            MetadataCatalog.get(all_name).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(all_name).set(**meta)
        MetadataCatalog.get(all_name).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        ################################ part sem seg with object sem seg ############################
        name_obj_cond = all_name + "_obj_condition"
        DatasetCatalog.register(
            name_obj_cond,
            lambda x=image_dir, y=gt_dir: load_binary_mask(
                y, x, gt_ext="png", image_ext="jpg", label_count="_obj_label_count.json"
            ),
        )
        MetadataCatalog.get(name_obj_cond).set(
                image_root=image_dir,
                sem_seg_root=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
            )
        if name == 'train':
            MetadataCatalog.get(name_obj_cond).set(**base_meta)
            MetadataCatalog.get(name_obj_cond).set(obj_map=obj_map)
            MetadataCatalog.get(name_obj_cond).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(name_obj_cond).set(**meta)
        MetadataCatalog.get(name_obj_cond).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in base_meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in base_meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        ################################ part sem seg in few shot setting ############################
        if name == 'train':
            name_few_shot = all_name + "_few_shot"
            DatasetCatalog.register(
                name_few_shot,
                lambda x=image_dir, y=gt_dir: load_obj_part_sem_seg(
                    y, x, gt_ext="png", image_ext="jpg", data_list=f'{root}/train_16shot.json'
                ),
            )
            MetadataCatalog.get(name_few_shot).set(
                    image_root=image_dir,
                    sem_seg_root=gt_dir,
                    evaluator_type="sem_seg",
                    ignore_label=255,
                )

            MetadataCatalog.get(name_few_shot).set(**meta)
            MetadataCatalog.get(name_few_shot).set(
                evaluation_set={
                    "base": [
                        meta["stuff_classes"].index(n) for n in meta["stuff_classes"]
                    ],
                },
                trainable_flag=[1] * len(meta["stuff_classes"]),
            )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_part_116(_root)

