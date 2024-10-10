# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

OBJ_CLASS_NAMES = [
    "school bus",
    "box turtle",
    "tench",
    "warplane",
    "schooner",
    "pirate",
    "garbage truck",
    "golden retriever",
    "killer whale",
    "bald eagle",
    "impala",
    "green lizard",
    "ambulance",
    "yawl",
    "Indian cobra",
    "albatross",
    "mountain bike",
    "goldfish",
    "moped",
    "ice bear",
    "beer bottle",
    "jeep",
    "Komodo dragon",
    "airliner",
    "tiger",
    "green mamba",
    "leopard",
    "gazelle",
    "minibus",
    "barracouta",
    "gorilla",
    "motor scooter",
    "giant panda",
    "water bottle",
    "wine bottle",
    "orangutan",
    "tree frog",
    "chimpanzee",
    "goose",
    "American alligator",
]

OBJ_NOVEL_CLASS_NAMES = [
    "ice bear", "impala", "golden retriever",
    "Indian cobra",
    "box turtle", "American alligator",
    "schooner",
    "tench",
    "bald eagle",
    "jeep", "school bus",
    "motor scooter",
    "chimpanzee",
    "wine bottle",
    "airliner",
]

OBJ_BASE_CLASS_NAMES = [c for c in OBJ_CLASS_NAMES if c not in OBJ_NOVEL_CLASS_NAMES]

CLASS_NAMES = [
    "impala's head",
    "impala's body",
    "impala's foot",
    "impala's tail",
    "barracouta's head",
    "barracouta's body",
    "barracouta's fin",
    "barracouta's tail",
    "albatross's head",
    "albatross's body",
    "albatross's wing",
    "albatross's foot",
    "albatross's tail",
    "garbage truck's body",
    "garbage truck's tier",
    "garbage truck's side mirror",
    "minibus's body",
    "minibus's tier",
    "minibus's side mirror",
    "orangutan's head",
    "orangutan's body",
    "orangutan's hand",
    "orangutan's foot",
    "orangutan's tail",
    "goldfish's head",
    "goldfish's body",
    "goldfish's fin",
    "goldfish's tail",
    "gorilla's head",
    "gorilla's body",
    "gorilla's hand",
    "gorilla's foot",
    "gorilla's tail",
    "ambulance's body",
    "ambulance's tier",
    "ambulance's side mirror",
    "motor scooter's body",
    "motor scooter's head",
    "motor scooter's seat",
    "motor scooter's tier",
    "yawl's body",
    "yawl's sail",
    "green lizard's head",
    "green lizard's body",
    "green lizard's foot",
    "green lizard's tail",
    "golden retriever's head",
    "golden retriever's body",
    "golden retriever's foot",
    "golden retriever's tail",
    "green mamba's head",
    "green mamba's body",
    "beer bottle's mouth",
    "beer bottle's body",
    "tiger's head",
    "tiger's body",
    "tiger's foot",
    "tiger's tail",
    "tree frog's head",
    "tree frog's body",
    "tree frog's foot",
    "tree frog's tail",
    "leopard's head",
    "leopard's body",
    "leopard's foot",
    "leopard's tail",
    "jeep's body",
    "jeep's tier",
    "jeep's side mirror",
    "chimpanzee's head",
    "chimpanzee's body",
    "chimpanzee's hand",
    "chimpanzee's foot",
    "chimpanzee's tail",
    "goose's head",
    "goose's body",
    "goose's wing",
    "goose's foot",
    "goose's tail",
    "water bottle's mouth",
    "water bottle's body",
    "American alligator's head",
    "American alligator's body",
    "American alligator's foot",
    "American alligator's tail",
    "giant panda's head",
    "giant panda's body",
    "giant panda's foot",
    "giant panda's tail",
    "tench's head",
    "tench's body",
    "tench's fin",
    "tench's tail",
    "wine bottle's mouth",
    "wine bottle's body",
    "ice bear's head",
    "ice bear's body",
    "ice bear's foot",
    "ice bear's tail",
    "pirate's body",
    "pirate's sail",
    "box turtle's head",
    "box turtle's body",
    "box turtle's foot",
    "box turtle's tail",
    "warplane's head",
    "warplane's body",
    "warplane's engine",
    "warplane's wing",
    "warplane's tail",
    "schooner's body",
    "schooner's sail",
    "Komodo dragon's head",
    "Komodo dragon's body",
    "Komodo dragon's foot",
    "Komodo dragon's tail",
    "Indian cobra's head",
    "Indian cobra's body",
    "gazelle's head",
    "gazelle's body",
    "gazelle's foot",
    "gazelle's tail",
    "mountain bike's body",
    "mountain bike's head",
    "mountain bike's seat",
    "mountain bike's tier",
    "school bus's body",
    "school bus's tier",
    "school bus's side mirror",
    "killer whale's head",
    "killer whale's body",
    "killer whale's fin",
    "killer whale's tail",
    "moped's body",
    "moped's head",
    "moped's seat",
    "moped's tier",
    "airliner's head",
    "airliner's body",
    "airliner's engine",
    "airliner's wing",
    "airliner's tail",
    "bald eagle's head",
    "bald eagle's body",
    "bald eagle's wing",
    "bald eagle's foot",
    "bald eagle's tail",
]

BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if c.split('\'s', maxsplit=1)[0] not in OBJ_NOVEL_CLASS_NAMES]

PART_CLASS_NAMES = [
    'body',
    'engine',
    'fin',
    'foot',
    'hand',
    'head',
    'mouth',
    'sail',
    'seat',
    'side mirror',
    'tail',
    'tier',
    'wing'
]

PART_BASE_CLASS_NAMES = [
    'body',
    'engine',
    'fin',
    'foot',
    'hand',
    'head',
    'mouth',
    'sail',
    'seat',
    'side mirror',
    'tail',
    'tier',
    'wing'
]


PART_IN_CATEGORIES = [
    {'id': 0, 'name': 'Quadruped Head', 'supercategory': 'Quadruped'},
    {'id': 1, 'name': 'Quadruped Body', 'supercategory': 'Quadruped'},
    {'id': 2, 'name': 'Quadruped Foot', 'supercategory': 'Quadruped'},
    {'id': 3, 'name': 'Quadruped Tail', 'supercategory': 'Quadruped'},
    {'id': 4, 'name': 'Biped Head', 'supercategory': 'Biped'},
    {'id': 5, 'name': 'Biped Body', 'supercategory': 'Biped'},
    {'id': 6, 'name': 'Biped Hand', 'supercategory': 'Biped'},
    {'id': 7, 'name': 'Biped Foot', 'supercategory': 'Biped'},
    {'id': 8, 'name': 'Biped Tail', 'supercategory': 'Biped'},
    {'id': 9, 'name': 'Fish Head', 'supercategory': 'Fish'},
    {'id': 10, 'name': 'Fish Body', 'supercategory': 'Fish'},
    {'id': 11, 'name': 'Fish Fin', 'supercategory': 'Fish'},
    {'id': 12, 'name': 'Fish Tail', 'supercategory': 'Fish'},
    {'id': 13, 'name': 'Bird Head', 'supercategory': 'Bird'},
    {'id': 14, 'name': 'Bird Body', 'supercategory': 'Bird'},
    {'id': 15, 'name': 'Bird Wing', 'supercategory': 'Bird'},
    {'id': 16, 'name': 'Bird Foot', 'supercategory': 'Bird'},
    {'id': 17, 'name': 'Bird Tail', 'supercategory': 'Bird'},
    {'id': 18, 'name': 'Snake Head', 'supercategory': 'Snake'},
    {'id': 19, 'name': 'Snake Body', 'supercategory': 'Snake'},
    {'id': 20, 'name': 'Reptile Head', 'supercategory': 'Reptile'},
    {'id': 21, 'name': 'Reptile Body', 'supercategory': 'Reptile'},
    {'id': 22, 'name': 'Reptile Foot', 'supercategory': 'Reptile'},
    {'id': 23, 'name': 'Reptile Tail', 'supercategory': 'Reptile'},
    {'id': 24, 'name': 'Car Body', 'supercategory': 'Car'},
    {'id': 25, 'name': 'Car Tier', 'supercategory': 'Car'},
    {'id': 26, 'name': 'Car Side Mirror', 'supercategory': 'Car'},
    {'id': 27, 'name': 'Bicycle Body', 'supercategory': 'Bicycle'},
    {'id': 28, 'name': 'Bicycle Head', 'supercategory': 'Bicycle'},
    {'id': 29, 'name': 'Bicycle Seat', 'supercategory': 'Bicycle'},
    {'id': 30, 'name': 'Bicycle Tier', 'supercategory': 'Bicycle'},
    {'id': 31, 'name': 'Boat Body', 'supercategory': 'Boat'},
    {'id': 32, 'name': 'Boat Sail', 'supercategory': 'Boat'},
    {'id': 33, 'name': 'Aeroplane Head', 'supercategory': 'Aeroplane'},
    {'id': 34, 'name': 'Aeroplane Body', 'supercategory': 'Aeroplane'},
    {'id': 35, 'name': 'Aeroplane Engine', 'supercategory': 'Aeroplane'},
    {'id': 36, 'name': 'Aeroplane Wing', 'supercategory': 'Aeroplane'},
    {'id': 37, 'name': 'Aeroplane Tail', 'supercategory': 'Aeroplane'},
    {'id': 38, 'name': 'Bottle Mouth', 'supercategory': 'Bottle'},
    {'id': 39, 'name': 'Bottle Body', 'supercategory': 'Bottle'}
]


CLASS_COLORS = {"impala's head": [29, 196, 114], "impala's body": [253, 158, 121], "impala's foot": [246, 58, 75], "impala's tail": [29, 144, 195], "barracouta's head": [173, 213, 35], "barracouta's body": [53, 171, 63], "barracouta's fin": [48, 214, 232], "barracouta's tail": [125, 151, 55], "albatross's head": [13, 19, 137], "albatross's body": [33, 48, 116], "albatross's wing": [18, 228, 217], "albatross's foot": [188, 139, 101], "albatross's tail": [86, 231, 188], "garbage truck's body": [29, 157, 247], "garbage truck's tier": [101, 65, 62], "garbage truck's side mirror": [207, 153, 113], "minibus's body": [131, 10, 168], "minibus's tier": [131, 135, 122], "minibus's side mirror": [238, 98, 0], "orangutan's head": [196, 254, 2], "orangutan's body": [42, 148, 171], "orangutan's hand": [106, 192, 252], "orangutan's foot": [82, 175, 61], "orangutan's tail": [93, 183, 105], "goldfish's head": [15, 15, 124], "goldfish's body": [101, 52, 14], "goldfish's fin": [147, 120, 55], "goldfish's tail": [99, 74, 225], "gorilla's head": [4, 195, 183], "gorilla's body": [29, 37, 38], "gorilla's hand": [31, 175, 163], "gorilla's foot": [139, 21, 152], "gorilla's tail": [245, 147, 189], "ambulance's body": [186, 240, 5], "ambulance's tier": [233, 201, 204], "ambulance's side mirror": [25, 222, 60], "motor scooter's body": [233, 131, 55], "motor scooter's head": [21, 100, 43], "motor scooter's seat": [246, 24, 123], "motor scooter's tier": [231, 111, 197], "yawl's body": [115, 131, 1], "yawl's sail": [244, 120, 207], "green lizard's head": [246, 221, 102], "green lizard's body": [194, 211, 185], "green lizard's foot": [246, 207, 182], "green lizard's tail": [140, 151, 141], "golden retriever's head": [145, 0, 153], "golden retriever's body": [119, 4, 123], "golden retriever's foot": [167, 211, 190], "golden retriever's tail": [19, 30, 200], "green mamba's head": [78, 149, 29], "green mamba's body": [17, 140, 32], "beer bottle's mouth": [125, 116, 118], "beer bottle's body": [131, 251, 24], "tiger's head": [178, 192, 131], "tiger's body": [242, 246, 85], "tiger's foot": [40, 4, 98], "tiger's tail": [182, 241, 148], "tree frog's head": [200, 224, 94], "tree frog's body": [32, 129, 160], "tree frog's foot": [182, 130, 222], "tree frog's tail": [214, 244, 200], "leopard's head": [228, 170, 218], "leopard's body": [75, 58, 144], "leopard's foot": [142, 253, 31], "leopard's tail": [156, 221, 163], "jeep's body": [212, 203, 143], "jeep's tier": [239, 247, 129], "jeep's side mirror": [113, 207, 158], "chimpanzee's head": [253, 238, 72], "chimpanzee's body": [179, 174, 176], "chimpanzee's hand": [71, 28, 21], "chimpanzee's foot": [159, 138, 165], "chimpanzee's tail": [219, 235, 214], "goose's head": [14, 236, 158], "goose's body": [78, 166, 194], "goose's wing": [94, 182, 252], "goose's foot": [184, 192, 128], "goose's tail": [159, 117, 225], "water bottle's mouth": [194, 241, 23], "water bottle's body": [178, 131, 73], "American alligator's head": [193, 220, 102], "American alligator's body": [110, 190, 89], "American alligator's foot": [34, 207, 44], "American alligator's tail": [154, 213, 81], "giant panda's head": [53, 60, 21], "giant panda's body": [250, 15, 175], "giant panda's foot": [250, 133, 3], "giant panda's tail": [1, 190, 172], "tench's head": [226, 68, 35], "tench's body": [248, 50, 67], "tench's fin": [78, 1, 140], "tench's tail": [7, 99, 115], "wine bottle's mouth": [176, 241, 15], "wine bottle's body": [121, 235, 242], "ice bear's head": [6, 109, 189], "ice bear's body": [100, 1, 209], "ice bear's foot": [207, 50, 98], "ice bear's tail": [147, 5, 24], "pirate's body": [141, 114, 8], "pirate's sail": [62, 38, 198], "box turtle's head": [9, 102, 185], "box turtle's body": [243, 11, 149], "box turtle's foot": [114, 137, 251], "box turtle's tail": [54, 217, 87], "warplane's head": [243, 182, 149], "warplane's body": [237, 125, 16], "warplane's engine": [33, 99, 176], "warplane's wing": [164, 196, 234], "warplane's tail": [41, 239, 24], "schooner's body": [132, 105, 170], "schooner's sail": [178, 117, 43], "Komodo dragon's head": [223, 129, 161], "Komodo dragon's body": [244, 41, 73], "Komodo dragon's foot": [162, 71, 237], "Komodo dragon's tail": [241, 21, 213], "Indian cobra's head": [56, 0, 119], "Indian cobra's body": [96, 254, 127], "gazelle's head": [221, 141, 189], "gazelle's body": [196, 161, 6], "gazelle's foot": [20, 4, 94], "gazelle's tail": [177, 152, 47], "mountain bike's body": [230, 119, 92], "mountain bike's head": [220, 1, 150], "mountain bike's seat": [167, 56, 144], "mountain bike's tier": [18, 170, 72], "school bus's body": [94, 14, 106], "school bus's tier": [10, 110, 171], "school bus's side mirror": [212, 230, 227], "killer whale's head": [151, 200, 221], "killer whale's body": [128, 58, 71], "killer whale's fin": [180, 46, 92], "killer whale's tail": [111, 221, 26], "moped's body": [63, 213, 52], "moped's head": [55, 227, 251], "moped's seat": [66, 2, 101], "moped's tier": [222, 127, 92], "airliner's head": [179, 235, 15], "airliner's body": [67, 59, 222], "airliner's engine": [61, 253, 168], "airliner's wing": [80, 30, 228], "airliner's tail": [125, 1, 118], "bald eagle's head": [244, 221, 238], "bald eagle's body": [0, 172, 69], "bald eagle's wing": [241, 31, 132], "bald eagle's foot": [192, 48, 114], "bald eagle's tail": [107, 126, 110]}
OBJ_CLASS_COLORS = {'school bus': [156, 151, 92], 'box turtle': [179, 94, 238], 'tench': [186, 66, 81], 'warplane': [216, 214, 45], 'schooner': [218, 110, 253], 'pirate': [192, 108, 156], 'garbage truck': [110, 148, 151], 'golden retriever': [3, 143, 33], 'killer whale': [156, 212, 81], 'bald eagle': [165, 83, 175], 'impala': [81, 106, 200], 'green lizard': [134, 18, 52], 'ambulance': [249, 10, 71], 'yawl': [182, 178, 9], 'Indian cobra': [200, 107, 254], 'albatross': [32, 63, 152], 'mountain bike': [213, 219, 32], 'goldfish': [175, 33, 10], 'moped': [110, 3, 208], 'ice bear': [135, 66, 200], 'beer bottle': [253, 171, 97], 'jeep': [170, 15, 103], 'Komodo dragon': [93, 91, 14], 'airliner': [235, 90, 6], 'tiger': [169, 7, 1], 'green mamba': [44, 98, 42], 'leopard': [246, 79, 170], 'gazelle': [227, 1, 239], 'minibus': [222, 11, 139], 'barracouta': [137, 29, 116], 'gorilla': [151, 25, 252], 'motor scooter': [65, 168, 44], 'giant panda': [115, 74, 27], 'water bottle': [1, 64, 230], 'wine bottle': [250, 175, 13], 'orangutan': [163, 246, 168], 'tree frog': [252, 6, 239], 'chimpanzee': [63, 61, 97], 'goose': [92, 86, 0], 'American alligator': [143, 87, 160]}
PART_CLASS_COLORS = {'body': [219, 80, 137], 'engine': [156, 117, 179], 'fin': [145, 192, 42], 'foot': [116, 138, 181], 'hand': [145, 44, 96], 'head': [218, 108, 251], 'mouth': [243, 219, 126], 'sail': [99, 216, 14], 'seat': [119, 189, 24], 'side mirror': [141, 40, 44], 'tail': [171, 43, 206], 'tier': [180, 28, 80], 'wing': [92, 234, 59]}


obj_map = {OBJ_CLASS_NAMES.index(c): i for i,c in enumerate(OBJ_BASE_CLASS_NAMES)}
obj_part_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}

def _get_obj_part_meta(cat_list, obj_list):
    id_to_name = {i: c for i, c in enumerate(cat_list)}
    thing_dataset_id_to_contiguous_id = {i: i for i in range(len(cat_list))}
    return {
        # "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        # "stuff_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "stuff_classes": cat_list,
        "obj_classes": obj_list,
        "obj_base_classes": OBJ_BASE_CLASS_NAMES,
        "part_classes": PART_CLASS_NAMES,
        "part_base_classes": PART_BASE_CLASS_NAMES,
        "part_categories": PART_IN_CATEGORIES,
        "stuff_colors": [CLASS_COLORS[c] for c in cat_list],
        "obj_colors": [OBJ_CLASS_COLORS[c] for c in obj_list],
        "part_colors": [PART_CLASS_COLORS[c] for c in PART_CLASS_NAMES],
    }


_PART_IN = {
    "partimagenet_obj_part_sem_seg_train": ("images/train", "train.json"),
    "partimagenet_obj_part_sem_seg_val": ("images/val", "val.json"),
}


def load_json(_root, image_root, json_file, extra_annotation_keys=None, per_image=False, val_all=False, data_list=None):
    dataset_dicts = []
    obj_dataset_dicts = load_coco_json(os.path.join(_root, json_file), os.path.join(_root, image_root), extra_annotation_keys=extra_annotation_keys)
    if data_list is not None:
        img_list = json.load(open(data_list,'r'))
        img_list = [item["file_name"] for item in img_list]
    for dataset_dict in obj_dataset_dicts:
        if data_list is not None:
            if dataset_dict["file_name"] not in img_list:
                continue
        if not per_image:
            for anno in dataset_dict['annotations']:
                record = {}
                record['file_name'] = dataset_dict['file_name']
                record['height'] = dataset_dict['height']
                record['width'] = dataset_dict['width']
                record['obj_annotations'] = [anno]
                record["obj_sem_seg_file_name"] = 'NA'
                record['category_id'] = anno['category_id']
                record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations').replace('JPEG','png')
                # if val_all and anno['category_id'] in obj_map:
                #     continue
                dataset_dicts.append(record)
        else:
            record = {}
            record['file_name'] = dataset_dict['file_name']
            record['height'] = dataset_dict['height']
            record['width'] = dataset_dict['width']
            record['obj_annotations'] = dataset_dict['annotations']
            record["obj_sem_seg_file_name"] = 'NA'
            # record['category_id'] = anno['category_id']
            record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations').replace('JPEG','png')
            dataset_dicts.append(record)
    return dataset_dicts


def load_train_val_json(_root, image_root, val_json_file):
    val_dataset_dicts = load_json(_root, image_root, val_json_file)
    # train_dataset_dicts = load_json(_root, image_root.replace('val','train'), val_json_file.replace('val','train'), val_all=True)
    
    return val_dataset_dicts # + train_dataset_dicts


def register_partimagenet(root):
    root = os.path.join(root, "PartImageNet")
    meta = _get_obj_part_meta(CLASS_NAMES, OBJ_CLASS_NAMES)
    base_meta = _get_obj_part_meta(BASE_CLASS_NAMES, OBJ_BASE_CLASS_NAMES)
    
    for name, (image_root, json_file) in _PART_IN.items():
        all_name = name
        name_obj_cond = all_name + "_obj_condition"

        MetadataCatalog.get(all_name).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65536,
            )

        if 'train' in name:
            DatasetCatalog.register(
                all_name, lambda: load_json(root, "images/train", "train.json", per_image=True)
            )
            DatasetCatalog.register(
                name_obj_cond, lambda: load_json(root, "images/train", "train.json")
            )
        else:
            # DatasetCatalog.register(
            #     all_name, lambda: load_json(root, "images/val", "val.json", per_image=True)
            # )
            DatasetCatalog.register(
                # name_obj_cond, lambda: load_train_val_json(root, "images/val", "val.json", per_image=True)
                name_obj_cond, lambda: load_json(root, "images/val", "val.json", per_image=True)
            ) ### test on both excusive train and val set
            
        if 'train' in name:
            MetadataCatalog.get(all_name).set(**base_meta)
            MetadataCatalog.get(all_name).set(obj_map=obj_map)
            MetadataCatalog.get(all_name).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(all_name).set(**meta)

        MetadataCatalog.get(name_obj_cond).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65536,
        )
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
        
        if name == 'train':
            MetadataCatalog.get(name_obj_cond).set(**base_meta)
            MetadataCatalog.get(name_obj_cond).set(obj_map=obj_map)
            MetadataCatalog.get(name_obj_cond).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(name_obj_cond).set(**meta)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_partimagenet(_root)