import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from .coco import load_coco_json
from .utils import load_binary_mask
import json

OBJ_CLASS_NAMES = ['airplane', 'armchair', 'bed', 'bench', 'bookcase', 'bus', 'cabinet', 'car', 'chair', 'chandelier', 'chest of drawers', 'clock', 'coffee table', 'computer', 'cooking stove', 'desk', 'dishwasher', 'door', 
                   'fan', 'glass', 'kitchen island', 'lamp', 'light', 'microwave', 'minibike', 'ottoman', 'oven', 'person', 'pool table', 'refrigerator', 'sconce', 'shelf', 'sink', 'sofa', 'stool', 
                   'swivel chair', 'table', 'television receiver', 'toilet', 'traffic light', 'truck', 'van', 'wardrobe', 'washer']
CLASS_NAMES = ["person's arm", "person's back", "person's foot", "person's gaze", "person's hand", "person's head", "person's leg", "person's neck", "person's torso", "door's door frame", "door's handle", "door's knob", 
               "door's panel", "clock's face", "clock's frame", "toilet's bowl", "toilet's cistern", "toilet's lid", "cabinet's door", "cabinet's drawer", "cabinet's front", "cabinet's shelf", 
               "cabinet's side", "cabinet's skirt", "cabinet's top", "sink's bowl", "sink's faucet", "sink's pedestal", "sink's tap", "sink's top", "lamp's arm", "lamp's base", "lamp's canopy", "lamp's column", 
               "lamp's cord", "lamp's highlight", "lamp's light source", "lamp's shade", "lamp's tube", "sconce's arm", "sconce's backplate", "sconce's highlight", "sconce's light source", "sconce's shade", "chair's apron",
               "chair's arm", "chair's back", "chair's base", "chair's leg", "chair's seat", "chair's seat cushion", "chair's skirt", "chair's stretcher", "chest of drawers's apron", "chest of drawers's door", "chest of drawers's drawer", 
               "chest of drawers's front", "chest of drawers's leg", "chandelier's arm", "chandelier's bulb", "chandelier's canopy", "chandelier's chain", "chandelier's cord", "chandelier's highlight", "chandelier's light source", "chandelier's shade",
               "bed's footboard", "bed's headboard", "bed's leg", "bed's side rail", "table's apron", "table's drawer", "table's leg", "table's shelf", "table's top", "table's wheel", "armchair's apron", "armchair's arm", "armchair's back", 
               "armchair's back pillow", "armchair's leg", "armchair's seat", "armchair's seat base", "armchair's seat cushion", "ottoman's back", "ottoman's leg", "ottoman's seat", "shelf's door", "shelf's drawer", "shelf's front", "shelf's shelf", 
               "swivel chair's back", "swivel chair's base", "swivel chair's seat", "swivel chair's wheel", "fan's blade", "fan's canopy", "fan's tube", "coffee table's leg", "coffee table's top", "stool's leg", "stool's seat", "sofa's arm", "sofa's back", 
               "sofa's back pillow", "sofa's leg", "sofa's seat base", "sofa's seat cushion", "sofa's skirt", "computer's computer case", "computer's keyboard", "computer's monitor", "computer's mouse", "desk's apron", "desk's door", "desk's drawer", "desk's leg",
               "desk's shelf", "desk's top", "wardrobe's door", "wardrobe's drawer", "wardrobe's front", "wardrobe's leg", "wardrobe's mirror", "wardrobe's top", "car's bumper", "car's door", "car's headlight", "car's hood", "car's license plate", "car's logo", 
               "car's mirror", "car's wheel", "car's window", "car's wiper", "bus's bumper", "bus's door", "bus's headlight", "bus's license plate", "bus's logo", "bus's mirror", "bus's wheel", "bus's window", "bus's wiper", "oven's button panel", "oven's door", 
               "oven's drawer", "oven's top", "cooking stove's burner", "cooking stove's button panel", "cooking stove's door", "cooking stove's drawer", "cooking stove's oven", "cooking stove's stove", "microwave's button panel", "microwave's door", "microwave's front",
               "microwave's side", "microwave's top", "microwave's window", "refrigerator's button panel", "refrigerator's door", "refrigerator's drawer", "refrigerator's side", "kitchen island's door", "kitchen island's drawer", "kitchen island's front", "kitchen island's side", 
               "kitchen island's top", "dishwasher's button panel", "dishwasher's handle", "dishwasher's skirt", "bookcase's door", "bookcase's drawer", "bookcase's front", "bookcase's side", "television receiver's base", "television receiver's buttons", "television receiver's frame",
               "television receiver's keys", "television receiver's screen", "television receiver's speaker", "glass's base", "glass's bowl", "glass's opening", "glass's stem", "pool table's bed", "pool table's leg", "pool table's pocket", "van's bumper", "van's door", "van's headlight", 
               "van's license plate", "van's logo", "van's mirror", "van's taillight", "van's wheel", "van's window", "van's wiper", "airplane's door", "airplane's fuselage", "airplane's landing gear", "airplane's propeller", "airplane's stabilizer", "airplane's turbine engine", 
               "airplane's wing", "truck's bumper", "truck's door", "truck's headlight", "truck's license plate", "truck's logo", "truck's mirror", "truck's wheel", "truck's window", "minibike's license plate", "minibike's mirror", "minibike's seat", "minibike's wheel", "washer's button panel", 
               "washer's door", "washer's front", "washer's side", "bench's arm", "bench's back", "bench's leg", "bench's seat", "traffic light's housing", "traffic light's pole", "light's aperture", "light's canopy", "light's diffusor", "light's highlight", "light's light source", "light's shade"]
OBJ_NOVEL_CLASS_NAMES = ['bench', 'bus', 'fan', 'desk', 'stool', 'truck', 'van', 'swivel chair', 'oven', 'ottoman', 'kitchen island']
OBJ_BASE_CLASS_NAMES = [c for c in OBJ_CLASS_NAMES if c not in OBJ_NOVEL_CLASS_NAMES]
BASE_CLASS_NAMES = [c for i, c in enumerate(CLASS_NAMES) if c.split('\'s')[0] not in OBJ_NOVEL_CLASS_NAMES]

PART_CLASS_NAMES = sorted(list(set([c.split('\'s')[1].strip() for c in CLASS_NAMES])))
PART_BASE_CLASS_NAMES = sorted(list(set([c.split('\'s')[1].strip() for c in CLASS_NAMES if c.split('\'s')[0] not in OBJ_NOVEL_CLASS_NAMES])))

CLASS_COLORS = {
    "person's arm": [50, 231, 251], "person's back": [46, 133, 60], "person's foot": [10, 52, 35], "person's gaze": [44, 164, 247], "person's hand": [59, 123, 33], "person's head": [133, 83, 34], "person's leg": [114, 146, 47], "person's neck": [159, 176, 206], "person's torso": [211, 139, 180], 
    "door's door frame": [238, 59, 253], "door's handle": [190, 150, 121], "door's knob": [58, 20, 134], "door's panel": [157, 203, 49], "clock's face": [140, 108, 62], "clock's frame": [235, 203, 46], "toilet's bowl": [81, 234, 119], "toilet's cistern": [184, 136, 70], "toilet's lid": [220, 196, 59], 
    "cabinet's door": [66, 184, 92], "cabinet's drawer": [74, 66, 79], "cabinet's front": [86, 144, 90], "cabinet's shelf": [190, 196, 29], "cabinet's side": [22, 110, 159], "cabinet's skirt": [172, 143, 20], "cabinet's top": [98, 66, 112], 
    "sink's bowl": [220, 63, 228], "sink's faucet": [61, 64, 160], "sink's pedestal": [32, 135, 232], "sink's tap": [199, 10, 243], "sink's top": [221, 226, 190], 
    "lamp's arm": [162, 75, 4], "lamp's base": [252, 7, 187], "lamp's canopy": [183, 130, 98], "lamp's column": [183, 244, 182], "lamp's cord": [123, 213, 68], "lamp's highlight": [88, 129, 31], "lamp's light source": [139, 193, 248], "lamp's shade": [191, 97, 83], "lamp's tube": [29, 118, 205], 
    "sconce's arm": [3, 2, 76], "sconce's backplate": [168, 130, 82], "sconce's highlight": [125, 213, 47], "sconce's light source": [4, 15, 228], "sconce's shade": [53, 164, 80], 
    "chair's apron": [200, 224, 14], "chair's arm": [66, 19, 244], "chair's back": [168, 32, 25], "chair's base": [73, 113, 153], "chair's leg": [57, 148, 22], "chair's seat": [128, 137, 130], "chair's seat cushion": [79, 234, 164], "chair's skirt": [160, 209, 225], "chair's stretcher": [26, 36, 206], 
    "chest of drawers's apron": [111, 66, 100], "chest of drawers's door": [151, 99, 136], "chest of drawers's drawer": [42, 56, 135], "chest of drawers's front": [185, 4, 29], "chest of drawers's leg": [97, 119, 11], 
    "chandelier's arm": [214, 254, 62], "chandelier's bulb": [100, 78, 216], "chandelier's canopy": [94, 138, 204], "chandelier's chain": [165, 52, 181], "chandelier's cord": [58, 156, 78], "chandelier's highlight": [51, 26, 115], "chandelier's light source": [179, 28, 20], "chandelier's shade": [153, 190, 46], 
    "bed's footboard": [53, 48, 45], "bed's headboard": [186, 146, 129], "bed's leg": [14, 69, 107], "bed's side rail": [32, 232, 65], "table's apron": [210, 109, 151], "table's drawer": [14, 119, 75], "table's leg": [191, 182, 28], "table's shelf": [175, 160, 52], "table's top": [18, 0, 47], "table's wheel": [1, 82, 163], 
    "armchair's apron": [34, 132, 97], "armchair's arm": [131, 76, 166], "armchair's back": [87, 38, 183], "armchair's back pillow": [221, 166, 202], "armchair's leg": [178, 156, 79], "armchair's seat": [2, 190, 54], "armchair's seat base": [180, 66, 72], "armchair's seat cushion": [105, 239, 183], 
    "ottoman's back": [88, 73, 160], "ottoman's leg": [172, 191, 108], "ottoman's seat": [206, 14, 59], "shelf's door": [184, 226, 49], "shelf's drawer": [48, 196, 167], "shelf's front": [87, 118, 108], "shelf's shelf": [80, 170, 63], "swivel chair's back": [230, 140, 28], "swivel chair's base": [45, 110, 123], "swivel chair's seat": [3, 94, 66], "swivel chair's wheel": [139, 38, 41], 
    "fan's blade": [226, 235, 201], "fan's canopy": [93, 246, 101], "fan's tube": [234, 77, 136], "coffee table's leg": [175, 176, 47], "coffee table's top": [180, 242, 44], "stool's leg": [135, 30, 173], "stool's seat": [56, 246, 172], "sofa's arm": [241, 211, 32], "sofa's back": [112, 129, 142], "sofa's back pillow": [104, 55, 225], "sofa's leg": [102, 21, 2], "sofa's seat base": [51, 87, 240], "sofa's seat cushion": [184, 40, 227], "sofa's skirt": [161, 6, 249], 
    "computer's computer case": [252, 37, 182], "computer's keyboard": [22, 51, 161], "computer's monitor": [17, 211, 124], "computer's mouse": [230, 80, 3], "desk's apron": [70, 57, 19], "desk's door": [0, 250, 135], "desk's drawer": [41, 138, 241], "desk's leg": [3, 66, 125], "desk's shelf": [250, 42, 158], "desk's top": [238, 154, 184], "wardrobe's door": [133, 45, 72], "wardrobe's drawer": [45, 232, 5], "wardrobe's front": [166, 130, 79], "wardrobe's leg": [54, 113, 84], "wardrobe's mirror": [115, 200, 44], "wardrobe's top": [22, 125, 125], "car's bumper": [88, 131, 113], "car's door": [50, 22, 121], "car's headlight": [84, 140, 240], "car's hood": [53, 46, 105], "car's license plate": [26, 205, 197], "car's logo": [175, 162, 182], "car's mirror": [108, 79, 109], "car's wheel": [8, 14, 170], "car's window": [57, 105, 105], "car's wiper": [85, 65, 2], "bus's bumper": [10, 93, 83], "bus's door": [12, 153, 32], "bus's headlight": [5, 250, 78], "bus's license plate": [238, 185, 60], "bus's logo": [153, 121, 238], "bus's mirror": [145, 143, 73], "bus's wheel": [5, 156, 140], "bus's window": [218, 88, 139], "bus's wiper": [5, 207, 97], "oven's button panel": [87, 1, 48], "oven's door": [168, 0, 25], "oven's drawer": [113, 115, 67], "oven's top": [23, 21, 97], "cooking stove's burner": [234, 23, 12], "cooking stove's button panel": [69, 133, 183], "cooking stove's door": [214, 19, 237], "cooking stove's drawer": [22, 21, 197], "cooking stove's oven": [247, 176, 31], "cooking stove's stove": [209, 57, 179], "microwave's button panel": [109, 110, 175], "microwave's door": [82, 70, 128], "microwave's front": [112, 225, 21], "microwave's side": [53, 217, 223], "microwave's top": [55, 231, 101], "microwave's window": [148, 169, 246], "refrigerator's button panel": [176, 167, 110], "refrigerator's door": [48, 223, 122], "refrigerator's drawer": [229, 175, 80], "refrigerator's side": [57, 216, 3], "kitchen island's door": [238, 150, 155], "kitchen island's drawer": [121, 54, 42], "kitchen island's front": [230, 70, 221], "kitchen island's side": [231, 39, 239], "kitchen island's top": [182, 149, 23], "dishwasher's button panel": [241, 189, 254], "dishwasher's handle": [82, 228, 49], "dishwasher's skirt": [204, 208, 25], "bookcase's door": [100, 149, 172], "bookcase's drawer": [30, 116, 92], "bookcase's front": [5, 248, 53], "bookcase's side": [42, 42, 242], "television receiver's base": [145, 219, 44], "television receiver's buttons": [134, 105, 75], "television receiver's frame": [36, 127, 90], "television receiver's keys": [154, 109, 142], "television receiver's screen": [163, 250, 163], "television receiver's speaker": [152, 113, 218], "glass's base": [143, 46, 34], "glass's bowl": [52, 49, 62], "glass's opening": [196, 201, 208], "glass's stem": [16, 110, 132], "pool table's bed": [102, 167, 84], "pool table's leg": [70, 70, 82], "pool table's pocket": [157, 157, 67], "van's bumper": [34, 110, 57], "van's door": [13, 179, 90], "van's headlight": [176, 138, 110], "van's license plate": [29, 29, 241], "van's logo": [65, 45, 110], "van's mirror": [123, 147, 119], "van's taillight": [194, 176, 248], "van's wheel": [217, 230, 11], "van's window": [152, 141, 30], "van's wiper": [126, 100, 236], "airplane's door": [41, 92, 24], "airplane's fuselage": [136, 79, 151], "airplane's landing gear": [83, 156, 189], "airplane's propeller": [124, 227, 145], "airplane's stabilizer": [18, 251, 192], "airplane's turbine engine": [131, 165, 64], "airplane's wing": [84, 56, 60], "truck's bumper": [181, 53, 196], "truck's door": [110, 92, 199], "truck's headlight": [232, 107, 254], "truck's license plate": [92, 65, 196], "truck's logo": [239, 186, 60], "truck's mirror": [252, 214, 77], "truck's wheel": [25, 30, 131], "truck's window": [91, 195, 107], "minibike's license plate": [160, 238, 96], "minibike's mirror": [137, 20, 70], "minibike's seat": [26, 251, 7], "minibike's wheel": [68, 168, 151], "washer's button panel": [162, 240, 244], "washer's door": [218, 47, 88], "washer's front": [66, 104, 220], "washer's side": [4, 4, 161], "bench's arm": [82, 243, 41], "bench's back": [209, 225, 163], "bench's leg": [201, 252, 167], "bench's seat": [8, 94, 231], "traffic light's housing": [82, 39, 139], "traffic light's pole": [4, 84, 97], "light's aperture": [75, 112, 85], "light's canopy": [191, 147, 46], "light's diffusor": [8, 4, 35], "light's highlight": [119, 4, 236], "light's light source": [190, 139, 18], "light's shade": [220, 82, 47]}
OBJ_CLASS_COLORS = {'airplane': [125, 222, 195], 'armchair': [180, 63, 208], 'bed': [28, 83, 98], 'bench': [99, 54, 3], 'bookcase': [89, 130, 150], 'bus': [124, 237, 112], 'cabinet': [68, 72, 182], 'car': [53, 2, 102], 'chair': [26, 246, 4], 'chandelier': [147, 221, 89], 'chest of drawers': [198, 240, 118], 'clock': [77, 251, 38], 'coffee table': [123, 54, 190], 'computer': [62, 248, 158], 'cooking stove': [186, 217, 44], 'desk': [214, 163, 43], 'dishwasher': [80, 49, 171], 'door': [4, 128, 237], 'fan': [185, 138, 247], 'glass': [104, 11, 131], 'kitchen island': [112, 240, 98], 'lamp': [151, 246, 102], 'light': [72, 191, 154], 'microwave': [114, 101, 184], 'minibike': [238, 140, 31], 'ottoman': [8, 124, 92], 'oven': [88, 64, 212], 'person': [97, 246, 215], 'pool table': [86, 169, 26], 'refrigerator': [127, 52, 25], 'sconce': [163, 24, 245], 'shelf': [205, 85, 82], 'sink': [213, 25, 230], 'sofa': [131, 170, 136], 'stool': [137, 179, 151], 'swivel chair': [225, 201, 240], 'table': [247, 133, 166], 'television receiver': [226, 151, 96], 'toilet': [56, 118, 134], 'traffic light': [208, 29, 236], 'truck': [224, 26, 52], 'van': [123, 108, 209], 'wardrobe': [8, 189, 125], 'washer': [108, 45, 45]}
PART_CLASS_COLORS = {'aperture': [169, 141, 31], 'apron': [146, 15, 36], 'arm': [80, 104, 201], 'back': [174, 113, 149], 'back pillow': [174, 44, 125], 'backplate': [188, 117, 14], 'base': [136, 250, 23], 'bed': [76, 40, 175], 'blade': [206, 12, 164], 'bowl': [127, 126, 108], 'bulb': [78, 128, 176], 'bumper': [208, 165, 67], 'burner': [199, 65, 180], 'button panel': [34, 226, 172], 'buttons': [74, 98, 153], 'canopy': [2, 212, 158], 'chain': [83, 111, 98], 'cistern': [205, 93, 247], 'column': [8, 244, 93], 'computer case': [175, 149, 249], 'cord': [249, 114, 254], 'diffusor': [202, 230, 165], 'door': [198, 185, 148], 'door frame': [140, 152, 62], 'drawer': [143, 59, 55], 'face': [75, 250, 142], 'faucet': [177, 252, 149], 'foot': [99, 191, 232], 'footboard': [131, 205, 221], 'frame': [162, 29, 226], 'front': [1, 250, 197], 'fuselage': [109, 36, 121], 'gaze': [138, 228, 180], 'hand': [94, 177, 113], 'handle': [173, 184, 37], 'head': [152, 8, 232], 'headboard': [157, 99, 208], 'headlight': [44, 20, 166], 'highlight': [51, 75, 30], 'hood': [249, 131, 37], 'housing': [119, 168, 239], 'keyboard': [63, 80, 132], 'keys': [241, 247, 92], 'knob': [5, 52, 192], 'landing gear': [139, 61, 126], 'leg': [224, 56, 12], 'license plate': [201, 19, 162], 'lid': [5, 156, 65], 'light source': [56, 243, 182], 'logo': [103, 26, 165], 'mirror': [200, 93, 151], 'monitor': [102, 93, 132], 'mouse': [212, 206, 82], 'neck': [236, 58, 177], 'opening': [150, 195, 36], 'oven': [23, 4, 251], 'panel': [58, 205, 83], 'pedestal': [219, 49, 177], 'pocket': [55, 111, 126], 'pole': [71, 62, 107], 'propeller': [136, 230, 252], 'screen': [141, 247, 229], 'seat': [41, 222, 253], 'seat base': [33, 152, 189], 'seat cushion': [20, 113, 19], 'shade': [14, 28, 22], 'shelf': [59, 36, 125], 'side': [37, 216, 9], 'side rail': [152, 200, 233], 'skirt': [138, 100, 236], 'speaker': [26, 123, 80], 'stabilizer': [244, 12, 114], 'stem': [254, 144, 110], 'stove': [94, 235, 137], 'stretcher': [245, 37, 123], 'taillight': [229, 251, 61], 'tap': [224, 177, 151], 'top': [133, 141, 216], 'torso': [98, 182, 202], 'tube': [67, 177, 178], 'turbine engine': [82, 190, 201], 'wheel': [216, 58, 57], 'window': [169, 75, 221], 'wing': [200, 225, 39], 'wiper': [167, 79, 195]}

obj_map = {OBJ_CLASS_NAMES.index(c): i for i,c in enumerate(OBJ_BASE_CLASS_NAMES)}
obj_part_map = {CLASS_NAMES.index(c): i for i,c in enumerate(BASE_CLASS_NAMES)}


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "ade_obj_part_sem_seg_train": (
        "images/training",
        "ade20k_instance_train.json",
    ),
    "ade_obj_part_sem_seg_val": (
        "images/validation",
        "ade20k_instance_val.json",
    )
}

def _get_obj_part_meta(cat_list, obj_list):
    ret = {
        "stuff_classes": cat_list,
        "obj_classes": obj_list,
        "obj_base_classes": OBJ_BASE_CLASS_NAMES,
        "part_classes": PART_CLASS_NAMES,
        "part_base_classes": PART_BASE_CLASS_NAMES,
        "stuff_colors": [CLASS_COLORS[c] for c in cat_list],
        "obj_colors": [OBJ_CLASS_COLORS[c] for c in obj_list],
        "part_colors": [PART_CLASS_COLORS[c] for c in PART_CLASS_NAMES],
    }
    return ret

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
                if 'part_category_id' in anno:
                    part_ids = anno['part_category_id']
                    for part_id in part_ids:
                        record = {}
                        record['file_name'] = dataset_dict['file_name']
                        record['height'] = dataset_dict['height']
                        record['width'] = dataset_dict['width']
                        record['obj_annotations'] = anno
                        record["obj_sem_seg_file_name"] = 'NA'
                        record['category_id'] = part_id
                        record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
                        dataset_dicts.append(record)
                else:
                    record = {}
                    record['file_name'] = dataset_dict['file_name']
                    record['height'] = dataset_dict['height']
                    record['width'] = dataset_dict['width']
                    record['obj_annotations'] = [anno]
                    record["obj_sem_seg_file_name"] = 'NA'
                    record['category_id'] = anno['category_id']
                    record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
                    if val_all and anno['category_id'] in obj_map:
                        continue
                    dataset_dicts.append(record)
        else:
            record = {}
            record['file_name'] = dataset_dict['file_name']
            record['height'] = dataset_dict['height']
            record['width'] = dataset_dict['width']
            record['obj_annotations'] = dataset_dict['annotations']
            record["obj_sem_seg_file_name"] = 'NA'
            # record['category_id'] = anno['category_id']
            record["sem_seg_file_name"] = dataset_dict['file_name'].replace('images','annotations_detectron2_part').replace('jpg','png')
            dataset_dicts.append(record)
    return dataset_dicts

def load_train_val_json(_root, image_root, val_json_file, few_shot=False, data_list=None):
    val_dataset_dicts = load_json(_root, image_root, val_json_file)
    train_dataset_dicts = load_json(_root, image_root.replace('validation','training'), val_json_file.replace('val','train'), val_all=True)
    
    if few_shot:
        assert data_list is not None
        train_fw = json.load(open(data_list,'r'))
        train_fw_file_names = [t['file_name'] for t in train_fw]
        tmp = []
        for record in train_dataset_dicts:
            if record['file_name'] not in train_fw_file_names:
                tmp.append(record)
        train_dataset_dicts = tmp

    return val_dataset_dicts + train_dataset_dicts
        
def register_ade20k_part_234(root):
    root = os.path.join(root, "ADE20KPart234")
    meta = _get_obj_part_meta(CLASS_NAMES, OBJ_CLASS_NAMES)
    base_meta = _get_obj_part_meta(BASE_CLASS_NAMES, OBJ_BASE_CLASS_NAMES)
    for name, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        all_name = name
        name_obj_cond = all_name + "_obj_condition"
        name_few_shot = all_name + "_few_shot"
        if 'train' in name:
            DatasetCatalog.register(all_name, lambda: load_json(root, "images/training", "ade20k_instance_train.json",per_image=True)) ### image level annotations
            DatasetCatalog.register(name_obj_cond, lambda: load_json(root, "images/training", "ade20k_instance_train.json")) ### object instance level annotations
            DatasetCatalog.register(name_few_shot, lambda: load_json(root, "images/training", "ade20k_instance_train.json", data_list=f'{root}/train_16shot.json')) ### object instance level annotations in few shot
            MetadataCatalog.get(all_name).set(
                image_root=image_root,
                sem_seg_root=image_root,
                evaluator_type="sem_seg",
                ignore_label=65535,
            )
        else:
            DatasetCatalog.register(name_obj_cond, lambda: load_train_val_json(root, "images/validation", "ade20k_instance_val.json")) ### test on both excusive train and val set
            DatasetCatalog.register(name_few_shot, lambda: load_train_val_json(root, "images/validation", "ade20k_instance_val.json", few_shot=True, data_list=f'{root}/train_16shot.json')) ### few shot test set
        
        MetadataCatalog.get(name_obj_cond).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65535,
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
        
        if 'train' in name:
            MetadataCatalog.get(name_obj_cond).set(**base_meta)
            MetadataCatalog.get(name_obj_cond).set(obj_map=obj_map)
            MetadataCatalog.get(name_obj_cond).set(obj_part_map=obj_part_map)
            MetadataCatalog.get(all_name).set(**base_meta)
            MetadataCatalog.get(all_name).set(obj_map=obj_map)
            MetadataCatalog.get(all_name).set(obj_part_map=obj_part_map)
        else:
            MetadataCatalog.get(name_obj_cond).set(**meta)
            
        MetadataCatalog.get(name_few_shot).set(**meta)
        MetadataCatalog.get(name_few_shot).set(
            image_root=image_root,
            sem_seg_root=image_root,
            evaluator_type="sem_seg",
            ignore_label=65535,
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
        MetadataCatalog.get(name_few_shot).set(
            evaluation_set={
                "base": [
                    meta["stuff_classes"].index(n) for n in meta["stuff_classes"]
                ],
            },
            trainable_flag=[
                1 if n in meta["stuff_classes"] else 0
                for n in meta["stuff_classes"]
            ],
        )
        
    
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_part_234(_root)
