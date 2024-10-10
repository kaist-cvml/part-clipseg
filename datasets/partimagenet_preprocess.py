import os
import glob
import json
import shutil
import argparse


SUPERCLASS_OBJ_CLASS = {
    "Quadruped": [ "tiger", "giant panda", "leopard", "gazelle", "ice bear", "impala", "golden retriever"], 
    "Snake": [ "green mamba", "Indian cobra"], 
    "Reptile": ["green lizard", "Komodo dragon", "tree frog", "box turtle", "American alligator"], 
    "Boat": ["yawl", "pirate", "schooner"], 
    "Fish": ["barracouta", "goldfish", "killer whale", "tench"], 
    "Bird": ["albatross", "goose", "bald eagle"],
    "Car": ["garbage truck", "minibus", "ambulance", "jeep", "school bus"], 
    "Bicycle": [ "motor scooter", "mountain bike", "moped"],
    "Biped": [ "gorilla",  "orangutan", "chimpanzee"],
    "Bottle": ["beer bottle", "water bottle", "wine bottle"], 
    "Aeroplane": ["warplane", "airliner"]
}

SUPERCLASS_OBJ_NOVEL_CLASS = {
    "Quadruped": [ "ice bear", "impala", "golden retriever"], 
    "Snake": [ "Indian cobra"], 
    "Reptile": ["box turtle", "American alligator"], 
    "Boat": ["schooner"], 
    "Fish": ["tench"], 
    "Bird": ["bald eagle"],
    "Car": ["jeep", "school bus"], 
    "Bicycle": ["motor scooter"],
    "Biped": ["chimpanzee"],
    "Bottle": ["wine bottle"], 
    "Aeroplane": ["airliner"]
}

CLASS_MAP =  [
    {"id": 0, "name": "impala's head", "superclass": "Quadruped"}, {"id": 1, "name": "impala's body", "superclass": "Quadruped"}, {"id": 2, "name": "impala's foot", "superclass": "Quadruped"}, {"id": 3, "name": "impala's tail", "superclass": "Quadruped"}, {"id": 4, "name": "barracouta's head", "superclass": "Fish"}, {"id": 5, "name": "barracouta's body", "superclass": "Fish"}, {"id": 6, "name": "barracouta's fin", "superclass": "Fish"}, {"id": 7, "name": "barracouta's tail", "superclass": "Fish"}, {"id": 8, "name": "albatross's head", "superclass": "Bird"}, {"id": 9, "name": "albatross's body", "superclass": "Bird"}, 
    {"id": 10, "name": "albatross's wing", "superclass": "Bird"}, {"id": 11, "name": "albatross's foot", "superclass": "Bird"}, {"id": 12, "name": "albatross's tail", "superclass": "Bird"}, {"id": 13, "name": "garbage truck's body", "superclass": "Car"}, {"id": 14, "name": "garbage truck's tier", "superclass": "Car"}, {"id": 15, "name": "garbage truck's side mirror", "superclass": "Car"}, {"id": 16, "name": "minibus's body", "superclass": "Car"}, {"id": 17, "name": "minibus's tier", "superclass": "Car"}, {"id": 18, "name": "minibus's side mirror", "superclass": "Car"}, {"id": 19, "name": "orangutan's head", "superclass": "Biped"}, 
    {"id": 20, "name": "orangutan's body", "superclass": "Biped"}, {"id": 21, "name": "orangutan's hand", "superclass": "Biped"}, {"id": 22, "name": "orangutan's foot", "superclass": "Biped"}, {"id": 23, "name": "orangutan's tail", "superclass": "Biped"}, {"id": 24, "name": "goldfish's head", "superclass": "Fish"}, {"id": 25, "name": "goldfish's body", "superclass": "Fish"}, {"id": 26, "name": "goldfish's fin", "superclass": "Fish"}, {"id": 27, "name": "goldfish's tail", "superclass": "Fish"}, {"id": 28, "name": "gorilla's head", "superclass": "Biped"}, {"id": 29, "name": "gorilla's body", "superclass": "Biped"}, 
    {"id": 30, "name": "gorilla's hand", "superclass": "Biped"}, {"id": 31, "name": "gorilla's foot", "superclass": "Biped"}, {"id": 32, "name": "gorilla's tail", "superclass": "Biped"}, {"id": 33, "name": "ambulance's body", "superclass": "Car"}, {"id": 34, "name": "ambulance's tier", "superclass": "Car"}, {"id": 35, "name": "ambulance's side mirror", "superclass": "Car"}, {"id": 36, "name": "motor scooter's body", "superclass": "Bicycle"}, {"id": 37, "name": "motor scooter's head", "superclass": "Bicycle"}, {"id": 38, "name": "motor scooter's seat", "superclass": "Bicycle"}, {"id": 39, "name": "motor scooter's tier", "superclass": "Bicycle"}, 
    {"id": 40, "name": "yawl's body", "superclass": "Boat"}, {"id": 41, "name": "yawl's sail", "superclass": "Boat"}, {"id": 42, "name": "green lizard's head", "superclass": "Reptile"}, {"id": 43, "name": "green lizard's body", "superclass": "Reptile"}, {"id": 44, "name": "green lizard's foot", "superclass": "Reptile"}, {"id": 45, "name": "green lizard's tail", "superclass": "Reptile"}, {"id": 46, "name": "golden retriever's head", "superclass": "Quadruped"}, {"id": 47, "name": "golden retriever's body", "superclass": "Quadruped"}, {"id": 48, "name": "golden retriever's foot", "superclass": "Quadruped"}, {"id": 49, "name": "golden retriever's tail", "superclass": "Quadruped"}, 
    {"id": 50, "name": "green mamba's head", "superclass": "Snake"}, {"id": 51, "name": "green mamba's body", "superclass": "Snake"}, {"id": 52, "name": "beer bottle's mouth", "superclass": "Bottle"}, {"id": 53, "name": "beer bottle's body", "superclass": "Bottle"}, {"id": 54, "name": "tiger's head", "superclass": "Quadruped"}, {"id": 55, "name": "tiger's body", "superclass": "Quadruped"}, {"id": 56, "name": "tiger's foot", "superclass": "Quadruped"}, {"id": 57, "name": "tiger's tail", "superclass": "Quadruped"}, {"id": 58, "name": "tree frog's head", "superclass": "Reptile"}, {"id": 59, "name": "tree frog's body", "superclass": "Reptile"}, 
    {"id": 60, "name": "tree frog's foot", "superclass": "Reptile"}, {"id": 61, "name": "tree frog's tail", "superclass": "Reptile"}, {"id": 62, "name": "leopard's head", "superclass": "Quadruped"}, {"id": 63, "name": "leopard's body", "superclass": "Quadruped"}, {"id": 64, "name": "leopard's foot", "superclass": "Quadruped"}, {"id": 65, "name": "leopard's tail", "superclass": "Quadruped"}, {"id": 66, "name": "jeep's body", "superclass": "Car"}, {"id": 67, "name": "jeep's tier", "superclass": "Car"}, {"id": 68, "name": "jeep's side mirror", "superclass": "Car"}, {"id": 69, "name": "chimpanzee's head", "superclass": "Biped"}, 
    {"id": 70, "name": "chimpanzee's body", "superclass": "Biped"}, {"id": 71, "name": "chimpanzee's hand", "superclass": "Biped"}, {"id": 72, "name": "chimpanzee's foot", "superclass": "Biped"}, {"id": 73, "name": "chimpanzee's tail", "superclass": "Biped"}, {"id": 74, "name": "goose's head", "superclass": "Bird"}, {"id": 75, "name": "goose's body", "superclass": "Bird"}, {"id": 76, "name": "goose's wing", "superclass": "Bird"}, {"id": 77, "name": "goose's foot", "superclass": "Bird"}, {"id": 78, "name": "goose's tail", "superclass": "Bird"}, {"id": 79, "name": "water bottle's mouth", "superclass": "Bottle"}, 
    {"id": 80, "name": "water bottle's body", "superclass": "Bottle"}, {"id": 81, "name": "American alligator's head", "superclass": "Reptile"}, {"id": 82, "name": "American alligator's body", "superclass": "Reptile"}, {"id": 83, "name": "American alligator's foot", "superclass": "Reptile"}, {"id": 84, "name": "American alligator's tail", "superclass": "Reptile"}, {"id": 85, "name": "giant panda's head", "superclass": "Quadruped"}, {"id": 86, "name": "giant panda's body", "superclass": "Quadruped"}, {"id": 87, "name": "giant panda's foot", "superclass": "Quadruped"}, {"id": 88, "name": "giant panda's tail", "superclass": "Quadruped"}, {"id": 89, "name": "tench's head", "superclass": "Fish"}, 
    {"id": 90, "name": "tench's body", "superclass": "Fish"}, {"id": 91, "name": "tench's fin", "superclass": "Fish"}, {"id": 92, "name": "tench's tail", "superclass": "Fish"}, {"id": 93, "name": "wine bottle's mouth", "superclass": "Bottle"}, {"id": 94, "name": "wine bottle's body", "superclass": "Bottle"}, {"id": 95, "name": "ice bear's head", "superclass": "Quadruped"}, {"id": 96, "name": "ice bear's body", "superclass": "Quadruped"}, {"id": 97, "name": "ice bear's foot", "superclass": "Quadruped"}, {"id": 98, "name": "ice bear's tail", "superclass": "Quadruped"}, {"id": 99, "name": "pirate's body", "superclass": "Boat"}, 
    {"id": 100, "name": "pirate's sail", "superclass": "Boat"}, {"id": 101, "name": "box turtle's head", "superclass": "Reptile"}, {"id": 102, "name": "box turtle's body", "superclass": "Reptile"}, {"id": 103, "name": "box turtle's foot", "superclass": "Reptile"}, {"id": 104, "name": "box turtle's tail", "superclass": "Reptile"}, {"id": 105, "name": "warplane's head", "superclass": "Aeroplane"}, {"id": 106, "name": "warplane's body", "superclass": "Aeroplane"}, {"id": 107, "name": "warplane's engine", "superclass": "Aeroplane"}, {"id": 108, "name": "warplane's wing", "superclass": "Aeroplane"}, {"id": 109, "name": "warplane's tail", "superclass": "Aeroplane"}, 
    {"id": 110, "name": "schooner's body", "superclass": "Boat"}, {"id": 111, "name": "schooner's sail", "superclass": "Boat"}, {"id": 112, "name": "Komodo dragon's head", "superclass": "Reptile"}, {"id": 113, "name": "Komodo dragon's body", "superclass": "Reptile"}, {"id": 114, "name": "Komodo dragon's foot", "superclass": "Reptile"}, {"id": 115, "name": "Komodo dragon's tail", "superclass": "Reptile"}, {"id": 116, "name": "Indian cobra's head", "superclass": "Snake"}, {"id": 117, "name": "Indian cobra's body", "superclass": "Snake"}, {"id": 118, "name": "gazelle's head", "superclass": "Quadruped"}, {"id": 119, "name": "gazelle's body", "superclass": "Quadruped"},
    {"id": 120, "name": "gazelle's foot", "superclass": "Quadruped"}, {"id": 121, "name": "gazelle's tail", "superclass": "Quadruped"}, {"id": 122, "name": "mountain bike's body", "superclass": "Bicycle"}, {"id": 123, "name": "mountain bike's head", "superclass": "Bicycle"}, {"id": 124, "name": "mountain bike's seat", "superclass": "Bicycle"}, {"id": 125, "name": "mountain bike's tier", "superclass": "Bicycle"}, {"id": 126, "name": "school bus's body", "superclass": "Car"}, {"id": 127, "name": "school bus's tier", "superclass": "Car"}, {"id": 128, "name": "school bus's side mirror", "superclass": "Car"}, {"id": 133, "name": "moped's body", "superclass": "Bicycle"}, 
    {"id": 134, "name": "moped's head", "superclass": "Bicycle"}, {"id": 135, "name": "moped's seat", "superclass": "Bicycle"}, {"id": 136, "name": "moped's tier", "superclass": "Bicycle"}, {"id": 137, "name": "airliner's head", "superclass": "Aeroplane"}, {"id": 138, "name": "airliner's body", "superclass": "Aeroplane"}, {"id": 139, "name": "airliner's engine", "superclass": "Aeroplane"}, {"id": 140, "name": "airliner's wing", "superclass": "Aeroplane"}, {"id": 141, "name": "airliner's tail", "superclass": "Aeroplane"}, {"id": 142, "name": "bald eagle's head", "superclass": "Bird"}, {"id": 143, "name": "bald eagle's body", "superclass": "Bird"}, 
    {"id": 144, "name": "bald eagle's wing", "superclass": "Bird"}, {"id": 145, "name": "bald eagle's foot", "superclass": "Bird"}, {"id": 146, "name": "bald eagle's tail", "superclass": "Bird"}
]


OBJ_CLASS_NAMES = []
OBJ_BASE_CLASS_NAMES = []
OBJ_NOVEL_CLASS_NAMES = []

def load_imagenet_labels(imagenet_labels_filename):
    imagenet_id2name = {}
    with open(imagenet_labels_filename, "r") as f:        
        for line in f:
            line = line.strip()
            line_split = line.split(' ',maxsplit=1)
            imagenet_id2name[line_split[0]] = line_split[1].split(',', maxsplit=1)[0]

    return imagenet_id2name


def preprocess_partimagenet(data_dir, imagenet_labels_filename):
    # Load imagenet labels
    imagenet_id2name = load_imagenet_labels(imagenet_labels_filename)
    
    # Prepare class names
    for obj_classes in SUPERCLASS_OBJ_CLASS.values():
        OBJ_CLASS_NAMES.extend(obj_classes)

    for obj_classes in SUPERCLASS_OBJ_NOVEL_CLASS.values():
        OBJ_NOVEL_CLASS_NAMES.extend(obj_classes)

    # Generate class IDs
    OBJ_CLASS_IDS = [k for k, v in imagenet_id2name.items() if v in OBJ_CLASS_NAMES]
    OBJ_NOVEL_CLASS_IDS = [k for k, v in imagenet_id2name.items() if v in OBJ_NOVEL_CLASS_NAMES]
    OBJ_BASE_CLASS_IDS = list(set(OBJ_CLASS_IDS) - set(OBJ_NOVEL_CLASS_IDS))

    # Copy the original directory
    # temp_data_dir = data_dir + 'Cache'
    # if os.path.exists(temp_data_dir):
    #     shutil.rmtree(temp_data_dir)
    # shutil.copytree(data_dir, temp_data_dir)

    def clean_directory(data_dir, image_dir, anno_dir, valid_ids):
        for data_file in os.listdir(image_dir):
            if data_file.split('_')[0] not in valid_ids:
                os.remove(os.path.join(image_dir, data_file))

        for data_file in os.listdir(anno_dir):
            if data_file in ['train.json', 'val.json']:
                # move to data_dir
                shutil.move(os.path.join(anno_dir, data_file), os.path.join(data_dir, data_file))
            elif data_file.split('_')[0] not in valid_ids:
                os.remove(os.path.join(anno_dir, data_file))

    # Clean directories in the copied data directory
    clean_directory(data_dir, os.path.join(data_dir, 'images', 'train'), os.path.join(data_dir, 'annotations', 'train'), OBJ_BASE_CLASS_IDS)
    clean_directory(data_dir, os.path.join(data_dir, 'images', 'val'), os.path.join(data_dir, 'annotations', 'val'), OBJ_CLASS_IDS)

    return OBJ_CLASS_IDS, OBJ_BASE_CLASS_IDS


def preprocess_map_json(data_dir, OBJ_CLASS_IDS, OBJ_BASE_CLASS_IDS):
    def update_json(map_json_path, valid_ids, image_dir):
        map_json = json.load(open(map_json_path))
        valid_images = []
        valid_annotations = []

        for image in map_json['images']:
            file_name = image['file_name'].split('_', maxsplit=1)[0]
            if file_name in valid_ids and os.path.exists(os.path.join(image_dir, image['file_name'])):
                valid_images.append(image)

        valid_image_ids = {img['id'] for img in valid_images}

        for anno in map_json['annotations']:
            if anno['image_id'] in valid_image_ids:
                valid_annotations.append(anno)

        map_json['images'] = valid_images
        map_json['annotations'] = valid_annotations
        map_json['categories'] = CLASS_MAP

        with open(map_json_path, 'w') as f:
            json.dump(map_json, f)

    update_json(os.path.join(data_dir, 'train.json'), OBJ_BASE_CLASS_IDS, os.path.join(data_dir, 'images', 'train'))
    update_json(os.path.join(data_dir, 'val.json'), OBJ_CLASS_IDS, os.path.join(data_dir, 'images', 'val'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='PartImageNet')
    parser.add_argument('--imagenet_labels_filename', type=str, default='LOC_synset_mapping.txt')

    args = parser.parse_args()

    OBJ_CLASS_IDS, OBJ_BASE_CLASS_IDS = preprocess_partimagenet(args.data_dir, args.imagenet_labels_filename)
    preprocess_map_json(args.data_dir, OBJ_CLASS_IDS, OBJ_BASE_CLASS_IDS)

    return

if __name__ == '__main__':
    main()