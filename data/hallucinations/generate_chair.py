import json
import os


# clone https://github.com/AoiDragon/POPE in this folder
annotations = [json.loads(l) for l in open("POPE/segmentation/coco_ground_truth_segmentation.json").readlines()]

annotations = {a["image"]: a["objects"] for a in annotations}

for split in ["test", "val"]:
    data = json.load(open(f"../mscoco/coco_{split}.json"))
    my_data = []
    for d in data:
        image = d["image_id"]
        entry = {
            "context": "English",
            "label": "",
            "image_id": image,
            "text_label": annotations[image]
        }
        my_data.append(entry)
    with open(f"chair_{split}.json", "w") as f:
        json.dump(my_data, f)