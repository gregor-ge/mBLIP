import json
import os

root = "POPE/output/coco"

for file in ["coco_pope_adversarial.json", "coco_pope_popular.json", "coco_pope_random.json"]:
    data = [json.loads(l) for l in open(os.path.join(root, file)).readlines()]

    my_data = []
    for d in data:
        entry = {
            "context": d["text"],
            "label": "",
            "image_id": d["image"],
            "text_label": d["label"]
        }
        my_data.append(entry)
    with open(file, "w") as f:
        json.dump(my_data, f)