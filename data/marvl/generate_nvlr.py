import os
import json


langs = ["id", "sw", "ta", "tr", "zh"]

marvl_root = "iglue/datasets/nlvr2/annotations"

for lang in langs:

    dev = [json.loads(row) for row in open(os.path.join(marvl_root, f"dev-{lang}_gmt.jsonl")).readlines()]

    dev_input = []
    for row in dev:
        sentence = row["sentence"]
        left_image = row["identifier"][:-2] + "-img0.png"
        right_image = row["identifier"][:-2] + "-img1.png"
        label = str(row["label"])
        
        entry = {
            "context": sentence,
            "left_image": left_image,
            "right_image": right_image,
            "label": "",
            "text_label": label
        }
        dev_input.append(entry)
    json.dump(dev_input, open(f"nlvr_dev_{lang}.json", "w", encoding="utf-8"))


for data_type in ["train", "test", "dev"]:
    dev = [json.loads(row) for row in open(os.path.join(marvl_root, f"{data_type}.jsonl")).readlines()]

    dev_input = []
    for row in dev:
        sentence = row["sentence"]
        if data_type == "train":
            left_image = str(row["directory"]) + "/" + row["identifier"][:-2] + "-img0.png"
            right_image = str(row["directory"]) + "/" + row["identifier"][:-2] + "-img1.png"
        else:
            left_image = row["identifier"][:-2] + "-img0.png"
            right_image = row["identifier"][:-2] + "-img1.png"
        label = row["label"]
        
        entry = {
            "context": sentence,
            "left_image": left_image,
            "right_image": right_image,
            "label": label if data_type == "train" else "",
            "text_label": label
        }
        dev_input.append(entry)
    print(data_type, len(dev_input))
    json.dump(dev_input, open(f"nlvr_{data_type}_en.json", "w", encoding="utf-8"))