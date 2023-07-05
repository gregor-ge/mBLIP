import os
import json


langs = ["bn", "de", "en", "id", "ko", "pt", "ru", "zh"]

xgqa_root = "iglue/datasets/xGQA/annotations/few_shot"

train = json.load(
    open(os.path.join("iglue/datasets/gqa/annotations", "train.json")))
train_input = []
for row in train:
    image = row["img_id"]
    label = list(row["label"].keys())[0]
    context = row["sent"]

    entry = {
        "context": context,
        "label": label,
        "image_id": image,
        "text_label": label
    }
    train_input.append(entry)

json.dump(train_input, open(f"xgqa_input_train.json", "w", encoding="utf-8"))

for lang in langs:
    dev = json.load(open(os.path.join(xgqa_root, lang, "dev.json")))

    dev_input = []
    for row in dev.values():
        image = row["imageId"]
        label = row["answer"]
        context = row["question"]

        entry = {
            "context": context,
            "label": "",
            "image_id": image,
            "text_label": label
        }
        dev_input.append(entry)

    json.dump(dev_input, open(f"xgqa_input_val_{lang}.json", "w", encoding="utf-8"))

    test = json.load(open(os.path.join(xgqa_root, lang, "test.json")))

    test_input = []
    for row in test.values():
        image = row["imageId"]
        label = row["answer"]
        context = row["question"]

        entry = {
            "context": context,
            "label": "",
            "image_id": image,
            "text_label": label
        }
        test_input.append(entry)
    json.dump(test_input, open(f"xgqa_input_test_{lang}.json", "w", encoding="utf-8"))