import os
import json


langs = ["ar", "es", "fr", "ru", "en"]

xvnli_root = "iglue/datasets/XVNLI/annotations"

label_map = {
    "neutral": "maybe",
    "entailment": "yes",
    "contradiction": "no"
}

train = [json.loads(row) for row in
         open(os.path.join("iglue/datasets/XVNLI/annotations/en", "train.jsonl")).readlines()]
train_input = []
for row in train:
    image = row["Flikr30kID"]
    pairID = row["pairID"]
    label = row["annotator_labels"][0]
    if label not in {"neutral", "contradiction", "entailment"}:
        print("train ", pairID)
        continue
    label = label_map[label]
    context = row["sentence2"]

    entry = {
        "context": context,
        "label": label,
        "image_id": image,
        "pair_id": pairID,
        "text_label": label
    }
    train_input.append(entry)

json.dump(train_input, open(f"xvnli_input_train.json", "w", encoding="utf-8"))


for lang in langs:

    dev = [json.loads(row) for row in open(os.path.join(xvnli_root, "en",
                                                        "dev.jsonl" if lang == "en" else f"dev-{lang}_gmt.jsonl")).readlines()]

    dev_input = []
    for row in dev:
        image = row["Flikr30kID"]
        pairID = row["pairID"]
        label = row["annotator_labels"][0]
        if label not in {"neutral", "contradiction", "entailment"}:
            print("val ", pairID)
            continue
        label = label_map[label]
        context = row["sentence2"]

        entry = {
            "context": context,
            "label": "",
            "image_id": image,
            "pair_id": pairID,
            "text_label": label
        }
        dev_input.append(entry)

    json.dump(dev_input, open(f"xvnli_input_val_{lang}.json", "w", encoding="utf-8"))

    test = [json.loads(row) for row in open(os.path.join(xvnli_root, lang, "test.jsonl")).readlines()]


    test_input = []
    for row in test:
        image = row["Flikr30kID"]
        pairID = row["pairID"]
        label = row["gold_label"]
        if label not in {"neutral", "contradiction", "entailment"}:
            print("test ", pairID)
            continue
        label = label_map[label]
        context = row["sentence2"]

        entry = {
            "context": context,
            "label": "",
            "image_id": image,
            "pair_id": pairID,
            "text_label": label
        }
        test_input.append(entry)
    json.dump(test_input, open(f"xvnli_input_test_{lang}.json", "w", encoding="utf-8"))