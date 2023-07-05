import os
import json
import unicodedata

marvl_root = "iglue/datasets/marvl/zero_shot/annotations"
marvl_image_root = "iglue/datasets/marvl/images"

langs = ["id", "sw", "ta", "tr", "zh"]

for lang in langs:
    print(lang)
    test = [json.loads(row) for row in open(os.path.join(marvl_root, f"marvl-{lang}.jsonl")).readlines()]

    test_input = []
    for row in test:
        concept = row["concept"]
        language = row["language"]
        caption = row["caption"]
        left_image_path = unicodedata.normalize("NFKD", os.path.join(language, "images", concept, row["left_img"]))
        right_image_path = unicodedata.normalize("NFKD", os.path.join(language, "images", concept, row["right_img"]))
        # making sure that the non-ascii paths are correctly encoded and match the file system
        assert os.path.isfile(os.path.join(marvl_image_root, left_image_path))
        assert os.path.isfile(os.path.join(marvl_image_root, right_image_path))
        label = str(row["label"])
        
        entry = {
            "context": caption,
            "concept": concept,
            "language": language,
            "left_image": left_image_path,
            "right_image": right_image_path,
            "label": "",
            "text_label": label
        }
        test_input.append(entry)
        
    json.dump(test_input, open(f"marvl_test_{lang}.json", "w", encoding="utf-8"), ensure_ascii=False)

    # making REALLY sure that the non-ascii paths are correctly encoded and match the file system
    test_input = json.load(open(f"marvl_test_{lang}.json", "r", encoding="utf-8"))
    for entry in test_input:
        assert os.path.isfile(os.path.join(marvl_image_root, entry["left_image"]))
        assert os.path.isfile(os.path.join(marvl_image_root, entry["right_image"]))

