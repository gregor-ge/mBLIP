import json
import os
from collections import defaultdict

import numpy as np
import random

lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}
lang2code = {v:k for k,v in lang_code2lang.items()}

templates_lang = ["Caption the image in {}.",
             "Short {} image caption: ", "Image caption (in {}): ",
             "Briefly describe the image in {}.", "Write a short {} image description.",
             "Summarize the image in {}."]

templates_en = ["Caption the image.",
             "Short image caption: ",
             "Briefly describe the image.",
                "Write a short image description.",
             "Summarize the image."]

def generate_en(data):
    random.seed(42)
    train_data = []
    for example in data["images"]:
        if example["split"] != "train":
            continue
        for sent in example["sentences"]:
            entry = {
                "context": random.choice(templates_en),
                "label": sent["raw"],
                "image_id": example["filename"],
            }
            train_data.append(entry)
    print(len(train_data))
    json.dump(train_data, open("coco_train_en.json", "w"))

def generate_mt(data, mt_templates, postfix=""):
    random.seed(42)
    train_data = []
    lang_counter = defaultdict(lambda: 0)
    for example in data:
        lang = example["context"]
        idx = lang_counter[lang] % (len(templates_lang)+len(mt_templates[lang2code[lang]]))
        if idx < len(templates_lang):
            context = templates_lang[idx].format(lang)
        else:
            context = mt_templates[lang2code[lang]][idx-len(templates_lang)]
        entry = {
            "context": context,
            "label": example["label"],
            "image_id": example["image_id"],
        }
        train_data.append(entry)
        lang_counter[lang] += 1
        if lang_counter[lang] <= 10:
            print(entry)
    print(len(train_data))
    json.dump(train_data, open(f"coco_train_mt{postfix}.json", "w"))


def generate_eval(data):
    val_annotations = []
    val_images = []
    test_annotations = []
    test_images = []
    for example in data["images"]:
        if example["split"] == "train":
            continue
        if example["split"] == "test":
            for sent in example["sentences"]:
                    annot = {
                        "id": len(test_annotations),
                        "caption": sent["raw"],
                        "image_id": example["filename"],
                    }
                    test_annotations.append(annot)
            test_images.append(dict(id=example["filename"]))
        else:
            for sent in example["sentences"]:
                    annot = {
                        "id": len(val_annotations),
                        "caption": sent["raw"],
                        "image_id": example["filename"],
                    }
                    val_annotations.append(annot)
            val_images.append(dict(id=example["filename"]))
            
    coco_val = {
        "annotations": val_annotations,
        "images": val_images
    }
    json.dump(coco_val, open(f"coco_coco_val.json", "w", encoding="utf-8"))

    coco_test = {
        "annotations": test_annotations,
        "images": test_images
    }
    json.dump(coco_test, open(f"coco_coco_test.json", "w", encoding="utf-8"))

    val_dataset = [
        {
            "context": "English",
            "label": "",
            "image_id": img["id"],
            "text_label": ""
        } for img in val_images
    ]
    json.dump(val_dataset, open(f"coco_val.json", "w", encoding="utf-8"))
    test_dataset = [
        {
            "context": "English",
            "label": "",
            "image_id": img["id"],
            "text_label": ""
        } for img in test_images
    ]
    json.dump(test_dataset, open(f"coco_test.json", "w", encoding="utf-8"))

if __name__ == "__main__":
    raw_data = json.load(open("mscoco/dataset_coco.json"))
    # generate_en(raw_data)
    # generate_eval(raw_data)

    if not os.path.exists("coco_train_mt_raw.json"):
        print("Run translate_train.py now to translate the captions before calling this script again to generate the data files.")
    else:
        raw_data = json.load(open("coco_train_mt_raw.json", encoding="utf-8"))
        templates_mt = json.load(open("caption_templates.json", encoding="utf-8"))
        generate_mt(raw_data, templates_mt)