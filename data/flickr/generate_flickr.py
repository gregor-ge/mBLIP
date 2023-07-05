import json
import os
from collections import defaultdict

import numpy as np
import random

import unicodedata

from spacy.lang.zh import Chinese
from spacy.lang.th import Thai
from spacy.lang.ja import Japanese


lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}
lang2code = {v:k for k,v in lang_code2lang.items()}

templates_en = ["Caption the image.",
             "Short image caption: ",
             "Briefly describe the image.", "Write a short image description.",
             "Summarize the image."]

def generate_en(flickr_root):
    data = [json.loads(row) for row in open(os.path.join(flickr_root, "annotations", "train_ann.jsonl")).readlines()]
    random.seed(42)
    train_data = []
    for example in data:
        for sent in example["sentences"]:
            entry = {
                "context": random.choice(templates_en),
                "label": sent,
                "image_id": example["img_path"],
            }
            train_data.append(entry)
    print(len(train_data))
    json.dump(train_data, open("flickr_train.json", "w"))

def generate_eval_en(flickr_root):
    val_annotations = []
    val_images = []
    test_annotations = []
    test_images = []
    test_data = [json.loads(row) for row in open(os.path.join(flickr_root, "annotations", "test_ann.jsonl")).readlines()]
    for example in test_data:
        for sent in example["sentences"]:
            annot = {
                "id": len(test_annotations),
                "caption": sent,
                "image_id": example["img_path"],
            }
            test_annotations.append(annot)
        test_images.append(dict(id=example["img_path"]))

    val_data = [json.loads(row) for row in
                 open(os.path.join(flickr_root, "annotations", "valid_ann.jsonl")).readlines()]
    for example in val_data:
        for sent in example["sentences"]:
            annot = {
                "id": len(val_annotations),
                "caption": sent,
                "image_id": example["img_path"],
            }
            val_annotations.append(annot)
        val_images.append(dict(id=example["img_path"]))

    coco_val = {
        "annotations": val_annotations,
        "images": val_images
    }
    json.dump(coco_val, open(f"flickr_coco_val.json", "w", encoding="utf-8"))

    coco_test = {
        "annotations": test_annotations,
        "images": test_images
    }
    json.dump(coco_test, open(f"flickr__coco_test.json", "w", encoding="utf-8"))

    val_dataset = [
        {
            "context": "English",
            "label": "",
            "image_id": img["id"],
            "text_label": ""
        } for img in val_images
    ]
    json.dump(val_dataset, open(f"flickr_val.json", "w", encoding="utf-8"))
    test_dataset = [
        {
            "context": "English",
            "label": "",
            "image_id": img["id"],
            "text_label": ""
        } for img in test_images
    ]
    json.dump(test_dataset, open(f"flickr__test.json", "w", encoding="utf-8"))

def generate_eval(xflickr_root):
    chinese = Chinese()  # .from_config({"nlp": {"tokenizer": {"segmenter": "jieba"}}})
    japanese = Japanese()
    thai = Thai()
    for lang in ["de", "es", "id", "ja", "ru", "tr", "zh"]:
        test_annotations = []
        test_images = []
        test_data = [json.loads(row) for row in
                     open(os.path.join(xflickr_root, "annotations", lang, "test.jsonl")).readlines()]
        for example in test_data:
            for sent in example["sentences"]:
                if lang == "zh":
                    sent = " ".join([word.text for word in chinese(sent)])
                if lang == "ja":
                    sent = " ".join([word.text for word in japanese(sent)])
                if lang == "th":
                    sent = " ".join([word.text for word in thai(sent)])
                sent = unicodedata.normalize("NFC", sent)
                annot = {
                    "id": len(test_annotations),
                    "caption": sent,
                    "image_id": example["img_path"],
                }
                test_annotations.append(annot)
            test_images.append(dict(id=example["img_path"]))

        coco_test = {
            "annotations": test_annotations,
            "images": test_images
        }
        json.dump(coco_test, open(f"flickr_{lang}_coco_test.json", "w", encoding="utf-8"))
        test_dataset = [
            {
                "context": lang_code2lang[lang],
                "label": "",
                "image_id": img["id"],
                "text_label": lang
            } for img in test_images
        ]
        json.dump(test_dataset, open(f"flickr_{lang}_test.json", "w", encoding="utf-8"))

if __name__ == "__main__":
    flickr_root = "iglue/datasets/flickr30k"
    xflickr_root = "iglue/datasets/xFlickrCO"
    generate_en(flickr_root)
    generate_eval_en(flickr_root)
    generate_eval(xflickr_root)