import json
import os
from collections import defaultdict

import numpy as np
import random

lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}
lang2code = {v:k for k,v in lang_code2lang.items()}


vqa_templates = ["{0}. Short {1} answer:",
                "Question: {0}. Brief answer (in {1}):",
                 "Give a short answer in {1} to the following question. {0}",
                 "Answer the provided question in {1} with three words or less. {0}",
                 "What is the {1} answer to this question? {0}"]

vqg_templates = [
    "Given the image, generate a question in {} whose answer is: {}. Question:",
    'Based on the image, create a question (in {}) for which the answer is "{}".',
    'From the image provided, come up with a {} question that leads to the reply: {}. Question:'
]

def generate_raw(root):
    questions = json.load(open(os.path.join(root, "v2_OpenEnded_mscoco_train2014_questions.json")))
    annotations = json.load(open(os.path.join(root, "v2_mscoco_train2014_annotations.json")))

    raw_data = []
    for question, annotation in zip(questions["questions"], annotations["annotations"]):
        assert question["question_id"] == annotation["question_id"]
        image_id = f"COCO_train2014_{int(annotation['image_id']):012d}.jpg"

        entry = {
            "context": question["question"],
            "label": annotation["multiple_choice_answer"],
            "image_id": image_id
        }
        raw_data.append(entry)

    print(len(raw_data))
    json.dump(raw_data, open("vqav2_raw.json", "w"))

def generate_en(data):
    random.seed(42)
    vqa_data = []
    vqg_data = []
    for example in data:
        vqa_data.append({
            "context": random.choice(vqa_templates).format(example["context"]),
            "label": example["label"],
            "image_id": example["image_id"],
        })
        vqg_data.append({
            "context": random.choice(vqg_templates).format("English", example["label"]),
            "label": example["context"],
            "image_id": example["image_id"],
        })
    print(len(vqa_data), len(vqg_data))
    json.dump(vqa_data, open("vqav2_vqa_en.json", "w"))
    json.dump(vqg_data, open("vqav2_vqg_en.json", "w"))

def generate_mt(data_q, data_g, postfix=""):
    random.seed(42)
    vqa_data = []
    vqg_data = []
    print(len(data_g))
    mt_label_g_count = 0
    mt_label_q_count = 0
    for example_q, example_g in zip(data_q, data_g):
        if len(example_q["context"].split("#")) == 2:
            lang, context = example_q["context"].split("#")
            lang = lang.strip()
            label = example_q["label"]
            vqa_data.append({
                "context": random.choice(vqa_templates).format(context, lang),
                "label": label,
                "image_id": example_q["image_id"],
            })
        if len(example_g["context"].split("#")) == 2:
            # skip questions which have # in them (bad choice of delimiter in hindsight)
            lang, context = example_g["context"].split("#")
            lang = lang.strip()
            label = example_q["label"]
            vqg_data.append({
                "context": random.choice(vqg_templates).format(lang, label),
                "label": context,
                "image_id": example_g["image_id"],
            })
    print(len(vqa_data), len(vqg_data))
    print(mt_label_q_count, mt_label_g_count)
    json.dump(vqa_data, open(f"vqav2_vqa_mt{postfix}.json", "w"))
    json.dump(vqg_data, open(f"vqav2_vqg_mt{postfix}.json", "w"))


def generate_eval(root):
    questions = json.load(open(os.path.join(root, "v2_OpenEnded_mscoco_val2014_questions.json")))
    annotations = json.load(open(os.path.join(root, "v2_mscoco_val2014_annotations.json")))

    raw_data = []
    for question, annotation in zip(questions["questions"], annotations["annotations"]):
        assert question["question_id"] == annotation["question_id"]
        image_id = f"COCO_val2014_{int(annotation['image_id']):012d}.jpg"

        entry = {
            "context": question["question"],
            "label": "",
            "image_id": image_id,
            "text_label": [ans["answer"] for ans in annotation["answers"]]
        }
        raw_data.append(entry)
    json.dump(raw_data, open("vqav2_val.json", "w"))

if __name__ == "__main__":
    root = "vqav2"
    generate_raw(root)

    # raw_data = json.load(open("vqav2_raw.json"))
    # generate_en(raw_data)
    # generate_eval("iglue/datasets/vqav2")

    if not os.path.exists("vqav2_mt_q_raw.json"):
        print("Run translate_train.py now to translate the questions before calling this script again to generate the data files.")
    else:
        raw_data_q = json.load(open("vqav2_mt_q_raw.json", encoding="utf-8"))
        raw_data_g = json.load(open("vqav2_mt_g_raw.json", encoding="utf-8"))
        generate_mt(raw_data_q, raw_data_g)