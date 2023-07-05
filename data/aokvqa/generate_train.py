import json
import os
from collections import defaultdict

import numpy as np
import random

lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}
lang2code = {v:k for k,v in lang_code2lang.items()}


cot_answer_templates = ["{0}. So the answer is {1}",
                        "{0} so {1}",
                        "{0}. This means the answer is {1}"]

cot_question_templates = ["Reason the answer to the following question. {}",
                          "Use reasoning to come to an answer for this question. {}",
                          "Think step-by-step to answer this question. {}"]

explanation_templates = [
    "Question: {}: Answer: {}. Explanation:",
    "Question: {}: Answer: {}. The reason is because ",
    'The answer to the question "{}" is "{}". Why?',
    'Why is the answer to the question "{}"  "{}"?',
    'Explain why the answer to the question "{}" is "{}".',
]


def generate_en(data):
    random.seed(42)
    mscoco_image_root = "iglue/datasets/coco_images"

    cot_data = []
    explain_data = []

    for example in data:
        assert example["split"] == "train"
        image_id = f"COCO_train2014_{int(example['image_id']):012d}.jpg"
        if not os.path.exists(os.path.join(mscoco_image_root, image_id)):
            #Skip examples with images not in coco train split
            continue

        answer = example["choices"][example["correct_choice_idx"]]
        for rationale in example["rationales"]:
            explain_entry = {
                "context": random.choice(explanation_templates).format(example["question"], answer),
                "label": rationale,
                "image_id": image_id
            }
            explain_data.append(explain_entry)
            if rationale[-1] == ".":
                rationale = rationale[:-1]

            cot_entry = {
                "context": random.choice(cot_question_templates).format(example["question"]),
                "label": random.choice(cot_answer_templates).format(rationale, answer),
                "image_id": image_id
            }
            cot_data.append(cot_entry)
            if len(cot_data) < 5:
                print(explain_entry)
                print(cot_entry)
    print(len(explain_data), len(cot_data))
    json.dump(explain_data, open("aokvqa_explain_en.json", "w"))
    json.dump(cot_data, open("aokvqa_cot_en_v2.json", "w"))




if __name__ == "__main__":
    data = json.load(open("a-okvqa/aokvqa_v1p0_train.json"))
    generate_en(data)
