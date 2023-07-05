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
             "Briefly describe the image.", "Write a short image description.",
             "Summarize the image."]

def generate_en(data, out):
    random.seed(42)
    train_data = []
    for example in data:
        entry = {
            "context": random.choice(templates_en),
            "label": example["caption"],
            "image_id": example["jpg"].split("/")[-1],
        }
        train_data.append(entry)
        if len(train_data) < 10:
            print(entry)
    json.dump(train_data, open(out, "w"))

def generate_mt(data, mt_templates, out_file):
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
    json.dump(train_data, open(out_file, "w"))


if __name__ == "__main__":
    raw_file = "ccs_synthetic_filtered_large_2273005_raw.json"
    raw_data = json.load(open(raw_file))
    en_out = "ccs_synthetic_filtered_large_2273005_en.json"
    generate_en(raw_data, en_out)

    if not os.path.exists("ccs_synthetic_filtered_large_2273005_mt.json"):
        print("Run translate_train.py now to translate the captions before calling this script again to generate the data files.")
    else:
        mt_out = "ccs_synthetic_filtered_large_2273005_mt.json"
        raw_data = json.load(open("ccs_synthetic_filtered_large_2273005_mt_raw.json", encoding="utf-8"))
        templates_mt = json.load(open("caption_templates.json", encoding="utf-8"))
        generate_mt(raw_data, templates_mt, mt_out)