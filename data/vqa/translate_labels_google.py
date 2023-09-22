import json
import os
import time
from copy import deepcopy
import re
import numpy as np
import torch
from collections import Counter
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, \
    AutoModelForSeq2SeqLM
from tqdm import tqdm
from deep_translator import GoogleTranslator

# mt5_lang_dist = {'en': 8.485280635632062, 'de': 3.106855454823264, 'ru': 3.7791586024243635, 'es': 3.1476011001324213, 'fr': 2.9438728735866335, 'it': 2.475297952531322, 'pt': 2.403993073240296, 'pl': 2.190078435367219, 'nl': 2.016909442803299, 'tr': 1.9659773861668521, 'ja': 1.9557909748395628, 'vi': 1.904858918203116, 'id': 1.8335540389120901, 'cs': 1.752062748293775, 'zh': 1.701130691657328, 'fa': 1.701130691657328, 'ar': 1.6909442803300387, 'sv': 1.6400122236935917, 'ro': 1.6094529897117236, 'el': 1.568707344402566, 'uk': 1.5381481104206978, 'hu': 1.5075888764388297, 'da': 1.4057247631659358, 'fi': 1.3751655291840676, 'no': 1.354792706529489, 'bg': 1.3140470612203312, 'hi': 1.2325557706020163, 'sk': 1.2121829479474373, 'ko': 1.1612508913109902, 'th': 1.1612508913109902, 'ca': 1.1408780686564117, 'ms': 1.1103188346745436, 'iw': 1.0797596006926753, 'lt': 1.0593867780380966, 'sl': 0.967709076092492, 'mr': 0.9473362534379133, 'bn': 0.9269634307833344, 'et': 0.9065906081287557, 'lv': 0.886217785474177, 'az': 0.83528572883773, 'gl': 0.8047264948558618, 'cy': 0.7741672608739937, 'sq': 0.7741672608739937, 'ta': 0.7436080268921255, 'sr': 0.7334216155648361, 'ne': 0.7028623815829679, 'lb': 0.6926759702556785, 'hy': 0.6621167362738103, 'kk': 0.6621167362738103, 'ka': 0.6519303249465209, 'mt': 0.6519303249465209, 'af': 0.6417439136192316, 'fil': 0.6315575022919422, 'is': 0.6315575022919422, 'mk': 0.6315575022919422, 'ml': 0.6315575022919422, 'mn': 0.6315575022919422, 'ur': 0.6213710909646528, 'be': 0.600998268310074, 'eu': 0.5806254456554951, 'tg': 0.550066211673627, 'te': 0.5296933890190483, 'kn': 0.5195069776917589, 'ky': 0.5093205663644695, 'sw': 0.5093205663644695, 'so': 0.4889477437098907, 'my': 0.4787613323826013, 'uz': 0.46857492105531195, 'km': 0.46857492105531195, 'sd': 0.45838850972802253, 'gu': 0.43801568707344374, 'jv': 0.42782927574615437, 'zu': 0.42782927574615437, 'si': 0.417642864418865, 'eo': 0.4074564530915756, 'ga': 0.4074564530915756, 'pa': 0.3768972191097074, 'ceb': 0.36671080778241805, 'mg': 0.36671080778241805, 'ps': 0.36671080778241805, 'sn': 0.3565243964551286, 'gd': 0.30559233981868167, 'ku': 0.34633798512783925, 'su': 0.34633798512783925, 'ht': 0.30559233981868167, 'ha': 0.3361515738005499, 'ny': 0.29540592849139224, 'am': 0.29540592849139224, 'yi': 0.28521951716410293, 'lo': 0.28521951716410293, 'mi': 0.25466028318223477, 'sm': 0.25466028318223477, 'ig': 0.24447387185494535, 'xh': 0.22410104920036658, 'st': 0.22410104920036658, 'yo': 0.2037282265457878}
lang_code2lang = {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'az': 'Azerbaijani', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bn': 'Bangla', 'ca': 'Catalan', 'ceb': 'Cebuano', 'co': 'Corsican', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'fi': 'Finnish', 'fil': 'Filipino', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'haw': 'Hawaiian', 'hi': 'Hindi', 'hmn': 'Hmong, Mong', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'id': 'Indonesian', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'former Hebrew', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'ku': 'Kurdish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lo': 'Lao', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mg': 'Malagasy', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'ny': 'Nyanja', 'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sd': 'Sindhi', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'st': 'Southern Sotho', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'und': 'Unknown language', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zh': 'Chinese', 'zu': 'Zulu'}
# mt5_langs = list(mt5_lang_dist.keys())[1:]
mt5_langs = ['en', 'de', 'ru', 'es', 'fr', 'it', 'pt', 'pl', 'nl', 'tr', 'ja', 'vi', 'id', 'cs', 'zh', 'fa', 'ar', 'sv', 'ro', 'el', 'uk', 'hu', 'da', 'fi', 'no', 'bg', 'hi', 'sk', 'ko', 'th', 'ca', 'ms', 'iw', 'lt', 'sl', 'mr', 'bn', 'et', 'lv', 'az', 'gl', 'cy', 'sq', 'ta', 'sr', 'ne', 'lb', 'hy', 'kk', 'ka', 'mt', 'af', 'fil', 'is', 'mk', 'ml', 'mn', 'ur', 'be', 'eu', 'tg', 'te', 'kn', 'ky', 'sw', 'so', 'my', 'uz', 'km', 'sd', 'gu', 'jv', 'zu', 'si', 'eo', 'ga', 'pa', 'ceb', 'mg', 'ps', 'sn', 'gd', 'ku', 'su', 'ht', 'ha', 'ny', 'am', 'yi', 'lo', 'mi', 'sm', 'ig', 'xh', 'st', 'yo']
print(mt5_langs)

root = "."
file = "vqav2_raw.json"
data = json.load(open(os.path.join(root, file)))

labels_counter = Counter(d["label"] for d in data)
print("# labels:", len(labels_counter))
pattern = re.compile(r'\d')
labels = [l for l, num in labels_counter.items() if num >= 15 and not pattern.search(l) and len(l)>1]
print("# labels:", len(labels))

proxies = {
    "https": "13.236.6.61"
}

def translate(translator, labels):
    results = []
    bs = 100
    for i in tqdm(range(0, len(labels), bs)):
        ls = labels[i:i+bs]
        translations = translator.translate(text=" # ".join(ls), proxies=proxies)
        t_ls = translations.split("#")
        t_ls = [l.strip() for l in t_ls]
        if len(t_ls) < len(ls):
            print("Labels disappeared. Padding so this does not affect next batch")
            t_ls = t_ls + ["" for _ in range(len(ls)-len(t_ls))]
        if len(t_ls) > len(ls):
            print("Too many labels. Truncating")
            t_ls = t_ls[:len(ls)]
        assert len(t_ls) == len(ls), f"translations {len(t_ls)}, correct {len(ls)}"
        results.extend(t_ls)
        time.sleep(1)
    return results

mt_data = dict()
print(f"Total examples: {len(data)}")

if os.path.exists("vqav2_mt_google_labels.json"):
    mt_data = json.load(open("vqav2_mt_google_labels.json"))
    done = set(mt_data.keys())
else:
    done = set()

for lang in tqdm(mt5_langs, total=len(mt5_langs)):
    print(f"{lang}")
    if lang == "en" or lang in done:
        print("Skip ", lang)
        continue
    if lang == "zh":
        target = "zh-CN"
    elif lang == "fil":
        target = "tl"
    elif lang == "jv":
        target = "jw"
    else:
        target = lang
    translator = GoogleTranslator(source='en', target=target)
    back_translator = GoogleTranslator(source=target, target="en")

    translated_labels = translate(translator, labels)
    back_translated = translate(back_translator, translated_labels)

    keep = dict()
    for gold, back, mt in zip(labels, back_translated, translated_labels):
        if gold.lower() == back.lower():
            keep[gold] = mt
    print()
    print(len(keep))
    print("Total questions", sum([labels_counter[k] for k in keep.keys()]))
    print("Total questions (without yes/no)", sum([labels_counter[k] for k in keep.keys() if k not in {"yes", "no"}]))
    mt_data[lang] = keep

    with open(os.path.join(root, f"vqav2_mt_google_labels.json"), "w", encoding="utf-8") as f:
        json.dump(mt_data, f, ensure_ascii=False, indent=2)


