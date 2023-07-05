import json
import os
import unicodedata
from collections import defaultdict
from spacy.lang.zh import Chinese
from spacy.lang.th import Thai
from spacy.lang.ja import Japanese

raw_data = [json.loads(row) for row in open("Crossmodal3600/captions.jsonl", "r", encoding="utf-8").readlines()]

images = []
annotations = defaultdict(list)

xm3600_langs = ['fa', 'te', 'ko', 'fi', 'fil', 'mi', 'hu', 'id', 'hr', 'fr', 'quz',
                'sv', 'zh', 'sw', 'no', 'vi', 'da', 'ja', 'nl', 'he', 'th', 'ru',
                'it', 'hi', 'uk', 'de', 'pt', 'tr', 'cs', 'pl', 'bn', 'ar', 'ro', 'en', 'es', 'el']

full_lang = ["Farsi", "Telugu", "Korean", "Finnish", "Filipino", "Maori", "Hungarian", "Indonesian", "Croatian", "French", "Quechua",
             "Swedish", "Chinese", "Swahili", "Norwegian", "Vietnamese", "Danish", "Japanese", "Dutch", "Hebrew", "Thai", "Russian",
             "Italian", "Hindi", "Ukranian", "German", "Portugese", "Turkish", "Czech", "Polish", "Bengali", "Arabic",
             "Romanian", "English", "Spanish", "Greek"]

print(len(xm3600_langs), len(full_lang))


# tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-v-base")
chinese = Chinese() #.from_config({"nlp": {"tokenizer": {"segmenter": "jieba"}}})
japanese = Japanese()
thai = Thai()

for row in raw_data:
    image = row["image/key"]
    for lang in xm3600_langs:
        if lang in row:
            for caption in row[lang]["caption"]:
                if lang == "zh":
                    caption = " ".join([word.text for word in chinese(caption)])
                if lang == "ja":
                    caption = " ".join([word.text for word in japanese(caption)])
                if lang == "th":
                    caption = " ".join([word.text for word in thai(caption)])
                caption = unicodedata.normalize("NFC", caption)
                annotations[lang].append({
                    "image_id": image,
                    "id": len(annotations[lang]),
                    "caption": caption
                })
    images.append(dict(id=image))


for lang, full in zip(xm3600_langs, full_lang):
    coco = {
        "annotations": annotations[lang],
        "images": images
    }
    json.dump(coco, open(f"xm3600_coco_{lang}.json", "w", encoding="utf-8"))

    eval_dataset = [
        {
            "context": full,
            "label": "",
            "image_id": img["id"],
            "text_label": lang
        } for img in images
    ]
    json.dump(eval_dataset, open(f"xm3600_{lang}.json", "w", encoding="utf-8"))