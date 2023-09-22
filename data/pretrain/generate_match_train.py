import random
import json
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import open_clip
import datasets
import os
from tqdm import tqdm
from transformers import AutoTokenizer

matching_templates = [
    ('Does "{}" accurately describe the image?', 'Yes, it does.', 'No, it does not.'),
    ('Does the caption "{}" fit the picture?', 'Yes, it does.', 'No, it does not.'),
    ('Does  "{}" correctly summarize the image?', 'Yes, it does.', 'No, it does not.'),
    ('Is "{}" a good image description?', 'Yes, it is.', 'No, it is not.'),
    ('Is "{}" a correct caption for the picture?', 'Yes, it is.', 'No, it is not.'),
    ('Is the caption "{}" a good match for the image?', 'Yes, it is.', 'No, it is not.'),
    ("Decide if the following caption accurately describes the image: {}. Answer:", 'Yes, it does.', 'No, it does not.'),
    ("Is this caption a good match for the picture? {}. Answer: ", 'Yes, it is.', 'No, it is not.'),
    ("Decide if this captions is a correct summary of the image: {}.", 'Yes, it is.', 'No, it is not.'),
    ('Would "{}" be a good image summary?', 'Yes, it would.', 'No, it would not.'),
    ('Would the caption "{}" fit the picture?', 'Yes, it would.', 'No, it would not.'),
    ('Could you use "{}" as a caption for the image?', 'Yes, you could.', 'No, you could not.'),
]

def get_dataset(model, tokenizer):
    ds = datasets.load_dataset('json', split="train[1000:601000]", data_files="ccs_synthetic_filtered_large_2273005_en.json", num_proc=1)

    if not os.path.exists("ccs_synthetic_filtered_large_2273005_en.faiss"):
        def encode(examples):
            with torch.no_grad():
                input = tokenizer(examples["label"]).cuda()
                return {
                    'text_embedding': model.encode_text(input).cpu().numpy()
                }
        ds_with_embeddings = ds.map(encode, batched=True, batch_size=512)
        ds_with_embeddings.add_faiss_index(column='text_embedding')
        ds_with_embeddings.save_faiss_index('text_embedding', 'ccs_synthetic_filtered_large_2273005_en.faiss')
    else:
        ds.load_faiss_index('text_embedding', 'ccs_synthetic_filtered_large_2273005_en.faiss')
        ds_with_embeddings = ds
    return ds_with_embeddings


def get_examples(images, ds, model, preprocess, k=30):
    images = torch.stack([preprocess(Image.open(os.path.join("images_1m", image))) for image in images]).cuda()
    with torch.no_grad():
        image_embedding = model.encode_image(images).cpu().numpy()

    scores, retrieved_examples = ds.get_nearest_examples_batch('text_embedding', image_embedding, k=k)
    return scores, retrieved_examples


def generate_match(out_file="ccs_synthetic_match_1000-601000_en_raw.json"):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.cuda()

    ds = get_dataset(model, tokenizer)
    #
    # for idx in range(100, 130):
    #     print("##")
    #     print(ds[idx]["label"])
    #     scores, examples = get_examples(ds[idx]["image_id"], ds, model, preprocess, k=300)
    #     sentences = examples["label"]
    #     print(sentences[3])
    #     print(sentences[10])
    #     print(sentences[30])
    #     print(sentences[100])
    #     print(sentences[-1])
    #     print(ds[random.randint(0, 600000)]["label"])
    #     pass

    data = []
    batchsize = 125
    for i in tqdm(range(0, len(ds), batchsize)):
        exs = ds[i:i+batchsize]
        exs = [dict(zip(exs.keys(), values)) for values in zip(*exs.values())]
        if i%(2*batchsize) == 0:
            for ex in exs:
                ex["context"] = ex["label"]
                ex["original_label"] = ex["label"]
                ex["label"] = "yes"
            data.extend(exs)
        else:
            scores, examples = get_examples([ex["image_id"] for ex in exs], ds, model, preprocess, k=300)
            for ex, example in zip(exs, examples):
                sentences = example["label"]
                candidates = [ds[random.randint(0, len(ds))]["label"], sentences[2],
                              sentences[9], sentences[29], sentences[99], sentences[299]]
                candidate = random.choice(candidates)
                ex["context"] = candidate
                ex["original_label"] = ex["label"]
                ex["label"] = "no"
                data.append(ex)

    json.dump(data, open(out_file, "w"))


def generate_mt(data="ccs_synthetic_match_1000-601000_mt_seed42_raw.json",
                out_file="ccs_synthetic_match_1000-601000_mt_seed42.json"):
    data = json.load(open(data))
    examples = []

    for d in data:
        template, yes, no = random.choice(matching_templates)
        lang, context = d["context"].split("###")
        lang = lang.strip()
        context = context.strip()
        if d["label"] == "yes":
            entry = {
                "context": template.format(context),
                "label": yes,
                "image_id": d["image_id"]
            }
        else:
            entry = {
                "context": template.format(context),
                "label": no,
                "image_id": d["image_id"]
            }
        examples.append(entry)

    # tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xl")
    #
    # lens = tokenizer([e["context"] for e in examples])["input_ids"]
    # lens = [len(v) for v in lens]
    #
    # print(np.percentile(lens, q=[50, 75, 90, 85, 99]))


    json.dump(examples, open(out_file, "w", encoding="utf-8"), ensure_ascii=False)



# generate_match()

generate_mt()