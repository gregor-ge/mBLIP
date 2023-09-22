import random
import json
import torch
from PIL import Image
import open_clip
import datasets
import os
from tqdm import tqdm

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
    images = torch.stack([preprocess(Image.open(os.path.join("images", image))) for image in images]).cuda()
    with torch.no_grad():
        image_embedding = model.encode_image(images).cpu().numpy()

    scores, retrieved_examples = ds.get_nearest_examples_batch('text_embedding', image_embedding, k=k)
    return scores, retrieved_examples


def generate_match(out_file="ccs_synthetic_match_1000-601000_en_raw.json"):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.cuda()

    ds = get_dataset(model, tokenizer)

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

generate_match()