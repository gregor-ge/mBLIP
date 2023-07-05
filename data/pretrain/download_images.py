import random

import concurrent
import io
import nltk.corpus
import urllib
from copy import deepcopy
import os
import albumentations as A
import cv2
import numpy as np
from heapq import heapify, heappush, heappushpop
import datasets
from collections import defaultdict
import requests
from PIL import Image
import re
from tqdm import tqdm
import string
import json
import os
import csv

def download_single(url, filename):
    final_size = 256
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
    }
    row = dict()
    row["url"] = url
    fname = filename #os.path.join(root, filename)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        return row

    img_stream = None
    try:
        #response = requests.get(row['url'], stream=False, timeout=50, allow_redirects=True, headers=headers)
        request = urllib.request.Request(url, data=None, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as r:
            img_stream = io.BytesIO(r.read())
        #row['headers'] = dict(response.headers)
    except Exception as e:
        if img_stream is not None:
            img_stream.close()
        return row

    try:
        img_stream.seek(0)
        cv2.setNumThreads(1)
        img_buf = np.frombuffer(img_stream.read(), np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[-1] == 4:
            # alpha matting with white background
            alpha = img[:, :, 3, np.newaxis]
            img = alpha / 255 * img[..., :3] + 255 - alpha
            img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
        original_height, original_width = img.shape[:2]

        downscale = max(original_width, original_height) > final_size
        if downscale:
            img = A.longest_max_size(img, final_size, interpolation=cv2.INTER_CUBIC)
            img = A.pad(
                img,
                final_size,
                final_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

        img = cv2.imencode(f".jpg", img, params=[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

        img_stream.close()
        del img_stream

        with open(fname, "wb") as f:
            f.write(img)

    except:
        return row
    return row


def download(data):
    pbar = tqdm(total=len(data), position=0, desc="Downloading")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_single, entry["url"], entry["jpg"]) for entry in data]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    root = "."
    image_folder = "images"
    os.makedirs(os.path.join(root, image_folder), exist_ok=True)
    data_file = f"ccs_synthetic_filtered_large-npfilter_min10_max30.json"
    data = json.load(open(os.path.join(root, data_file), encoding="utf-8"))
    for i, entry in enumerate(data):
        entry["jpg"] = os.path.join(root, image_folder, f"{i:08d}.jpg")

    download(data)

    # Filter data down to those examples where we successfully donwloaded an image
    _data = []
    for entry in tqdm(data):
        if os.path.isfile(entry["jpg"]):
            _data.append(entry)
    print(f"{len(_data)}/{len(data)}")
    with open(os.path.join(root, f"ccs_synthetic_filtered_large_{len(_data)}_raw.json"), "w", encoding="utf-8") as f:
        json.dump(_data, f)
