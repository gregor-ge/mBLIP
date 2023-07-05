import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFile
from torchvision.transforms import transforms, InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from transformers import Blip2Processor, AutoTokenizer, PreTrainedTokenizerBase

@dataclass
class DataCollatorForVisualCLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token='[PAD]'

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = {key: [example[key] for example in features] for key in features[0].keys()}

        text_features = {k: features[k] for k in ["input_ids", "attention_mask"]}
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch["labels"] = torch.tensor(features["labels"])
        batch["pixel_values"] = torch.stack(features["pixel_values"])

        # add everything else as is
        for k, v in features.items():
            if k not in batch:
                batch[k] = v

        batch = batch.data  # BatchEncoding to dict
        return batch


class TokenizeCLM:
    def __init__(self, pretrained_model, context_column="sentence", target_column="sentence", template="{}",
                 max_len=256, target2str=None, text_target_column=None):
        self.context_column = context_column
        self.target_column = target_column
        self.template = template
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.decoder_only = "bloom" in pretrained_model or "llama" in pretrained_model
        self.target2str = target2str
        self.text_target_column = text_target_column
        self.max_len = max_len
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token='[PAD]'

    def __call__(self, examples):
        batch_size = len(examples[self.target_column])
        inputs = [self.template.format(x) for x in examples[self.context_column]]
        targets = [x if self.target2str is None or not x else self.target2str[x] for x in examples[self.target_column]]
        model_inputs = self.tokenizer(inputs, truncation=True, max_length=self.max_len)
        if self.target2str is not None and self.text_target_column is not None:
            model_inputs[self.text_target_column] = [self.target2str[x] for x in examples[self.text_target_column]]
        labels = self.tokenizer(targets, truncation=True, max_length=self.max_len)
        if self.decoder_only:  # have to mask the context for the loss
            for i in range(batch_size):
                sample_input_ids = model_inputs["input_ids"][i]
                label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
                model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
                labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
                model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class LoadTransformImage:
    def __init__(self, image_root, processor="Salesforce/blip2-flan-t5-xl", target_column="image_id", extension="",
                 train=False, train_scale=(0.5, 1.0), overwrite_image_size=None):
        self.image_root = image_root
        self.multiple_roots = not isinstance(image_root, str)
        self.target_column = target_column
        self.extension = extension
        self.transform = Blip2Processor.from_pretrained(processor)
        if overwrite_image_size:
            self.transform.size = {"height": overwrite_image_size, "width": overwrite_image_size}
        self.train = train
        if train:
            img_size = self.transform.image_processor.size["height"]
            min_scale, max_scale = train_scale
            mean = self.transform.image_processor.image_mean
            std = self.transform.image_processor.image_std
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        img_size,
                        scale=(min_scale, max_scale),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        self.error_images = set()

    def __call__(self, examples):
        all_imgs = []
        for img_id in examples[self.target_column]:
            if self.multiple_roots:
                image_root = ""
                for root in self.image_root:
                    if os.path.isfile(os.path.join(root, img_id + self.extension)):
                        image_root = root
            else:
                image_root = self.image_root

            image_path = os.path.join(image_root, img_id + self.extension)
            try:
                image = Image.open(image_path).convert('RGB')
                if self.train:
                    image = self.transform(image)
                else:
                    image = self.transform(image, return_tensors="pt")["pixel_values"].squeeze()
            except Exception as e:
                if image_path not in self.error_images:
                    self.error_images.add(image_path)
                    print("Failed to load image ", image_path)
                image = torch.zeros((3, 224, 224), dtype=torch.float32)
            all_imgs.append(image)

        examples["pixel_values"] = all_imgs
        return examples
    

class LoadTransformImageMarvl:
    def __init__(self, image_root, processor="Salesforce/blip2-flan-t5-xl", left_image_col="left_image", right_image_col="right_image", extension="",
                 train=False, train_scale=(0.5, 1.0)):
        self.image_root = image_root
        self.multiple_roots = not isinstance(image_root, str)
        self.left_image_col = left_image_col
        self.right_image_col = right_image_col
        self.extension = extension
        self.transform = Blip2Processor.from_pretrained(processor)
        self.train = train
        if train:
            img_size = self.transform.image_processor.size["height"]
            min_scale, max_scale = train_scale
            mean = self.transform.image_processor.image_mean
            std = self.transform.image_processor.image_std
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        img_size,
                        scale=(min_scale, max_scale),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        self.error_images = set()

    def __call__(self, examples):
        all_imgs = []
        for i in range(len(examples[self.left_image_col])):
            left_image_path = os.path.join(self.image_root, examples[self.left_image_col][i])
            right_image_path = os.path.join(self.image_root, examples[self.right_image_col][i])
            
            images = []
            try:
                left_image = Image.open(left_image_path).convert('RGB')
                left_image = self.transform(left_image, return_tensors="pt")["pixel_values"].squeeze()
            except Exception as e:
                if left_image_path not in self.error_images:
                    self.error_images.add(left_image_path)
                    print("Failed to load left image ", left_image_path)
                left_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            try:
                right_image = Image.open(right_image_path).convert('RGB')
                right_image = self.transform(right_image, return_tensors="pt")["pixel_values"].squeeze()
            except Exception as e:
                if right_image_path not in self.error_images:
                    self.error_images.add(right_image_path)
                    print("Failed to load image ", right_image_path)
                right_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            images.append(left_image)
            images.append(right_image)
            images = torch.stack(images, dim=0)
            all_imgs.append(images)
        
        all_imgs = torch.stack(all_imgs, dim=0)
        examples["pixel_values"] = all_imgs
        return examples

