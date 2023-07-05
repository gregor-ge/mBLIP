import json
import os
import re
import tempfile
import torch
import unicodedata
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from torchmetrics.functional.text import rouge_score
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from transformers import AutoTokenizer
import logging

# Validation Split Train Loss
def output_loss(self, outputs: dict, *args, **kwargs) -> dict:
    outputs["loss"] = outputs["loss"].detach().unsqueeze(0)
    return outputs

def validation_loss(loss):
    return torch.mean(loss)

# Caption Generation
def set_generation_mode(self, batch, mode="generate", *args, **kwargs):
    batch["mode"] = mode
    batch["generate_kwargs"] = kwargs
    return batch

class OutputGenerate:
    def __init__(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __call__(self, module, outputs=None, **kwargs) -> dict:
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output = dict(caption=captions)
        return output


class MyCOCOEvalCap(COCOEvalCap):
    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

def caption_evaluation(annotation_file, image_ids, captions, text_labels, print_examples=10):
    image_ids = [id for ids in image_ids for id in ids]
    captions = [c for caps in captions for c in caps]
    text_labels = [l for labels in text_labels for l in labels]

    if text_labels[0] in {"ja", "th", "zh"}:
        if text_labels[0] == "zh":
            from spacy.lang.zh import Chinese
            chinese = Chinese() #.from_config({"nlp": {"tokenizer": {"segmenter": "jieba"}}})
            captions = [" ".join([word.text for word in chinese(caption)]) for caption in captions]
        if text_labels[0] == "ja":
            from spacy.lang.ja import Japanese
            japanese = Japanese()
            captions = [" ".join([word.text for word in japanese(caption)]) for caption in captions]
        if text_labels[0] == "th":
            from spacy.lang.th import Thai
            thai = Thai()
            captions = [" ".join([word.text for word in thai(caption)]) for caption in captions]
        # tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-v-base")
        # captions = [" ".join(tokenizer.tokenize(c))[1:] for c in captions]
    captions = [unicodedata.normalize("NFC", c) for c in captions]

    for i in range(print_examples):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]}")

    prediction = [dict(image_id=imgid, caption=caption) for imgid, caption in zip (image_ids, captions)]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as temp_file:
        json.dump(prediction, temp_file)

        # For template annotation files, we take encode in text_label which file to load (not clean but works)
    if "{" in annotation_file:
        annotation_file = annotation_file.format(text_labels[0])
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(temp_file.name)

    # create coco_eval object by taking coco and coco_result
    coco_eval = MyCOCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    os.unlink(temp_file.name)
    results = dict()
    for metric, score in coco_eval.eval.items():
        results[metric] = score
    return results

def classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=True):
    image_ids = [id for ids in image_ids for id in ids]
    text_labels = [label for labels in text_labels for label in labels]
    captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)

    for i in range(print_examples):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    logging.info(f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
    logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")

    correct = 0
    total = 0
    relaxed_correct = 0
    for label, caption in zip(text_labels, captions):
        label = label.lower().strip()
        caption = caption.strip().lower()

        if label == caption:
            correct += 1
        if caption.startswith(label) or caption.endswith(label):
            relaxed_correct += 1
        total += 1
    acc = correct/total
    relaxed_acc = relaxed_correct/total
    return dict(acc=acc, relaxed_acc=relaxed_acc)
    # return acc


def vqa_classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=True):
    image_ids = [id for ids in image_ids for id in ids]
    text_labels = [label for labels in text_labels for label in labels]
    captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)

    for i in range(print_examples):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    correct = 0
    total = 0
    for labels, caption in zip(text_labels, captions):
        labels = [l.lower().strip() for l in labels]
        caption = caption.strip().lower()
        for i in range(len(labels)):
            other_labels = [labels[j] for j in range(len(labels)) if j!=i]
            hits = len([1 for label in other_labels if label==caption])
            correct += min(1, float(hits/3.0))
            total += 1
    acc = correct/total
    return acc

def vqa_maxm_classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=False):
    image_ids = [id for ids in image_ids for id in ids]
    text_labels = [label for labels in text_labels for label in labels]
    captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)

    for i in range(print_examples):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    correct = 0
    total = 0
    for labels, caption in zip(text_labels, captions):
        labels = [l.lower().strip() for l in labels]
        caption = caption.strip().lower()
        hits = len([1 for label in labels if label==caption])
        correct += min(1, hits)
        total += 1
    acc = correct/total

    # rouge = rouge_score(captions, text_labels, rouge_keys=("rougeL"))

    # return dict(acc=acc, rouge_l=rouge["rougeL_fmeasure"])

    return acc

#adapted from https://github.com/salesforce/LAVIS/blob/main/lavis/common/vqa_tools/vqa_eval.py
def vqa_clean(captions):
    manualMap = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = ["a", "an", "the"]
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(,)(\d)")
    punct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    cleaned_captions = []
    for cap in captions:
        cap = cap.replace("\n", "").replace("\t", "").strip()
        cap = processPunctuation(cap)
        cap = processDigitArticle(cap)
        cleaned_captions.append(cap)
    return cleaned_captions