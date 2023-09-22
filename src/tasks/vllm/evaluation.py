import json
import os
import re
import tempfile
from collections import Counter

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

    try:
        logging.info(f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
        logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")
        logging.info(f"Count: Total {len(captions)} | {Counter([c.split()[0] for c in captions])}")
    except:
        pass
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

    try:
        logging.info(f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
        logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")
    except:
        pass
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


chair_coco_object_synonyms = [['person', 'girl', 'boy', 'man', 'woman', 'kid', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'skier', 'snowboarder', 'skater', 'skateboarder', 'person', 'woman', 'guy', 'foreigner', 'child', 'gentleman', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager'], ['bicycle', 'bike', 'bicycle', 'bike', 'unicycle', 'minibike', 'trike'], ['car', 'automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi'], ['motorcycle', 'scooter', ' motor bike', 'motor cycle', 'motorbike', 'scooter', 'moped'], ['airplane', 'jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'jetliner', 'airbus', 'biplane', 'seaplane'], ['bus', 'minibus', 'trolley'], ['train', 'locomotive', 'tramway', 'caboose'], ['truck', 'pickup', 'lorry', 'hauler', 'firetruck'], ['boat', 'ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship'], ['traffic light', 'street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight'], ['fire hydrant', 'hydrant'], ['stop sign'], ['parking meter'], ['bench', 'pew'], ['bird', 'ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird'], ['cat', 'kitten', 'feline', 'tabby'], ['dog', 'puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky'], ['horse', 'colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco'], ['sheep', 'lamb', 'ram', 'lamb', 'goat', 'ewe'], ['cow', 'cattle', 'oxen', 'ox', 'calf', 'cattle', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison'], ['elephant'], ['bear', 'panda'], ['zebra'], ['giraffe'], ['backpack', 'knapsack'], ['umbrella'], ['handbag', 'wallet', 'purse', 'briefcase'], ['tie', 'bow', 'bow tie'], ['suitcase', 'suit case', 'luggage'], ['frisbee'], ['skis', 'ski'], ['snowboard'], ['sports ball', 'ball'], ['kite'], ['baseball bat'], ['baseball glove'], ['skateboard'], ['surfboard', 'longboard', 'skimboard', 'shortboard', 'wakeboard'], ['tennis racket', 'racket'], ['bottle'], ['wine glass'], ['cup'], ['fork'], ['knife', 'pocketknife', 'knive'], ['spoon'], ['bowl', 'container'], ['banana'], ['apple'], ['sandwich', 'burger', 'sub', 'cheeseburger', 'hamburger'], ['orange'], ['broccoli'], ['carrot'], ['hot dog'], ['pizza'], ['donut', 'doughnut', 'bagel'], ['cake', ' cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake'], ['chair', 'seat', 'stool'], ['couch', 'sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield'], ['potted plant', 'houseplant'], ['bed'], ['dining table', 'table', 'desk'], ['toilet', 'urinal', 'commode', 'toilet', 'lavatory', 'potty'], ['tv', 'monitor', 'televison', 'television'], ['laptop', 'computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer'], ['mouse'], ['remote'], ['keyboard'], ['cell phone', 'mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone'], ['microwave'], ['oven', 'stovetop', 'stove', 'stove top oven'], ['toaster'], ['sink'], ['refrigerator', 'fridge', 'fridge', 'freezer'], ['book'], ['clock'], ['vase'], ['scissors'], ['teddy bear', 'teddybear'], ['hair drier', 'hairdryer'], ['toothbrush']]
def chair(image_ids, text_labels, captions, print_examples=10):
    import nltk
    nltk.download('punkt')
    # nltk.download('omw-1.4')

    image_ids = [id for ids in image_ids for id in ids]
    text_labels = [label for labels in text_labels for label in labels]
    captions = [c for caps in captions for c in caps]

    print(f"Average characters: {sum(len(c) for c in captions)/len(captions)}")

    synonyms = chair_coco_object_synonyms
    mscoco_objects = []  # mscoco objects and *all* synonyms
    inverse_synonym_dict = {}
    for synonym in synonyms:
        mscoco_objects.extend(synonym)
        for s in synonym:
            inverse_synonym_dict[s] = synonym[0]

    # Some hard coded rules for implementing CHAIR metrics on MSCOCO

    # common 'double words' in MSCOCO that should be treated as a single word
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                         'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
                         'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
                         'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
                         'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog',
                         'teddy bear', 'home plate', 'train track']

    # Hard code some rules for special cases in MSCOCO
    # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'animal', 'cub']
    # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
    vehicle_words = ['jet', 'train']

    # double_word_dict will map double words to the word they should be treated as in our analysis

    double_word_dict = {}
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for animal_word in animal_words:
        double_word_dict['baby %s' % animal_word] = animal_word
        double_word_dict['adult %s' % animal_word] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict['passenger %s' % vehicle_word] = vehicle_word
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'
    def caption_to_words(caption):
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i + 2])
            if double_word in double_word_dict:
                double_words.append(double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(mscoco_objects)]
        words = [word for word in words if word in set(mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    num_caps = 0.
    num_hallucinated_caps = 0.
    hallucinated_word_count = 0.
    coco_word_count = 0.

    for i, (iid, cap, gt_objects) in enumerate(zip(image_ids, captions, text_labels)):


        # get all words in the caption, as well as corresponding node word
        words, node_words, idxs, raw_words = caption_to_words(cap)

        # count hallucinated words
        coco_word_count += len(node_words)
        hallucinated = False
        mscoco_hallucinated_words = []
        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                hallucinated_word_count += 1
                mscoco_hallucinated_words.append((word, node_word))
                # cap_dict['hallucination_idxs'].append(idx)
                hallucinated = True
        if print_examples > 0 and i < print_examples:
            print(f"{iid} -- {cap} -- {words} {node_words} -- {mscoco_hallucinated_words}")
        # count hallucinated caps
        num_caps += 1
        if hallucinated:
            num_hallucinated_caps += 1

        # cap_dict['metrics']['CHAIRs'] = int(hallucinated)
        # cap_dict['metrics']['CHAIRi'] = 0.
        # if len(words) > 0:
        #     cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words))


    chair_s = (num_hallucinated_caps / num_caps)
    chair_i = (hallucinated_word_count / coco_word_count)

    return dict(chair_s=chair_s, chair_i=chair_i)


#### SINGULARIZE ###################################################################################
# Adapted from Bermi Ferrer's Inflector for Python:
# http://www.bermi.org/inflector/

# Copyright (c) 2006 Bermi Ferrer Martinez
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software to deal in this software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of this software, and to permit
# persons to whom this software is furnished to do so, subject to the following
# condition:
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THIS SOFTWARE.

singular_rules = [
    (r'(?i)(.)ae$'            , '\\1a'    ),
    (r'(?i)(.)itis$'          , '\\1itis' ),
    (r'(?i)(.)eaux$'          , '\\1eau'  ),
    (r'(?i)(quiz)zes$'        , '\\1'     ),
    (r'(?i)(matr)ices$'       , '\\1ix'   ),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'   ),
    (r'(?i)^(ox)en'           , '\\1'     ),
    (r'(?i)(alias|status)es$' , '\\1'     ),
    (r'(?i)([octop|vir])i$'   , '\\1us'  ),
    (r'(?i)(cris|ax|test)es$' , '\\1is'   ),
    (r'(?i)(shoe)s$'          , '\\1'     ),
    (r'(?i)(o)es$'            , '\\1'     ),
    (r'(?i)(bus)es$'          , '\\1'     ),
    (r'(?i)([m|l])ice$'       , '\\1ouse' ),
    (r'(?i)(x|ch|ss|sh)es$'   , '\\1'     ),
    (r'(?i)(m)ovies$'         , '\\1ovie' ),
    (r'(?i)(.)ombies$'        , '\\1ombie'),
    (r'(?i)(s)eries$'         , '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'    ),
        # -f, -fe sometimes take -ves in the plural
        # (e.g., lives, wolves).
    (r"([aeo]l)ves$"          , "\\1f"    ),
    (r"([^d]ea)ves$"          , "\\1f"    ),
    (r"arves$"                , "arf"     ),
    (r"erves$"                , "erve"    ),
    (r"([nlw]i)ves$"          , "\\1fe"   ),
    (r'(?i)([lr])ves$'        , '\\1f'    ),
    (r"([aeo])ves$"           , "\\1ve"   ),
    (r'(?i)(sive)s$'          , '\\1'     ),
    (r'(?i)(tive)s$'          , '\\1'     ),
    (r'(?i)(hive)s$'          , '\\1'     ),
    (r'(?i)([^f])ves$'        , '\\1fe'   ),
    # -ses suffixes.
    (r'(?i)(^analy)ses$'      , '\\1sis'  ),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'),
    (r'(?i)(.)opses$'         , '\\1opsis'),
    (r'(?i)(.)yses$'          , '\\1ysis' ),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$', '\\1ose'),
    (r'(?i)(.)oses$'          , '\\1osis' ),
    # -a
    (r'(?i)([ti])a$'          , '\\1um'   ),
    (r'(?i)(n)ews$'           , '\\1ews'  ),
    (r'(?i)s$'                , ''        ),
]

# For performance, compile the regular expressions only once:
singular_rules = [(re.compile(r[0]), r[1]) for r in singular_rules]

singular_uninflected = set((
    "bison"      , "debris"   , "headquarters", "pincers"    , "trout"     ,
    "bream"      , "diabetes" , "herpes"      , "pliers"     , "tuna"      ,
    "breeches"   , "djinn"    , "high-jinks"  , "proceedings", "whiting"   ,
    "britches"   , "eland"    , "homework"    , "rabies"     , "wildebeest",
    "carp"       , "elk"      , "innings"     , "salmon"     ,
    "chassis"    , "flounder" , "jackanapes"  , "scissors"   ,
    "christmas"  , "gallows"  , "mackerel"    , "series"     ,
    "clippers"   , "georgia"  , "measles"     , "shears"     ,
    "cod"        , "graffiti" , "mews"        , "species"    ,
    "contretemps",              "mumps"       , "swine"      ,
    "corps"      ,              "news"        , "swiss"      ,
))
singular_uncountable = set((
    "advice"     , "equipment", "happiness"   , "luggage"    , "news"      , "software"     ,
    "bread"      , "fruit"    , "information" , "mathematics", "progress"  , "understanding",
    "butter"     , "furniture", "ketchup"     , "mayonnaise" , "research"  , "water"        ,
    "cheese"     , "garbage"  , "knowledge"   , "meat"       , "rice"      ,
    "electricity", "gravel"   , "love"        , "mustard"    , "sand"      ,
))
singular_ie = set((
    "alergie"    , "cutie"    , "hoagie"      , "newbie"     , "softie"    , "veggie"       ,
    "auntie"     , "doggie"   , "hottie"      , "nightie"    , "sortie"    , "weenie"       ,
    "beanie"     , "eyrie"    , "indie"       , "oldie"      , "stoolie"   , "yuppie"       ,
    "birdie"     , "freebie"  , "junkie"      , "^pie"       , "sweetie"   , "zombie"       ,
    "bogie"      , "goonie"   , "laddie"      , "pixie"      , "techie"    ,
    "bombie"     , "groupie"  , "laramie"     , "quickie"    , "^tie"      ,
    "collie"     , "hankie"   , "lingerie"    , "reverie"    , "toughie"   ,
    "cookie"     , "hippie"   , "meanie"      , "rookie"     , "valkyrie"  ,
))
singular_irregular = {
       "atlantes": "atlas",
        "atlases": "atlas",
           "axes": "axe",
         "beeves": "beef",
       "brethren": "brother",
       "children": "child",
        "corpora": "corpus",
       "corpuses": "corpus",
    "ephemerides": "ephemeris",
           "feet": "foot",
        "ganglia": "ganglion",
          "geese": "goose",
         "genera": "genus",
          "genii": "genie",
       "graffiti": "graffito",
         "helves": "helve",
           "kine": "cow",
         "leaves": "leaf",
         "loaves": "loaf",
            "men": "man",
      "mongooses": "mongoose",
         "monies": "money",
          "moves": "move",
         "mythoi": "mythos",
         "numena": "numen",
       "occipita": "occiput",
      "octopodes": "octopus",
          "opera": "opus",
         "opuses": "opus",
            "our": "my",
           "oxen": "ox",
          "penes": "penis",
        "penises": "penis",
         "people": "person",
          "sexes": "sex",
    "soliloquies": "soliloquy",
          "teeth": "tooth",
         "testes": "testis",
        "trilbys": "trilby",
         "turves": "turf",
            "zoa": "zoon",
}

plural_prepositions = set((
    "about"  , "before" , "during", "of"   , "till" ,
    "above"  , "behind" , "except", "off"  , "to"   ,
    "across" , "below"  , "for"   , "on"   , "under",
    "after"  , "beneath", "from"  , "onto" , "until",
    "among"  , "beside" , "in"    , "out"  , "unto" ,
    "around" , "besides", "into"  , "over" , "upon" ,
    "at"     , "between", "near"  , "since", "with" ,
    "athwart", "betwixt",
               "beyond",
               "but",
               "by"))

VERB, NOUN, ADJECTIVE, ADVERB = "VB", "NN", "JJ", "RB"

def singularize(word, pos=NOUN, custom={}):
    """ Returns the singular of a given word.
    """
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in plural_prepositions:
            return singularize(w[0], pos, custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1]) + "'s"
    w = word.lower()
    for x in singular_uninflected:
        if x.endswith(w):
            return word
    for x in singular_uncountable:
        if x.endswith(w):
            return word
    for x in singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', singular_irregular[x], word)
    for suffix, inflection in singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word
