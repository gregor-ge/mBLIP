import json
import os.path
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
def process_conversations(max_sentences=3):

    raw_data = json.load(open("conversation_58k.json"))
    split_conversations = []
    for conversation in raw_data:
        num_qas = len(conversation["conversations"])//2

        for qa_idx in range(num_qas):
            human = conversation["conversations"][2*qa_idx]
            assert human["from"] == "human"
            context = human["value"].replace("<image>", "").replace("\n", "")
            gpt = conversation["conversations"][2*qa_idx+1]
            assert gpt["from"] == "gpt"
            label = gpt["value"]

            if label.count(".") > max_sentences:
                continue

            image_id = f"COCO_train2014_{int(conversation['id']):012d}.jpg"

            entry = {
                "context": context,
                "label": label,
                "image_id": image_id,
            }

            split_conversations.append(entry)

    # Analysis code for sequence lengths
    # print(len(split_conversations))
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xl")
    #
    # answer_by_sentence = defaultdict(list)
    # for entry in split_conversations:
    #     answer_by_sentence[entry["label"].count(".")].append(entry["label"])
    #
    # for k, v in answer_by_sentence.items():
    #     answer_by_sentence[k] = tokenizer(v)["input_ids"]
    #     answer_by_sentence[k] = [len(v) for v in answer_by_sentence[k]]
    #     print(k, len(answer_by_sentence[k]), np.percentile(answer_by_sentence[k], q=[50, 75, 90, 85, 99]))
    # print(np.percentile(sum(answer_by_sentence.values(), []), q=[50, 75, 90, 85, 99]))

    # question_types = defaultdict(list)
    # for entry in split_conversations:
    #     question_types[entry["context"][-1]].append(entry["context"])
    #
    # pass

    json.dump(split_conversations, open(f"conversation_58k_split_max3sent_en.json", "w"))

def process_detail():

    raw_data = json.load(open("detail_23k.json"))
    split_conversations = []
    for conversation in raw_data:
        num_qas = len(conversation["conversations"])//2
        assert num_qas == 1
        for qa_idx in range(num_qas):
            human = conversation["conversations"][2*qa_idx]
            assert human["from"] == "human"
            context = human["value"].replace("<image>", "").replace("\n", "")
            gpt = conversation["conversations"][2*qa_idx+1]
            assert gpt["from"] == "gpt"
            label = gpt["value"]

            image_id = f"COCO_train2014_{int(conversation['id']):012d}.jpg"

            entry = {
                "context": context,
                "label": label,
                "image_id": image_id,
            }

            split_conversations.append(entry)
    print(len(split_conversations))

    # Analysis code for sequence lengths
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xl")
    #
    # answer_by_sentence = defaultdict(list)
    # for entry in split_conversations:
    #     answer_by_sentence[entry["label"].count(".")].append(entry["label"])
    #
    # for k, v in answer_by_sentence.items():
    #     answer_by_sentence[k] = tokenizer(v)["input_ids"]
    #     answer_by_sentence[k] = [len(v) for v in answer_by_sentence[k]]
    #     print(k, len(answer_by_sentence[k]), np.percentile(answer_by_sentence[k], q=[50, 75, 90, 85, 99]))
    # print(np.percentile(sum(answer_by_sentence.values(), []), q=[50, 75, 90, 85, 99]))

    json.dump(split_conversations, open(f"detail_23k_en.json", "w"))

if __name__ == "__main__":
    process_conversations()
    process_detail()