import json
from collections import defaultdict

import spacy
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np

nltk.download('stopwords')

data_file = "ccs_synthetic_filtered_large.json"
all_blip_data = json.load(open(data_file))
print(len(all_blip_data))
print(all_blip_data[0])

#### This section takes some time hence we save the results in a file. Subsequent runs can comment this part out.
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
stop_words = set(stopwords.words('english'))
phrase2example = defaultdict(list)
for i, doc in tqdm(enumerate(nlp.pipe([ex["caption"] for ex in all_blip_data], n_process=-1)), total=len(all_blip_data)):
    noun_phrases = [" ".join(w for w in chunk.text.lower().split() if w not in stop_words) for chunk in doc.noun_chunks]
    noun_phrases = [np for np in noun_phrases if np] #filter empty
    for np in noun_phrases:
        phrase2example[np].append(i)
with open("ccs_synthetic_filtered_large-noun_phrases.json", "w") as f:
    json.dump(phrase2example, f)
####


phrase2example = json.load(open("ccs_synthetic_filtered_large-noun_phrases.json"))
phrase2count = {k: len(v) for k,v in phrase2example.items()}

train_examples = set()
# adjustable parameters
min_ex = 10
max_ex = 30
for k, examples in phrase2example.items():
    if len(examples) < min_ex:
        continue
    if len(examples) > max_ex:
        examples = np.random.choice(examples, max_ex, replace=False)
    train_examples.update(examples)
print(len(train_examples))
train_data = [all_blip_data[i] for i in train_examples]
json.dump(train_data, open(f"ccs_synthetic_filtered_large-npfilter_min{min_ex}_max{max_ex}.json", "w"))