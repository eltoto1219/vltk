import json
import os

from tqdm import tqdm

from mllib.legacy_utils import Config
from mllib.models import GeneralizedRCNN

c = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
print(c)

"""
data_dir = "/playpen1/home/avmendoz/data/"
labels = os.path.join(data_dir, "temp_gqa/gqa_labels.json")
train = os.path.join(data_dir, "temp_gqa/train/train.json")
valid = os.path.join(data_dir, "temp_gqa/train/valid.json")
testdev = os.path.join(data_dir, "temp_gqa/testdev.json")

labels = json.load(open(labels))
labels = set(labels.keys())
train = json.load(open(train))
valid = json.load(open(valid))
testdev = json.load(open(testdev))

data = []
for x in tqdm([train, valid, testdev]):
    for e in x:
        data.append(e)
data_labels = set()
for x in tqdm(data):
    label = x["label"]
    label = next(iter(label.keys()))
    data_labels.add(label)

print(f"uniq data labels {len(data_labels)}")
print(f"uniq gqa labels {len(labels)}")
print(data_labels - labels)
"""
